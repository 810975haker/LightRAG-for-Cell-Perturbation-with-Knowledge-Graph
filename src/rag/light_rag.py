from __future__ import annotations

import re
from typing import Dict, List

from config import (
    IFLYTEK_API_KEY,
    IFLYTEK_LLM_MODEL,
    PROTEIN_CALIBRATION_ENABLED,
    PROTEIN_CALIBRATION_PATH,
    KG_RANK_MAX_HOPS,
    KG_RANK_MIN_SCORE,
    iflytek_enabled,
)
from src.data.protein_calibration import ProteinCalibrator
from src.knowledge_graph.graph_store import KnowledgeGraphStore
from src.rag.vector_store import TextDocument, VectorStoreManager


class LightRAG:
    def __init__(self, vector_store: VectorStoreManager | None = None, graph_store: KnowledgeGraphStore | None = None):
        self.vector_store = vector_store or VectorStoreManager()
        self.graph_store = graph_store or KnowledgeGraphStore()
        self.protein_calibrator = ProteinCalibrator(
            file_path=PROTEIN_CALIBRATION_PATH,
            enabled=PROTEIN_CALIBRATION_ENABLED,
        )
        self.llm = None
        if iflytek_enabled():
            from src.rag.iflytek_llm import IFlytekLLM

            self.llm = IFlytekLLM(api_key=IFLYTEK_API_KEY, model_name=IFLYTEK_LLM_MODEL)

    def add_documents(self, documents):
        """添加文档到向量存储"""
        local_docs: List[TextDocument] = []
        for doc in documents:
            local_docs.append(TextDocument(content=doc.page_content, metadata=doc.metadata))
        self.vector_store.add_documents(local_docs)

    def query(self, question):
        """使用Light-RAG查询"""
        inferred_cell, inferred_perturbation = self._infer_query_context(question)
        if inferred_perturbation:
            candidate_cells = [inferred_cell] if inferred_cell else ["A549_cell_001", "A549"]
            for cell_id in candidate_cells:
                ranked = self.predict_perturbation(
                    cell_id=cell_id,
                    perturbation=inferred_perturbation,
                    question=question,
                    top_k=15,
                    max_hops=KG_RANK_MAX_HOPS,
                    min_score=max(0.0, KG_RANK_MIN_SCORE * 0.25),
                )
                concrete_answer = self._render_ranked_answer(ranked)
                has_ranked = bool(
                    (ranked.get("ranked_genes", []) or [])
                    or (ranked.get("ranked_proteins", []) or [])
                    or (ranked.get("ranked_pathways", []) or [])
                )
                if concrete_answer and has_ranked:
                    if not inferred_cell:
                        concrete_answer = (
                            "问题中未显式提供细胞，已默认使用 {}。\n".format(cell_id)
                            + concrete_answer
                        )
                    return {
                        "answer": concrete_answer,
                        "sources": [],
                        "ranked_result": ranked,
                    }

        source_docs = self.vector_store.similarity_search(question, top_k=5)
        if not source_docs:
            return {
                "answer": "未命中可用检索文档，且未识别出有效细胞与扰动目标。请在问题中明确细胞名与基因（如 A549 + EGFR 抑制）。",
                "sources": [],
            }

        context = "\n".join([doc.page_content for doc in source_docs])
        prompt = (
            "请根据以下生物知识上下文回答问题，并尽量使用结构化要点。\n"
            f"上下文:\n{context}\n\n问题:\n{question}"
        )
        if self.llm is None:
            answer = self._answer_without_llm(question, source_docs)
        else:
            answer = self._generate(prompt)

        return {
            "answer": answer,
            "sources": [doc.metadata for doc in source_docs]
        }

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        chunks = re.split(r"[。！？!?\n\r]+", str(text or ""))
        return [str(item).strip() for item in chunks if str(item).strip()]

    @staticmethod
    def _question_tokens(question: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", str(question or ""))
        normalized = []
        for token in tokens:
            text = str(token).strip().lower()
            if len(text) < 2:
                continue
            normalized.append(text)
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(normalized))

    def _answer_without_llm(self, question: str, source_docs: List[TextDocument]) -> str:
        """Return a conservative answer grounded only in retrieved text when LLM is unavailable."""
        tokens = self._question_tokens(question)
        candidates: List[tuple[int, str]] = []
        for doc in source_docs or []:
            for sent in self._split_sentences(doc.page_content):
                lowered = sent.lower()
                score = sum(1 for token in tokens if token in lowered)
                if score <= 0:
                    continue
                candidates.append((score, sent))

        if not candidates:
            return (
                "当前未接入可用大模型；并且在已检索文档中未找到可直接支持该问题的证据。\n"
                "请补充更具体条件（如细胞系、基因、扰动类型），或先导入相关文档后再提问。"
            )

        candidates.sort(key=lambda item: (-int(item[0]), len(item[1])))
        best_lines = []
        seen = set()
        for _, sentence in candidates:
            key = sentence.strip()
            if key in seen:
                continue
            seen.add(key)
            best_lines.append(key)
            if len(best_lines) >= 2:
                break

        return (
            "基于当前检索到的资料，可确认的信息：\n"
            + "\n".join("- " + line for line in best_lines)
            + "\n注：以上仅基于检索片段整理，不包含外部推断；若需更完整结论，请提供更具体问题。"
        )

    def _compute_omics_boost(self, gene_nodes: List[str]) -> Dict[str, float]:
        """查询候选基因的组学数据覆盖，返回每个基因的 omics 加权因子 (0~1)。

        因子越大表示该基因有越丰富的多组学证据支持。
        仅对至少有一条组学入边的基因返回 >0 的值。
        """
        if not gene_nodes:
            return {}
        store = self.graph_store
        if store.backend != "neo4j" or store.graph is None:
            return {}

        # Neo4j 批量查询：对每个基因统计各模态的 sample 数量
        query = (
            "UNWIND $genes AS g "
            "MATCH (s:Entity)-[r]->(n:Entity {name: g}) "
            "WHERE s.name STARTS WITH 'sample::' "
            "  AND type(r) IN ['EXPRESSES','HAS_CNV','HAS_METHYLATION','HAS_PROTEIN_ABUNDANCE'] "
            "WITH g, type(r) AS modality, count(r) AS sample_cnt, avg(coalesce(r.weight, 0.5)) AS avg_w "
            "RETURN g, collect({modality: modality, cnt: sample_cnt, avg_w: avg_w}) AS omics "
        )
        try:
            rows = store.graph.run(query, genes=gene_nodes).data()
        except Exception:
            return {}

        # 全局参考值：所有候选基因中各模态最大 sample 数
        max_per_modality = {"EXPRESSES": 1, "HAS_CNV": 1, "HAS_METHYLATION": 1, "HAS_PROTEIN_ABUNDANCE": 1}
        for row in rows:
            for item in row.get("omics", []) or []:
                m, c = item["modality"], item["cnt"]
                max_per_modality[m] = max(max_per_modality.get(m, 1), c)

        # 计算每个基因的加权因子
        boost_map: Dict[str, float] = {}
        # 模态权重：表达 0.4, CNV 0.25, 甲基化 0.20, 蛋白 0.15
        modal_weights = {"EXPRESSES": 0.40, "HAS_CNV": 0.25,
                         "HAS_METHYLATION": 0.20, "HAS_PROTEIN_ABUNDANCE": 0.15}
        for row in rows:
            gene = str(row.get("g", ""))
            score = 0.0
            for item in row.get("omics", []) or []:
                m, c = item["modality"], item["cnt"]
                normalized = c / max(1, max_per_modality.get(m, 1))
                score += modal_weights.get(m, 0.1) * normalized
            boost_map[gene] = round(min(1.0, score), 4)

        return boost_map

    def predict_perturbation(
        self,
        cell_id: str,
        perturbation: str,
        question: str = "",
        target_genes=None,
        top_k: int = 10,
        max_hops: int = KG_RANK_MAX_HOPS,
        min_score: float = KG_RANK_MIN_SCORE,
    ) -> Dict:
        cell_node = f"cell::{cell_id}"
        perturb_node = self._build_perturbation_node(perturbation)
        max_hops = max(1, int(max_hops))
        min_score = float(min_score)

        neighbors = self.graph_store.get_neighbors(cell_node, depth=max_hops)
        cell_candidates = self.graph_store.get_ranked_candidates(
            node_id=cell_node,
            max_hops=max_hops,
            min_score=min_score,
            include_prefixes=("gene::", "pathway::", "protein::"),
        )
        perturb_candidates = self.graph_store.get_ranked_candidates(
            node_id=perturb_node,
            max_hops=max_hops,
            min_score=min_score * 0.6,
            include_prefixes=("gene::", "pathway::", "protein::"),
        )
        perturb_direct_candidates = self.graph_store.get_ranked_candidates(
            node_id=perturb_node,
            max_hops=1,
            min_score=0.0,
            include_prefixes=("gene::", "protein::", "pathway::"),
        )
        ranked_candidates = self._merge_ranked_candidates(cell_candidates, perturb_candidates)

        # If cell context is sparse, keep direct perturbation 1-hop evidence as fallback pool.
        if not ranked_candidates and perturb_direct_candidates:
            ranked_candidates = sorted(
                [
                    {
                        **item,
                        "score": max(
                            float(item.get("score", 0.0) or 0.0),
                            float(item.get("raw_signal", 0.0) or 0.0),
                            0.005,
                        ),
                        "from_perturbation": True,
                    }
                    for item in perturb_direct_candidates
                ],
                key=lambda item: (
                    -float(item.get("score", 0.0) or 0.0),
                    int(item.get("hops", 1) or 1),
                    str(item.get("node", "")),
                ),
            )

        ranked_genes = [item for item in ranked_candidates if str(item.get("node", "")).startswith("gene::")]
        ranked_pathways = [
            item
            for item in ranked_candidates
            if self._is_biological_pathway_node(str(item.get("node", "")))
        ]
        ranked_proteins = [item for item in ranked_candidates if str(item.get("node", "")).startswith("protein::")]

        if not ranked_pathways and ranked_genes:
            pathway_pool = {}
            for gene_hit in ranked_genes[: min(12, max(4, top_k))]:
                gene_node = str(gene_hit.get("node", "")).strip()
                gene_score = float(gene_hit.get("score", 0.0) or 0.0)
                if not gene_node.startswith("gene::") or gene_score <= 0.0:
                    continue
                linked_pathways = self.graph_store.get_gene_linked_pathways(
                    gene_node=gene_node,
                    max_pathways=20,
                    min_edge_signal=max(0.005, min_score / 2.0),
                )
                for item in linked_pathways:
                    pathway_node = str(item.get("pathway", "")).strip()
                    if not self._is_biological_pathway_node(pathway_node):
                        continue
                    candidate_score = round(gene_score * float(item.get("signal", 0.0) or 0.0) * 0.95, 6)
                    prev = pathway_pool.get(pathway_node)
                    candidate = {
                        "node": pathway_node,
                        "type": "pathway",
                        "hops": int(gene_hit.get("hops", 1) or 1) + 1,
                        "raw_signal": round(float(item.get("signal", 0.0) or 0.0), 6),
                        "score": candidate_score,
                        "from_gene": gene_node,
                    }
                    if prev is None or float(candidate["score"]) > float(prev.get("score", 0.0) or 0.0):
                        pathway_pool[pathway_node] = candidate
            ranked_pathways = sorted(
                pathway_pool.values(),
                key=lambda item: (
                    -float(item.get("score", 0.0) or 0.0),
                    int(item.get("hops", 1) or 1),
                    str(item.get("node", "")),
                ),
            )

        direct_perturb_gene_nodes = {
            str(item.get("node", ""))
            for item in perturb_direct_candidates
            if str(item.get("node", "")).startswith("gene::")
        }
        direct_perturb_gene_labels = {
            self.graph_store.resolve_gene_label(node_name)
            for node_name in direct_perturb_gene_nodes
        }
        direct_perturb_gene_labels = {g for g in direct_perturb_gene_labels if g}

        if target_genes:
            target_set = {gene.lower() for gene in target_genes}
            ranked_genes = [
                hit
                for hit in ranked_genes
                if self.graph_store.resolve_gene_label(str(hit.get("node", "")).strip()).lower() in target_set
            ]

        # Prefer genes with direct perturbation evidence, but avoid over-pruning to only seed genes.
        if direct_perturb_gene_labels:
            direct_hits = []
            indirect_hits = []
            for hit in ranked_genes:
                label = self.graph_store.resolve_gene_label(str(hit.get("node", "")).strip())
                if label in direct_perturb_gene_labels:
                    boosted = {**hit, "score": float(hit.get("score", 0.0) or 0.0) * 1.35, "from_perturbation": True}
                    direct_hits.append(boosted)
                else:
                    indirect_hits.append(hit)

            # Only enforce hard filtering when direct evidence is sufficiently rich.
            if len(direct_hits) >= max(8, top_k):
                ranked_genes = direct_hits
            else:
                ranked_genes = sorted(
                    direct_hits + indirect_hits,
                    key=lambda item: (
                        -float(item.get("score", 0.0) or 0.0),
                        int(item.get("hops", 1) or 1),
                        str(item.get("node", "")),
                    ),
                )

        # ── 组学数据加权：查询候选基因的 omics 边，给有组学证据的基因加分 ──
        omics_boost = self._compute_omics_boost(
            [str(hit.get("node", "")).strip() for hit in ranked_genes
             if str(hit.get("node", "")).startswith("gene::")]
        )

        collapsed_genes = {}
        for hit in ranked_genes:
            raw_node = str(hit.get("node", "")).strip()
            display_gene = self.graph_store.resolve_gene_label(raw_node)
            if not display_gene:
                continue
            current_score = float(hit.get("score", 0.0) or 0.0)
            if str(display_gene).isdigit() and current_score < max(min_score * 2.0, 0.03):
                continue
            # 组学加权：有组学数据的基因得分上浮
            omics_factor = omics_boost.get(raw_node, 0.0)
            adjusted_score = current_score * (1.0 + 0.35 * omics_factor)
            hops = int(hit.get("hops", 1) or 1)
            prev = collapsed_genes.get(display_gene)
            candidate = {
                **hit,
                "node": "gene::{}".format(display_gene),
                "resolved_from": raw_node,
                "score": adjusted_score,
                "omics_boost": round(omics_factor, 4),
                "hops": hops,
            }
            if prev is None or current_score > float(prev.get("score", 0.0) or 0.0) or (
                current_score == float(prev.get("score", 0.0) or 0.0) and hops < int(prev.get("hops", 99) or 99)
            ):
                collapsed_genes[display_gene] = candidate

        ranked_genes = sorted(
            collapsed_genes.values(),
            key=lambda item: (-float(item.get("score", 0.0) or 0.0), int(item.get("hops", 1) or 1), str(item.get("node", "")),),
        )

        # Keep symbol-like genes ahead of numeric/Ensembl IDs for user-facing outputs.
        symbol_like_hits = []
        id_like_hits = []
        for hit in ranked_genes:
            gene_name = str(hit.get("node", "")).replace("gene::", "")
            if self._is_id_like_gene_label(gene_name):
                id_like_hits.append(hit)
            else:
                symbol_like_hits.append(hit)
        ranked_genes = symbol_like_hits + id_like_hits

        top_n = max(1, int(top_k))

        # If direct neighborhood is sparse, backfill genes connected to top pathways.
        desired_pool = max(top_n, min(25, top_n * 2))
        if len(ranked_genes) < desired_pool and ranked_pathways:
            supplement = []
            for pathway_hit in ranked_pathways[: max(3, top_n)]:
                pathway_node = str(pathway_hit.get("node", ""))
                pathway_score = float(pathway_hit.get("score", 0.0) or 0.0)
                if pathway_score <= 0.0:
                    continue
                linked_genes = self.graph_store.get_pathway_linked_genes(pathway_node, max_genes=40, min_edge_signal=min_score / 2.0)
                for link in linked_genes:
                    raw_gene_node = str(link.get("gene", "")).strip()
                    if not raw_gene_node.startswith("gene::"):
                        continue
                    display_gene = self.graph_store.resolve_gene_label(raw_gene_node)
                    if not display_gene:
                        continue
                    link_signal = float(link.get("signal", 0.0) or 0.0)
                    candidate_score = round(pathway_score * link_signal * 0.9, 6)
                    supplement.append(
                        {
                            "node": "gene::{}".format(display_gene),
                            "resolved_from": raw_gene_node,
                            "type": "gene",
                            "hops": int(pathway_hit.get("hops", 2) or 2) + 1,
                            "raw_signal": round(link_signal, 6),
                            "score": candidate_score,
                            "support_pathway": pathway_node,
                        }
                    )

            for hit in supplement:
                display_gene = str(hit.get("node", "")).replace("gene::", "")
                if not display_gene:
                    continue
                prev = collapsed_genes.get(display_gene)
                if prev is None or float(hit.get("score", 0.0) or 0.0) > float(prev.get("score", 0.0) or 0.0):
                    collapsed_genes[display_gene] = hit

            ranked_genes = sorted(
                collapsed_genes.values(),
                key=lambda item: (
                    -float(item.get("score", 0.0) or 0.0),
                    int(item.get("hops", 1) or 1),
                    str(item.get("node", "")),
                ),
            )

        # Extra fallback: infer biological pathways by 2-hop expansion from top genes.
        if not ranked_pathways and ranked_genes:
            pathway_pool = {}
            for gene_hit in ranked_genes[: min(top_n, 8)]:
                gene_node = str(gene_hit.get("node", "")).strip()
                gene_score = float(gene_hit.get("score", 0.0) or 0.0)
                if not gene_node.startswith("gene::") or gene_score <= 0.0:
                    continue
                candidates = self.graph_store.get_ranked_candidates(
                    node_id=gene_node,
                    max_hops=2,
                    min_score=max(0.001, min_score * 0.25),
                    include_prefixes=("pathway::",),
                )
                for cand in candidates:
                    pathway_node = str(cand.get("node", "")).strip()
                    if not self._is_biological_pathway_node(pathway_node):
                        continue
                    combined_score = round(gene_score * float(cand.get("score", 0.0) or 0.0) * 0.85, 6)
                    prev = pathway_pool.get(pathway_node)
                    payload = {
                        "node": pathway_node,
                        "type": "pathway",
                        "hops": int(gene_hit.get("hops", 1) or 1) + int(cand.get("hops", 2) or 2),
                        "raw_signal": round(float(cand.get("raw_signal", 0.0) or 0.0), 6),
                        "score": combined_score,
                        "from_gene": gene_node,
                    }
                    if prev is None or float(payload.get("score", 0.0) or 0.0) > float(prev.get("score", 0.0) or 0.0):
                        pathway_pool[pathway_node] = payload
            ranked_pathways = sorted(
                pathway_pool.values(),
                key=lambda item: (
                    -float(item.get("score", 0.0) or 0.0),
                    int(item.get("hops", 1) or 1),
                    str(item.get("node", "")),
                ),
            )

        selected_gene_hits = ranked_genes[:top_n]
        non_seed_hits = [hit for hit in ranked_genes if not self.graph_store.is_seed_gene(str(hit.get("node", "")))]
        if non_seed_hits and top_n >= 4:
            min_non_seed = min(max(1, top_n // 3), len(non_seed_hits))
            chosen = list(non_seed_hits[:min_non_seed])
            chosen_nodes = {str(item.get("node", "")) for item in chosen}
            for hit in ranked_genes:
                node_name = str(hit.get("node", ""))
                if node_name in chosen_nodes:
                    continue
                chosen.append(hit)
                if len(chosen) >= top_n:
                    break
            selected_gene_hits = chosen[:top_n]
        selected_pathway_hits = ranked_pathways[:top_n]
        selected_protein_hits = ranked_proteins[:top_n]
        display_gene_hits = [
            hit for hit in selected_gene_hits
            if not self._is_id_like_gene_label(str(hit.get("node", "")).replace("gene::", ""))
        ]
        if not display_gene_hits:
            display_gene_hits = selected_gene_hits

        top_genes = [str(hit.get("node", "")).replace("gene::", "") for hit in display_gene_hits]
        top_pathways = [str(hit.get("node", "")).replace("pathway::", "") for hit in selected_pathway_hits]
        top_proteins = [str(hit.get("node", "")).replace("protein::", "") for hit in selected_protein_hits]

        sign = -1.0 if any(token in perturbation.lower() for token in ["ko", "knockout", "inhibit", "kd"]) else 1.0
        gene_expression_changes = []
        for hit in display_gene_hits:
            gene = str(hit.get("node", "")).replace("gene::", "")
            base_score = float(hit.get("score", 0.0) or 0.0)
            relation_sign = float(hit.get("direction_sign", 1.0) or 1.0)
            delta = round(sign * relation_sign * base_score * 0.2, 4)
            direction = "增强" if delta > 0 else ("抑制" if delta < 0 else "不确定")
            gene_expression_changes.append(
                {
                    "gene": gene,
                    "predicted_delta": delta,
                    "direction": direction,
                    "hops": int(hit.get("hops", 1) or 1),
                    "weight": round(base_score, 4),
                    "confidence": 1.0,
                    "signal": round(base_score, 4),
                }
            )

        protein_abundance_changes = []
        for hit in selected_protein_hits:
            protein = str(hit.get("node", "")).replace("protein::", "")
            base_score = float(hit.get("score", 0.0) or 0.0)
            relation_sign = float(hit.get("direction_sign", 1.0) or 1.0)
            raw_delta = sign * relation_sign * base_score * 0.15
            calibration = self.protein_calibrator.calibrate_delta(protein, raw_delta)
            delta = round(float(calibration.get("delta", raw_delta) or raw_delta), 4)
            direction = "增强" if delta > 0 else ("抑制" if delta < 0 else "不确定")
            protein_abundance_changes.append(
                {
                    "protein": protein,
                    "predicted_delta": delta,
                    "direction": direction,
                    "hops": int(hit.get("hops", 1) or 1),
                    "weight": round(base_score, 4),
                    "confidence": 1.0,
                    "signal": round(base_score, 4),
                    "calibration_factor": float(calibration.get("factor", 1.0) or 1.0),
                    "calibration_source": str(calibration.get("source", "")),
                }
            )

        query_text = f"cell {cell_id} perturbation {perturbation} genes {' '.join(top_genes)} {question}".strip()
        docs = self.vector_store.similarity_search(query_text, top_k=5)
        doc_context = "\n".join(doc.page_content for doc in docs)

        prompt = (
            "你是细胞扰动分析助手。请根据知识图谱邻域和检索文档，"
            "预测该扰动对细胞的主要影响路径，返回简洁中文说明。\n"
            f"Cell: {cell_id}\nPerturbation: {perturbation}\n"
            f"Top Genes: {', '.join(top_genes) if top_genes else 'N/A'}\n"
            f"Top Proteins: {', '.join(top_proteins) if top_proteins else 'N/A'}\n"
            f"Top Pathways: {', '.join(top_pathways) if top_pathways else 'N/A'}\n"
            f"Graph Neighbors: {neighbors[:20]}\n"
            f"Documents: {doc_context[:2500]}\n"
            f"Extra Question: {question or 'N/A'}"
        )

        explanation = self._generate(prompt)
        confidence = min(0.99, 0.4 + 0.04 * len(gene_expression_changes) + 0.02 * len(docs))
        return {
            "task_type": "gene_expression_regression",
            "cell_id": cell_id,
            "perturbation": perturbation,
            "top_n": top_n,
            "top_genes": top_genes,
            "top_proteins": top_proteins,
            "top_pathways": top_pathways,
            "ranked_genes": selected_gene_hits,
            "ranked_proteins": selected_protein_hits,
            "ranked_pathways": selected_pathway_hits,
            "ranking_meta": {
                "max_hops": max_hops,
                "min_score": min_score,
                "cell_anchor": cell_node,
                "perturb_anchor": perturb_node,
                "protein_calibration_enabled": bool(PROTEIN_CALIBRATION_ENABLED),
                "protein_calibration_path": PROTEIN_CALIBRATION_PATH,
            },
            "gene_expression_changes": gene_expression_changes,
            "protein_abundance_changes": protein_abundance_changes,
            "predicted_effect": explanation,
            "confidence": round(confidence, 3),
            "evidence_count": len(neighbors) + len(docs),
        }

    def _infer_query_context(self, question: str) -> tuple[str, str]:
        text = str(question or "")
        cell_match = re.search(r"(?<![A-Za-z0-9_])([A-Za-z]+\d{2,}|cell[A-Za-z0-9_\-]+)(?![A-Za-z0-9_])", text)
        cell_id = cell_match.group(1) if cell_match else ""
        gene = ""
        gene_candidates = re.findall(r"(?<![A-Za-z0-9_])([A-Z]{2,}[A-Z0-9]{0,8})(?![A-Za-z0-9_])", text)
        for candidate in gene_candidates:
            if cell_id and candidate.upper() == str(cell_id).upper():
                continue
            gene = candidate
            break
        if not gene:
            return "", ""

        lower = text.lower()
        if any(token in lower for token in ["抑制", "敲除", "ko", "inhibit", "knockout", "kd"]):
            return cell_id, "KO_{}".format(gene)
        if any(token in lower for token in ["过表达", "激活", "oe", "activate", "overexpression"]):
            return cell_id, "OE_{}".format(gene)
        return cell_id, "Perturb_{}".format(gene)

    def _build_perturbation_node(self, perturbation: str) -> str:
        text = str(perturbation or "").strip()
        if not text:
            return "pathway::perturbation::UNKNOWN"
        normalized = text.upper().replace("-", "_").replace(" ", "_")
        if normalized.startswith("PATHWAY::PERTURBATION::"):
            return normalized.lower().replace("pathway::perturbation::", "pathway::perturbation::")
        return "pathway::perturbation::{}".format(normalized)

    def _is_biological_pathway_node(self, node_name: str) -> bool:
        text = str(node_name or "")
        if not text.startswith("pathway::"):
            return False
        if text.startswith("pathway::perturbation::"):
            return False
        if text.startswith("pathway::perturb_method::"):
            return False
        return True

    @staticmethod
    def _is_id_like_gene_label(gene_label: str) -> bool:
        text = str(gene_label or "").strip()
        if not text:
            return True
        upper = text.upper()
        return text.isdigit() or upper.startswith("ENSG") or upper.startswith("ENSMUSG")

    def _merge_ranked_candidates(self, primary: List[Dict], secondary: List[Dict]) -> List[Dict]:
        merged: Dict[str, Dict] = {}
        for item in primary or []:
            node = str(item.get("node", "")).strip()
            if not node:
                continue
            merged[node] = {**item, "score": float(item.get("score", 0.0) or 0.0), "from_perturbation": False}

        for item in secondary or []:
            node = str(item.get("node", "")).strip()
            if not node:
                continue
            base = float(item.get("score", 0.0) or 0.0)
            hop_bonus = 1.6 if int(item.get("hops", 99) or 99) == 1 else 1.35
            boosted = base * hop_bonus
            prev = merged.get(node)
            candidate = {**item, "score": boosted, "from_perturbation": True}
            if prev is None or boosted > float(prev.get("score", 0.0) or 0.0):
                merged[node] = candidate

        return sorted(
            merged.values(),
            key=lambda item: (-float(item.get("score", 0.0) or 0.0), int(item.get("hops", 1) or 1), str(item.get("node", ""))),
        )

    def _render_ranked_answer(self, ranked: Dict) -> str:
        genes = ranked.get("ranked_genes", []) or []
        proteins = ranked.get("ranked_proteins", []) or []
        pathways = ranked.get("ranked_pathways", []) or []
        if not genes and not proteins and not pathways:
            stats = self.graph_store.stats()
            return (
                "当前图谱中未检索到与该扰动可达的基因/蛋白/通路候选。"
                "请先确认图谱已导入且包含该细胞与扰动相关边。"
                "(entities={}, relations={})"
            ).format(stats.get("entities", 0), stats.get("relations", 0))

        gene_lines = []
        for item in (ranked.get("gene_expression_changes", []) or [])[:5]:
            gene_name = str(item.get("gene", ""))
            if self._is_id_like_gene_label(gene_name):
                continue
            gene_lines.append(
                "{}({}, delta={}, hops={})".format(
                    gene_name,
                    item.get("direction", "不确定"),
                    round(float(item.get("predicted_delta", 0.0) or 0.0), 4),
                    int(item.get("hops", 1) or 1),
                )
            )

        protein_lines = []
        for item in (ranked.get("protein_abundance_changes", []) or [])[:5]:
            protein_name = str(item.get("protein", ""))
            protein_lines.append(
                "{}({}, delta={}, hops={})".format(
                    protein_name,
                    item.get("direction", "不确定"),
                    round(float(item.get("predicted_delta", 0.0) or 0.0), 4),
                    int(item.get("hops", 1) or 1),
                )
            )

        pathway_lines = []
        for item in pathways[:5]:
            pathway_name = str(item.get("node", "")).replace("pathway::", "")
            pathway_lines.append(
                "{}(score={}, hops={})".format(
                    pathway_name,
                    round(float(item.get("score", 0.0) or 0.0), 4),
                    int(item.get("hops", 1) or 1),
                )
            )

        return (
            "根据图谱距离与相关度阈值过滤后的结果：\n"
            "- 主要受影响基因表达（按相关度降序）: {}\n"
            "- 主要受影响蛋白丰度（按相关度降序）: {}\n"
            "- 主要受影响通路（按相关度降序）: {}\n"
            "- 说明: 方向由扰动类型(如KO/OE)与关系符号共同决定，已自动舍弃低相关度或超过跳数阈值的候选。"
        ).format(
            "; ".join(gene_lines) if gene_lines else "无",
            "; ".join(protein_lines) if protein_lines else "无",
            "; ".join(pathway_lines) if pathway_lines else "无",
        )

    def _generate(self, prompt: str) -> str:
        if self.llm is not None:
            return self.llm._call(prompt)
        return (
            "当前未接入可用大模型，已切换到图谱规则回答模式。"
            "建议直接使用 /api/predict 获取可排序的基因/蛋白/通路结果。"
        )
