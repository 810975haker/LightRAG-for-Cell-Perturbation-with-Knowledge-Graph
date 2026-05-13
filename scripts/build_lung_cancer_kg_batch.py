from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.knowledge_graph.graph_store import KnowledgeGraphStore
from src.knowledge_graph.lung_cancer_etl import (
    KGTriple,
    _clamp01,
    build_evidence,
    canonical_relation,
    canonicalize_entity_id,
    deduplicate_triples,
    infer_effect_sign_from_relation,
    infer_node_type,
    is_seed_entity,
    load_seed_genes,
    normalize_entity,
    _perturbation_node,
    _perturbation_method_node,
)
from src.knowledge_graph.omics_etl import (
    OmicsMatrixSpec,
    build_omics_triples,
    iter_omics_triples_chunked,
    load_omics_manifest,
)

# 子批次大小：控制在内存中同时存在的三元组数量上限
SUB_BATCH_SIZE = 5000
# SQLite dedup 写入批次
SQLITE_BATCH = 1000


# ======================================================================
# 磁盘 SQLite 去重（跨源 + 跨子批次）
# ======================================================================

def _open_dedup_db(work_dir: Path) -> sqlite3.Connection:
    db_path = work_dir / "dedup_cache.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA cache_size = -8000")  # 8 MB page cache
    conn.execute(
        """CREATE TABLE IF NOT EXISTS dedup (
            head      TEXT NOT NULL,
            relation  TEXT NOT NULL,
            tail      TEXT NOT NULL,
            source    TEXT NOT NULL,
            version   TEXT NOT NULL,
            PRIMARY KEY (head, relation, tail, source, version)
        )"""
    )
    return conn


def _dedup_batch(conn: sqlite3.Connection, triples: List[KGTriple]) -> List[KGTriple]:
    """返回未在 dedup 表中的三元组（新边），并写入 dedup 表。

    使用临时表避免 SQLite 的 999 变量上限问题。"""
    clean = [(t.head, canonical_relation(t.relation), t.tail, t.source, t.version) for t in triples]
    if not clean:
        return []

    # 临时表：当前批次
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _b (h,r,t,s,v)")
    conn.execute("DELETE FROM _b")
    conn.executemany("INSERT INTO _b VALUES (?,?,?,?,?)", clean)

    # 找出不在 dedup 中的行
    cur = conn.execute("""
        SELECT b.h, b.r, b.t, b.s, b.v FROM _b b
        WHERE NOT EXISTS (
            SELECT 1 FROM dedup d
            WHERE d.head=b.h AND d.relation=b.r AND d.tail=b.t
              AND d.source=b.s AND d.version=b.v
        )
    """)
    new_keys = {tuple(row) for row in cur.fetchall()}

    # 写入 dedup
    if new_keys:
        conn.executemany(
            "INSERT OR IGNORE INTO dedup VALUES (?,?,?,?,?)",
            [list(r) for r in new_keys])

    # 映射回原始三元组（按首次出现）
    seen: Set[tuple] = set()
    new_triples: List[KGTriple] = []
    for t, key in zip(triples, clean):
        if key in new_keys and key not in seen:
            seen.add(key)
            new_triples.append(t)

    return new_triples


# ======================================================================
# 源 → 子批次迭代器
# ======================================================================

def _chunked(seq: List, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ======================================================================
# 源级迭代（每个源返回一个完整列表，外层做子批次切分）
# ======================================================================

def iter_source_triples(
    manifest: Dict,
    max_rows_per_source: int = 200000,
    min_string_score: int = 700,
) -> Iterator[tuple]:
    """逐源 yield (name, source, version, List[KGTriple])。"""
    from src.knowledge_graph.lung_cancer_etl import (
        parse_biogrid_tab3,
        parse_encode_eclip_report,
        parse_ensembl_gtf,
        parse_kegg_gene_pathway,
        parse_ncbi_gene_info,
        parse_npinter,
        parse_omnipath_signed_relations,
        parse_pathway_commons_sif,
        parse_reactome_pathway_relations,
        parse_reactome_pathways,
        parse_reactome_uniprot_pathway,
        parse_signor_signed_relations,
        parse_string_ppi,
        parse_string_protein_aliases,
        parse_string_protein_info,
        parse_wikipathways_query_json,
    )

    for item in manifest.get("downloads", []):
        if item.get("status") != "ok":
            continue
        path = Path(item.get("path", ""))
        if not path.exists():
            continue
        name = item.get("name", "")
        parser_name = item.get("parser", "")
        source = item.get("source", "unknown")
        version = item.get("version", "unknown")
        query_gene = item.get("query_gene", "")

        triples: List[KGTriple] = []
        try:
            if parser_name in ("reactome_pathway_relations",) or name == "reactome_pathway_relations":
                triples = parse_reactome_pathway_relations(path, source=source, version=version)
            elif parser_name in ("reactome_pathways",) or name == "reactome_pathways":
                triples = parse_reactome_pathways(path, source=source, version=version)
            elif parser_name in ("reactome_uniprot_pathway",) or name == "reactome_uniprot_pathway":
                triples = parse_reactome_uniprot_pathway(
                    path, source=source, version=version, max_rows=max_rows_per_source
                )
            elif parser_name in ("string_ppi",) or name == "string_human_ppi":
                triples = parse_string_ppi(
                    path, source=source, version=version,
                    min_score=min_string_score, max_rows=max_rows_per_source,
                )
            elif parser_name in ("pathway_commons_hgnc_sif",) or name == "pathway_commons_hgnc_sif":
                triples = parse_pathway_commons_sif(
                    path, source=source, version=version, max_rows=max_rows_per_source,
                )
            elif parser_name == "string_protein_info":
                triples = parse_string_protein_info(
                    path, source=source, version=version, max_rows=max_rows_per_source,
                )
            elif parser_name == "string_protein_aliases":
                triples = parse_string_protein_aliases(
                    path, source=source, version=version, max_rows=max_rows_per_source,
                )
            elif parser_name in ("npinter",) or name == "npinter_rna_interaction":
                triples = parse_npinter(
                    path, source=source, version=version, max_rows=max_rows_per_source,
                )
            elif parser_name in ("kegg_gene_pathway",) or name.startswith("kegg_gene_pathway_"):
                triples = parse_kegg_gene_pathway(
                    path, source=source, version=version,
                    query_gene=query_gene, max_rows=max_rows_per_source,
                )
            elif parser_name in ("wikipathways_query_json",) or name.startswith("wikipathways_gene_query_"):
                triples = parse_wikipathways_query_json(
                    path, source=source, version=version,
                    query_gene=query_gene, max_rows=max_rows_per_source,
                )
            elif parser_name == "encode_eclip_report":
                triples = parse_encode_eclip_report(
                    path, source=source, version=version, max_rows=max_rows_per_source,
                )
            elif parser_name == "biogrid_tab3":
                triples = parse_biogrid_tab3(
                    path, source=source, version=version, max_rows=max_rows_per_source,
                )
            elif parser_name == "signor_signed_relations":
                triples = parse_signor_signed_relations(
                    path, source=source, version=version, max_rows=max_rows_per_source,
                )
            elif parser_name == "omnipath_signed_relations":
                triples = parse_omnipath_signed_relations(
                    path, source=source, version=version, max_rows=max_rows_per_source,
                )
            elif parser_name == "ncbi_gene_info":
                triples = parse_ncbi_gene_info(
                    path, source=source, version=version, max_rows=max_rows_per_source,
                )
            elif parser_name == "ensembl_gtf":
                triples = parse_ensembl_gtf(
                    path, source=source, version=version, max_rows=max_rows_per_source,
                )
            else:
                print(f"  [SKIP] unknown parser: {parser_name} ({name})")
                continue
        except Exception:
            print(f"  [ERROR] parse failed: {name} ({source}): {traceback.format_exc()}")
            continue

        yield (name, source, version, triples)


# ======================================================================
# 行规范化（与原始 build_nx_graph 保持完全一致的 logic）
# ======================================================================

def iter_normalized_rows(
    triples: List[KGTriple],
    seed_genes: Set[str],
    min_edge_weight: float = 0.0,
    min_edge_confidence: float = 0.0,
) -> Iterator[Dict[str, Any]]:
    """规范化三元组为 Neo4j 行，自动跳过自环和低质量边。"""
    gene_symbols = {str(g).upper() for g in (seed_genes or set()) if str(g).strip()}
    for triple in triples:
        head = canonicalize_entity_id(triple.head, gene_symbols=gene_symbols)
        tail = canonicalize_entity_id(triple.tail, gene_symbols=gene_symbols)
        relation = canonical_relation(triple.relation)
        if not head or not tail:
            continue
        # 自环过滤
        if head == tail:
            continue

        weight = float(getattr(triple, "weight", 1.0))
        confidence = float(getattr(triple, "confidence", 0.5))
        # 全局质量阈值
        if min_edge_weight > 0.0 and weight < min_edge_weight:
            continue
        if min_edge_confidence > 0.0 and confidence < min_edge_confidence:
            continue

        head_type = infer_node_type(head)
        tail_type = infer_node_type(tail)

        # Normalize cell context
        if head_type == "Cell" and tail_type in {"gene", "protein", "pathway"}:
            head, tail = tail, head
            head_type, tail_type = tail_type, "Cell"
            relation = "belongs_to"
        elif tail_type == "Cell" and head_type in {"gene", "protein", "pathway"}:
            relation = "belongs_to"

        rel = _normalize_relation_by_types(relation, head_type, tail_type)

        if str(triple.relation) == "pathway_name" and head.startswith("pathway::"):
            pname = str(tail or "")
            if "::" in pname:
                pname = pname.split("::", 1)[-1]
            pname = pname.replace("_", " ").strip()
            if pname:
                yield {"node_update": True, "node_name": head, "pathway_name": pname}
            continue

        yield {
            "head": head, "tail": tail, "relation": rel,
            "source": triple.source, "version": triple.version,
            "evidence": triple.evidence,
            "weight": weight,
            "confidence": confidence,
            "effect_sign": float(
                getattr(triple, "effect_sign", 0.0)
                if abs(float(getattr(triple, "effect_sign", 0.0) or 0.0)) > 0.001
                else infer_effect_sign_from_relation(triple.relation)
            ),
            "head_type": head_type, "tail_type": tail_type,
            "head_is_seed": is_seed_entity(head, seed_genes),
            "tail_is_seed": is_seed_entity(tail, seed_genes),
        }


def _normalize_relation_by_types(relation: str, head_type: str, tail_type: str) -> str:
    """与 build_lung_cancer_kg.py:_normalize_relation_by_types 完全一致。"""
    rel = str(relation or "associated_with")
    if rel == "belongs_to":
        return "belongs_to" if tail_type == "Cell" and head_type in {"gene", "protein", "pathway", "sample", "mirna"} else "associated_with"
    if rel == "participates":
        return "participates" if head_type in {"gene", "protein"} and tail_type == "pathway" else "associated_with"
    if rel in {"activates", "inhibits"}:
        return rel if head_type in {"gene", "protein"} and tail_type in {"gene", "protein"} else "associated_with"
    if rel in {"expresses", "has_protein_abundance", "has_cnv", "has_methylation"}:
        return rel if head_type in {"sample", "Cell"} and tail_type in {"gene", "protein", "mirna"} else "associated_with"
    if rel == "has_condition":
        return rel if head_type in {"sample", "Cell"} and tail_type == "condition" else "associated_with"
    if rel == "regulates":
        return rel if head_type == "mirna" and tail_type in {"gene", "protein"} else "associated_with"
    # 扰动推导产生的关系类型（affects_*, has_perturbation_method, targets_gene 等）
    if rel.startswith("affects_") or rel.startswith("has_perturbation") or rel.startswith("targets_"):
        return rel
    if rel == "associated_with":
        return "associated_with"
    return "associated_with"


# ======================================================================
# 扰动种子收集（稀疏：只收集 seed gene 邻居 + gene↔pathway 边）
# ======================================================================

def _collect_perturbation_seeds(
    triple: KGTriple,
    seed_genes: Set[str],
    gene_neighbors: Dict[str, List[KGTriple]],
    gene_to_pathways: Dict[str, Set[str]],
    pathway_to_genes: Dict[str, Set[str]],
    gene_pathway_strength: Dict[tuple, float],
) -> None:
    head = canonicalize_entity_id(triple.head)
    tail = canonicalize_entity_id(triple.tail)

    if head.startswith("gene::"):
        g = head.split("::", 1)[-1]
        if g in seed_genes:
            gene_neighbors.setdefault(g, []).append(triple)
    if tail.startswith("gene::"):
        g = tail.split("::", 1)[-1]
        if g in seed_genes:
            gene_neighbors.setdefault(g, []).append(triple)

    rel = canonical_relation(triple.relation)
    if rel == "participates":
        if head.startswith("gene::") and tail.startswith("pathway::"):
            g = head.split("::", 1)[-1]
            gene_to_pathways.setdefault(g, set()).add(tail)
            pathway_to_genes.setdefault(tail, set()).add(head)
            s = _clamp01((float(getattr(triple, "weight", 0.5)) + float(getattr(triple, "confidence", 0.5))) / 2.0)
            gene_pathway_strength[(g, tail)] = max(s, gene_pathway_strength.get((g, tail), 0.0))
        elif tail.startswith("gene::") and head.startswith("pathway::"):
            g = tail.split("::", 1)[-1]
            gene_to_pathways.setdefault(g, set()).add(head)
            pathway_to_genes.setdefault(head, set()).add(tail)
            s = _clamp01((float(getattr(triple, "weight", 0.5)) + float(getattr(triple, "confidence", 0.5))) / 2.0)
            gene_pathway_strength[(g, head)] = max(s, gene_pathway_strength.get((g, head), 0.0))


# ======================================================================
# 扰动推导（从稀疏数据，逻辑与原 derive_perturbation_triples 一致）
# ======================================================================

def derive_perturbation_from_collected(
    gene_neighbors: Dict[str, List[KGTriple]],
    gene_to_pathways: Dict[str, Set[str]],
    pathway_to_genes: Dict[str, Set[str]],
    gene_pathway_strength: Dict[tuple, float],
    seed_genes: Set[str],
    methods: Optional[List[str]] = None,
    max_affected_per_seed: int = 120,
    max_shared_pathway_2hop_per_seed: int = 80,
    min_pathway_edge_strength: float = 0.45,
    min_shared_pathway_score: float = 0.5,
    shared_pathway_keep_best_per_gene: bool = True,
) -> List[KGTriple]:
    seeds = {normalize_entity(g).upper() for g in (seed_genes or set()) if normalize_entity(g)}
    if not seeds:
        return []

    perturb_methods = [m.upper() for m in (methods or ["KO", "KD", "OE", "CRISPRI", "CRISPRA", "RNAI", "INHIBIT"])]
    gene_symbols = {g.upper() for g in seeds}
    derived: List[KGTriple] = []

    for seed in sorted(seeds):
        linked = gene_neighbors.get(seed, [])
        linked_sorted = sorted(linked, key=lambda t: float(getattr(t, "weight", 0.0) or 0.0), reverse=True)

        for method in perturb_methods:
            pert_node = _perturbation_node(method, seed)
            method_node = _perturbation_method_node(method)

            derived.append(KGTriple(
                head=pert_node, relation="has_perturbation_method", tail=method_node,
                source="DerivedPerturbation", version="v1",
                evidence=build_evidence(raw=f"{pert_node}->{method_node}",
                                        structured={"seed_gene": seed, "method": method}),
                weight=0.95, confidence=0.9))
            derived.append(KGTriple(
                head=pert_node, relation="targets_gene", tail=f"gene::{seed}",
                source="DerivedPerturbation", version="v1",
                evidence=build_evidence(raw=f"{pert_node} targets {seed}",
                                        structured={"seed_gene": seed, "method": method}),
                weight=1.0, confidence=0.92))

            affected_count = 0
            for origin in linked_sorted:
                h = canonicalize_entity_id(origin.head, gene_symbols=gene_symbols)
                t = canonicalize_entity_id(origin.tail, gene_symbols=gene_symbols)
                if h == f"gene::{seed}":
                    affected = t
                elif t == f"gene::{seed}":
                    affected = h
                else:
                    continue
                if not affected or affected == f"gene::{seed}":
                    continue
                if not (affected.startswith("gene::") or affected.startswith("protein::")
                        or affected.startswith("pathway::") or affected.startswith("Cell::")):
                    continue
                rel = canonical_relation(origin.relation)
                base_weight = float(getattr(origin, "weight", 0.6) or 0.6)
                base_conf = float(getattr(origin, "confidence", 0.6) or 0.6)
                derived.append(KGTriple(
                    head=pert_node, relation=f"affects_{rel}", tail=affected,
                    source="DerivedPerturbation", version="v1",
                    evidence=build_evidence(
                        raw=f"derived from {origin.head} {origin.relation} {origin.tail}",
                        structured={"seed_gene": seed, "method": method,
                                    "derived_from": {"head": origin.head, "relation": origin.relation,
                                                     "tail": origin.tail, "source": origin.source}}),
                    weight=round(max(0.2, min(1.0, base_weight * 0.85)), 4),
                    confidence=round(max(0.2, min(1.0, base_conf * 0.85)), 4)))
                affected_count += 1
                if affected_count >= max_affected_per_seed:
                    break

            # Shared-pathway 2-hop
            shared_paths = sorted(gene_to_pathways.get(seed, set()))
            hop2_candidates: List[Dict[str, Any]] = []
            for pathway_node in shared_paths:
                ss = _clamp01(gene_pathway_strength.get((seed, pathway_node), 0.0), default=0.0)
                if ss < min_pathway_edge_strength:
                    continue
                for other_node in sorted(pathway_to_genes.get(pathway_node, set())):
                    if other_node == f"gene::{seed}":
                        continue
                    other_g = other_node.split("::", 1)[-1]
                    os_ = _clamp01(gene_pathway_strength.get((other_g, pathway_node), 0.0), default=0.0)
                    if os_ < min_pathway_edge_strength:
                        continue
                    pscore = _clamp01((ss * os_) ** 0.5)
                    if pscore < min_shared_pathway_score:
                        continue
                    hop2_candidates.append({
                        "tail": other_node, "via_pathway": pathway_node,
                        "pathway_score": pscore,
                        "weight": _clamp01(0.58 * (0.7 + 0.6 * pscore), default=0.58),
                        "confidence": _clamp01(0.62 * (0.7 + 0.6 * pscore), default=0.62),
                    })

            if shared_pathway_keep_best_per_gene:
                best_by_gene: Dict[str, Dict] = {}
                for c in hop2_candidates:
                    t = c["tail"]
                    if t not in best_by_gene or c["pathway_score"] > best_by_gene[t]["pathway_score"]:
                        best_by_gene[t] = c
                selected = sorted(best_by_gene.values(), key=lambda x: x["pathway_score"], reverse=True)
            else:
                selected = sorted(hop2_candidates, key=lambda x: x["pathway_score"], reverse=True)

            hop2_count = 0
            for cand in selected:
                derived.append(KGTriple(
                    head=pert_node, relation="affects_shared_pathway_2hop", tail=cand["tail"],
                    source="DerivedPerturbation", version="v1",
                    evidence=build_evidence(
                        raw=f"{seed}->{cand['via_pathway']}<-{cand['tail']}",
                        structured={"seed_gene": seed, "method": method,
                                    "via_pathway": cand["via_pathway"], "hop": 2,
                                    "pathway_score": round(float(cand["pathway_score"]), 6)}),
                    weight=round(float(cand["weight"]), 4),
                    confidence=round(float(cand["confidence"]), 4)))
                hop2_count += 1
                if hop2_count >= max_shared_pathway_2hop_per_seed:
                    break

    return deduplicate_triples(derived)


# ======================================================================
# CSV 行写入
# ======================================================================

def _write_csv_row(writer, triple: KGTriple):
    writer.writerow({
        "head": triple.head,
        "relation": canonical_relation(triple.relation),
        "raw_relation": triple.relation,
        "tail": triple.tail,
        "source": triple.source,
        "version": triple.version,
        "evidence": triple.evidence,
        "weight": _clamp01(triple.weight, default=1.0),
        "confidence": _clamp01(triple.confidence, default=0.5),
        "effect_sign": float(getattr(triple, "effect_sign",
                                     infer_effect_sign_from_relation(triple.relation)) or 0.0),
    })


# ======================================================================
# Main
# ======================================================================

def main():
    p = argparse.ArgumentParser(description="Batch-build lung cancer KG — memory-safe sub-batch streaming")
    p.add_argument("--manifest", default="data/raw/lung_cancer/download_manifest.json")
    p.add_argument("--out-triples", default="data/processed/lung_cancer/kg_triples.csv")
    p.add_argument("--omics-manifest", default="")
    p.add_argument("--replace", action="store_true")
    p.add_argument("--skip-csv", action="store_true")
    p.add_argument("--max-rows-per-source", type=int, default=200000)
    p.add_argument("--min-string-score", type=int, default=700)
    p.add_argument("--disable-perturbation-augmentation", action="store_true")
    p.add_argument("--max-derived-affected-per-seed", type=int, default=120)
    p.add_argument("--max-shared-pathway-2hop-per-seed", type=int, default=80)
    p.add_argument("--min-pathway-edge-strength", type=float, default=0.45)
    p.add_argument("--min-shared-pathway-score", type=float, default=0.5)
    p.add_argument("--shared-pathway-keep-all", action="store_true")
    p.add_argument("--sub-batch-size", type=int, default=SUB_BATCH_SIZE,
                   help="每子批次三元组数量（控制峰值内存）")
    p.add_argument("--min-edge-weight", type=float, default=0.0,
                   help="全局最小边权重阈值 (0-1)，低于此值的边不入库")
    p.add_argument("--min-edge-confidence", type=float, default=0.0,
                   help="全局最小边置信度阈值 (0-1)，低于此值的边不入库")
    args = p.parse_args()

    sub_batch = max(500, int(args.sub_batch_size))

    # ---- 加载 manifest & seed genes ----
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed_file = Path(manifest.get("seed_gene_file", "")) if manifest.get("seed_gene_file") else None
    seed_genes = load_seed_genes(seed_file)
    print(f"Seed genes: {len(seed_genes)}")

    # ---- 初始化 store ----
    store = KnowledgeGraphStore()

    # ---- Neo4j 不可用 → 回退到原始单次模式 ----
    if store.backend != "neo4j":
        print("[WARN] Neo4j 不可用，回退到 NetworkX 模式（大数据集可能内存溢出）")
        from src.knowledge_graph.lung_cancer_etl import (
            derive_perturbation_triples,
            parse_manifest_downloads,
            write_triples_csv,
        )
        from scripts.build_lung_cancer_kg import build_nx_graph

        triples = parse_manifest_downloads(
            manifest, max_rows_per_source=args.max_rows_per_source,
            min_string_score=args.min_string_score)
        omics_count = 0
        if args.omics_manifest:
            omics_path = Path(args.omics_manifest)
            if omics_path.exists():
                ot = build_omics_triples(load_omics_manifest(omics_path))
                omics_count = len(ot)
                triples = deduplicate_triples(list(triples) + list(ot))
        derived_count = 0
        if not args.disable_perturbation_augmentation:
            dt = derive_perturbation_triples(
                triples, seed_genes=seed_genes,
                max_affected_per_seed=args.max_derived_affected_per_seed,
                max_shared_pathway_2hop_per_seed=args.max_shared_pathway_2hop_per_seed,
                min_pathway_edge_strength=args.min_pathway_edge_strength,
                min_shared_pathway_score=args.min_shared_pathway_score,
                shared_pathway_keep_best_per_gene=(not args.shared_pathway_keep_all))
            derived_count = len(dt)
            triples = deduplicate_triples(list(triples) + list(dt))
        if not args.skip_csv:
            write_triples_csv(triples, Path(args.out_triples))
        graph = build_nx_graph(triples, seed_genes)
        store.save_graph(graph, replace=args.replace)
        print(f"Triples: {len(triples)}  Omics: {omics_count}  Derived: {derived_count}")
        print(f"Graph nodes: {graph.number_of_nodes()}  edges: {graph.number_of_edges()}")
        return

    # ---- Neo4j 流式路径 ----
    out_path = Path(args.out_triples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent

    # 局部包装：注入全局质量阈值
    def _normalize(triples):
        return iter_normalized_rows(
            triples, seed_genes,
            min_edge_weight=args.min_edge_weight,
            min_edge_confidence=args.min_edge_confidence)

    # 磁盘 SQLite dedup
    dedup = _open_dedup_db(work_dir)

    # 稀疏扰动数据收集器
    gene_neighbors: Dict[str, List[KGTriple]] = defaultdict(list)
    gene_to_pathways: Dict[str, Set[str]] = defaultdict(set)
    pathway_to_genes: Dict[str, Set[str]] = defaultdict(set)
    gene_pathway_strength: Dict[tuple, float] = {}

    # CSV
    csv_handle = None
    csv_writer = None
    if not args.skip_csv:
        csv_handle = out_path.open("w", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(csv_handle, fieldnames=[
            "head", "relation", "raw_relation", "tail", "source", "version",
            "evidence", "weight", "confidence", "effect_sign"])
        csv_writer.writeheader()

    total = 0
    total_omics = 0
    total_derived = 0
    is_first = args.replace

    # ===== Phase 1: 基础知识源（逐源 → 子批次 → Neo4j） =====
    print("=" * 60)
    print("Phase 1: Processing base knowledge sources")
    print("=" * 60)

    for src_name, src, ver, src_triples in iter_source_triples(
        manifest,
        max_rows_per_source=args.max_rows_per_source,
        min_string_score=args.min_string_score,
    ):
        if not src_triples:
            continue

        source_new = 0
        # 源内子批次切分
        for chunk in _chunked(src_triples, sub_batch):
            new_chunk = _dedup_batch(dedup, chunk)
            if not new_chunk:
                continue

            # 规范化 + 写入 Neo4j
            rows = list(_normalize(new_chunk))
            store.save_triples_streaming(iter(rows), replace=is_first)
            is_first = False

            # 收集扰动种子
            if not args.disable_perturbation_augmentation:
                for t in new_chunk:
                    _collect_perturbation_seeds(
                        t, seed_genes, gene_neighbors,
                        gene_to_pathways, pathway_to_genes, gene_pathway_strength)

            # CSV
            if csv_writer is not None:
                for t in new_chunk:
                    _write_csv_row(csv_writer, t)

            source_new += len(new_chunk)
            total += len(new_chunk)

        print(f"  [{src_name}] {source_new} new triples  (source: {src})  -> total {total}")

    # ===== Phase 2: Omics（逐组学数据集 → 分块迭代 → Neo4j）=====
    if args.omics_manifest:
        omics_path = Path(args.omics_manifest)
        if omics_path.exists():
            print("=" * 60)
            print("Phase 2: Processing omics datasets (chunked streaming)")
            print("=" * 60)
            for spec in load_omics_manifest(omics_path):
                if not spec.matrix_path:
                    continue
                ds_name = spec.name
                ds_new = 0
                chunk_idx = 0
                try:
                    for chunk in iter_omics_triples_chunked([spec], chunk_size=sub_batch):
                        if not chunk:
                            continue
                        chunk_idx += 1
                        new_chunk = _dedup_batch(dedup, chunk)
                        if not new_chunk:
                            continue
                        rows = list(_normalize(new_chunk))
                        store.save_triples_streaming(iter(rows), replace=False)
                        if not args.disable_perturbation_augmentation:
                            for t in new_chunk:
                                _collect_perturbation_seeds(
                                    t, seed_genes, gene_neighbors,
                                    gene_to_pathways, pathway_to_genes, gene_pathway_strength)
                        if csv_writer is not None:
                            for t in new_chunk:
                                _write_csv_row(csv_writer, t)
                        ds_new += len(new_chunk)
                        total += len(new_chunk)
                        if chunk_idx % 10 == 0:
                            print(f"  [{ds_name}] chunk {chunk_idx}, {ds_new} new so far ...", flush=True)
                except Exception:
                    print(f"  [ERROR] omics {ds_name}: {traceback.format_exc()}")
                    continue
                total_omics += ds_new
                print(f"  [{ds_name}] {ds_new} omics triples  (chunks: {chunk_idx})  -> total {total}")

    # ===== Phase 3: 扰动推导 =====
    if not args.disable_perturbation_augmentation:
        print("=" * 60)
        print("Phase 3: Deriving perturbation triples")
        print("=" * 60)
        print(f"  Seed genes with neighbors: {len(gene_neighbors)}")
        print(f"  Gene-pathway mappings: genes={len(gene_to_pathways)}  pathways={len(pathway_to_genes)}")

        derived = derive_perturbation_from_collected(
            gene_neighbors=gene_neighbors,
            gene_to_pathways=gene_to_pathways,
            pathway_to_genes=pathway_to_genes,
            gene_pathway_strength=gene_pathway_strength,
            seed_genes=seed_genes,
            max_affected_per_seed=args.max_derived_affected_per_seed,
            max_shared_pathway_2hop_per_seed=args.max_shared_pathway_2hop_per_seed,
            min_pathway_edge_strength=args.min_pathway_edge_strength,
            min_shared_pathway_score=args.min_shared_pathway_score,
            shared_pathway_keep_best_per_gene=(not args.shared_pathway_keep_all),
        )

        new_derived = _dedup_batch(dedup, derived)
        print(f"  Derived: {len(new_derived)} new triples (from {len(derived)} candidates)")

        if new_derived:
            for chunk in _chunked(new_derived, sub_batch):
                rows = list(_normalize(chunk))
                store.save_triples_streaming(iter(rows), replace=False)
                if csv_writer is not None:
                    for t in chunk:
                        _write_csv_row(csv_writer, t)
            total_derived = len(new_derived)
            total += total_derived

    # ===== 清理 =====
    if csv_handle is not None:
        csv_handle.close()
    dedup.close()

    print("=" * 60)
    print("BUILD COMPLETE")
    print(f"  Base triples:    {total - total_derived - total_omics}")
    print(f"  Omics triples:   {total_omics}")
    print(f"  Derived triples: {total_derived}")
    print(f"  Total triples:   {total}")
    print(f"  CSV:             {out_path if not args.skip_csv else '(skipped)'}")
    print(f"  Backend:         {store.backend}")
    try:
        s = store.stats()
        print(f"  Graph nodes:     {s.get('entities', '?')}")
        print(f"  Graph edges:     {s.get('relations', '?')}")
    except Exception:
        print("  Graph stats:     (unavailable)")


if __name__ == "__main__":
    main()
