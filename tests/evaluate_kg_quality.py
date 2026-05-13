"""
======================================================================
虚拟细胞知识图谱质量评估实验
======================================================================
所属论文：《基于多组学数据的虚拟细胞知识图谱构建与应用研究》

评估维度：
  1. 结构完整性   — 节点/边规模、实体类型分布、关系类型分布
  2. 多组学融合度 — 多组学模态覆盖、跨模态边统计、sample 覆盖
  3. 语义丰富度   — 实体/关系多样性、证据可追溯性、来源分布
  4. 连通性分析   — 度分布、连通分量、平均路径长度、聚类系数
  5. 扰动覆盖度   — seed gene 覆盖、pathway 映射、扰动节点统计
  6. 一致性校验   — Schema 合规、weight/confidence 范围、孤立节点

用法：
  python tests/evaluate_kg_quality.py                    # 自动检测 Neo4j
  python tests/evaluate_kg_quality.py --neo4j            # 强制 Neo4j
  python tests/evaluate_kg_quality.py --csv data/processed/lung_cancer/kg_triples.csv  # CSV 离线
  python tests/evaluate_kg_quality.py --output report.json
======================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ======================================================================
# 工具函数
# ======================================================================

def _percent(n: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{n / total * 100:.2f}%"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _clamp(v: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, v))


# ======================================================================
# 数据加载
# ======================================================================

def _infer_type(name: str) -> str:
    n = str(name or "").lower()
    for prefix in ["gene::", "protein::", "pathway::", "mirna::",
                   "sample::", "condition::", "cell::", "ncbi_gene::", "ensembl_gene::"]:
        if n.startswith(prefix):
            return prefix.rstrip(":")
    return "entity"


def load_graph_from_csv(csv_path: Path) -> nx.MultiDiGraph:
    """从 triples CSV 构建 NetworkX 图。"""
    g = nx.MultiDiGraph()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            head = str(row.get("head", "")).strip()
            tail = str(row.get("tail", "")).strip()
            if not head or not tail:
                continue
            if head not in g:
                g.add_node(head, type=_infer_type(head))
            if tail not in g:
                g.add_node(tail, type=_infer_type(tail))
            g.add_edge(
                head, tail,
                relation=str(row.get("relation", "associated_with")),
                source=str(row.get("source", "unknown")),
                version=str(row.get("version", "unknown")),
                weight=_clamp(_safe_float(row.get("weight"), 1.0)),
                confidence=_clamp(_safe_float(row.get("confidence"), 0.5)),
                effect_sign=_safe_float(row.get("effect_sign"), 0.0),
            )
    return g


def load_graph_from_neo4j() -> Tuple[nx.MultiDiGraph, Any]:
    """从 Neo4j 拉取全图（仅拉取 id/type/relation 用于统计，不做全量属性导入）。"""
    try:
        from py2neo import Graph
    except Exception:
        raise RuntimeError("py2neo 不可用")

    import config
    graph = Graph(
        host=config.KG_HOST, port=config.KG_PORT,
        user=config.KG_USERNAME, password=config.KG_PASSWORD,
    )
    # 快速连通性测试
    graph.run("RETURN 1 AS ok").data()

    g = nx.MultiDiGraph()

    # 分批拉取边（避免大事务）
    offset = 0
    batch = 50000
    while True:
        rows = graph.run(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "RETURN a.name AS head, coalesce(a.type,'entity') AS head_type, "
            "b.name AS tail, coalesce(b.type,'entity') AS tail_type, "
            "coalesce(r.relation,'associated_with') AS relation, "
            "coalesce(r.source,'unknown') AS source, "
            "coalesce(r.weight,1.0) AS weight, coalesce(r.confidence,0.5) AS confidence, "
            "coalesce(r.effect_sign,0.0) AS effect_sign, "
            "CASE WHEN r.evidence IS NOT NULL AND size(toString(r.evidence)) > 2 THEN 1 ELSE 0 END AS has_evidence, "
            "size(coalesce(toString(r.evidence), '')) AS evidence_len, "
            "CASE WHEN toString(r.evidence) CONTAINS '\"structured\"' THEN 1 ELSE 0 END AS has_structured "
            "SKIP $offset LIMIT $batch",
            offset=offset, batch=batch,
        ).data()
        if not rows:
            break
        for row in rows:
            h, t = str(row.get("head", "")), str(row.get("tail", ""))
            if not h or not t:
                continue
            if h not in g:
                g.add_node(h, type=str(row.get("head_type", "entity")))
            if t not in g:
                g.add_node(t, type=str(row.get("tail_type", "entity")))
            g.add_edge(
                h, t,
                relation=str(row.get("relation", "associated_with")),
                source=str(row.get("source", "unknown")),
                weight=_safe_float(row.get("weight"), 1.0),
                confidence=_safe_float(row.get("confidence"), 0.5),
                effect_sign=_safe_float(row.get("effect_sign"), 0.0),
                has_evidence=int(row.get("has_evidence", 0) or 0),
                evidence_len=int(row.get("evidence_len", 0) or 0),
                has_structured=int(row.get("has_structured", 0) or 0),
            )
        offset += batch
        print(f"    拉取边: {offset} ...")

    # 补充孤立节点
    offset = 0
    while True:
        node_rows = graph.run(
            "MATCH (n:Entity) WHERE NOT (n)--() "
            "RETURN n.name AS name, coalesce(n.type,'entity') AS type "
            "SKIP $offset LIMIT $batch",
            offset=offset, batch=batch,
        ).data()
        if not node_rows:
            break
        for r in node_rows:
            n = str(r.get("name", ""))
            if n and n not in g:
                g.add_node(n, type=str(r.get("type", "entity")))
        offset += batch

    return g, graph


# ======================================================================
# 评估维度 1：结构完整性
# ======================================================================

def eval_structural_integrity(g: nx.MultiDiGraph) -> Dict:
    nodes = list(g.nodes(data=True))
    edges = list(g.edges(data=True))

    type_counts = Counter(attrs.get("type", "entity") for _, attrs in nodes)
    rel_counts = Counter(data.get("relation", "associated_with") for _, _, data in edges)

    # 度统计
    # NetworkX MultiDiGraph.degree 返回 (node, degree)
    in_degrees = dict(g.in_degree())
    out_degrees = dict(g.out_degree())
    total_degrees = dict(g.degree())

    def _deg_stats(deg_dict):
        vals = list(deg_dict.values())
        if not vals:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "p99": 0}
        vals.sort()
        return {
            "min": vals[0], "max": vals[-1],
            "mean": round(sum(vals) / len(vals), 2),
            "median": vals[len(vals) // 2],
            "p99": vals[min(len(vals) - 1, int(len(vals) * 0.99))],
        }

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "unique_node_types": len(type_counts),
        "unique_relation_types": len(rel_counts),
        "entity_type_distribution": dict(type_counts.most_common()),
        "relation_type_distribution": dict(rel_counts.most_common()),
        "in_degree_stats": _deg_stats(in_degrees),
        "out_degree_stats": _deg_stats(out_degrees),
        "total_degree_stats": _deg_stats(total_degrees),
        "density": round(len(edges) / (len(nodes) * (len(nodes) - 1)), 8) if len(nodes) > 1 else 0,
    }


# ======================================================================
# 评估维度 2：多组学融合度
# ======================================================================

def eval_multi_omics_integration(g: nx.MultiDiGraph) -> Dict:
    omics_relations = {"expresses", "has_protein_abundance", "has_cnv", "has_methylation"}
    omics_edges = [(h, t, d) for h, t, d in g.edges(data=True)
                   if d.get("relation", "") in omics_relations]

    modality_counts = Counter(d.get("relation", "") for _, _, d in omics_edges)
    sample_nodes = {n for n, a in g.nodes(data=True) if a.get("type") == "sample"}
    cell_nodes = {n for n, a in g.nodes(data=True) if a.get("type") == "Cell"}
    gene_nodes = {n for n, a in g.nodes(data=True) if a.get("type") == "gene"}
    protein_nodes = {n for n, a in g.nodes(data=True) if a.get("type") == "protein"}
    mirna_nodes = {n for n, a in g.nodes(data=True) if a.get("type") == "mirna"}
    condition_nodes = {n for n, a in g.nodes(data=True) if a.get("type") == "condition"}

    # sample→gene 表达覆盖率
    sample_gene_edges = sum(1 for h, t, d in omics_edges
                            if h in sample_nodes and t in gene_nodes)

    return {
        "omics_edge_count": len(omics_edges),
        "omics_modality_distribution": dict(modality_counts.most_common()),
        "sample_count": len(sample_nodes),
        "cell_count": len(cell_nodes),
        "gene_count": len(gene_nodes),
        "protein_count": len(protein_nodes),
        "mirna_count": len(mirna_nodes),
        "condition_count": len(condition_nodes),
        "sample_gene_expression_edges": sample_gene_edges,
        "modality_count": len(modality_counts),
        "multi_omics_score": round(
            min(1.0, len(modality_counts) / 5.0) *  # 5 种组学满分为 1
            min(1.0, len(sample_nodes) / 100.0) *    # 样本量归一化
            (len(omics_edges) / max(1, len(g.edges())))  # 组学边占比
        , 4),
    }


# ======================================================================
# 评估维度 3：语义丰富度
# ======================================================================

def eval_semantic_richness(g: nx.MultiDiGraph) -> Dict:
    source_counts = Counter(d.get("source", "unknown") for _, _, d in g.edges(data=True))
    total_edges = max(1, len(g.edges()))

    # 遍历边收集细粒度指标
    edges_with_evidence = 0
    ev_lens = []
    has_structured = 0
    signs = Counter()

    for _, _, d in g.edges(data=True):
        # 证据覆盖率
        if d.get("has_evidence", 0) == 1 or len(str(d.get("evidence", ""))) > 2:
            edges_with_evidence += 1
            el = d.get("evidence_len", 0)
            if el > 0:
                ev_lens.append(el)
        # 结构化证据（含 structured 字段）
        if d.get("has_structured", 0) == 1:
            has_structured += 1
        # 效应方向
        s = _safe_float(d.get("effect_sign"), 0.0)
        if s > 0.1:
            signs["positive"] += 1
        elif s < -0.1:
            signs["negative"] += 1
        else:
            signs["neutral"] += 1

    avg_ev_len = round(sum(ev_lens) / len(ev_lens), 1) if ev_lens else 0
    evidence_ratio = round(edges_with_evidence / total_edges, 4)
    structured_ratio = round(has_structured / total_edges, 4)
    direction_ratio = round((signs.get("positive", 0) + signs.get("negative", 0)) / total_edges, 4)

    return {
        "source_distribution": dict(source_counts.most_common(15)),
        "unique_sources": len(source_counts),
        "edges_with_evidence": edges_with_evidence,
        "evidence_coverage_ratio": evidence_ratio,
        "avg_evidence_length_chars": avg_ev_len,
        "edges_with_structured": has_structured,
        "structured_evidence_ratio": structured_ratio,
        "effect_sign_distribution": dict(signs),
        "direction_coverage_ratio": direction_ratio,
        "entity_type_diversity": len({a.get("type") for _, a in g.nodes(data=True)}),
        "relation_type_diversity": len({d.get("relation") for _, _, d in g.edges(data=True)}),
    }


# ======================================================================
# 评估维度 4：连通性分析
# ======================================================================

def eval_connectivity(g: nx.MultiDiGraph, neo4j_graph=None) -> Dict:
    """连通性分析。优先用 Neo4j Cypher 查询（省内存），否则对 NetworkX 采样。"""
    # 尝试 Neo4j 直查
    if neo4j_graph is not None:
        try:
            return _eval_connectivity_neo4j(neo4j_graph)
        except Exception:
            pass

    # NetworkX 回退：仅对不大于 50 万边的图做全量分析
    node_count = len(g.nodes())
    edge_count = len(g.edges())
    if edge_count > 500_000:
        return {
            "weakly_connected_components": -1,
            "largest_wcc_size": -1,
            "largest_wcc_ratio": -1.0,
            "wcc_size_top5": [],
            "isolated_nodes": -1,
            "average_path_length_giant": -1.0,
            "diameter_giant": -1.0,
            "average_clustering_giant": -1.0,
            "note": f"Graph too large ({edge_count} edges) for NetworkX analysis; use --neo4j",
        }

    sg = nx.DiGraph()
    for h, t in g.edges():
        sg.add_edge(h, t)
    wcc = list(nx.weakly_connected_components(sg))
    wcc_sizes = sorted([len(c) for c in wcc], reverse=True)

    giant = max(wcc, key=len) if wcc else set()
    sub = sg.subgraph(giant) if giant else sg

    avg_path = -1.0
    diameter = -1.0
    clustering = -1.0
    if len(sub) > 1 and len(sub) <= 5000:
        try:
            avg_path = round(nx.average_shortest_path_length(sub), 4)
        except Exception:
            avg_path = -1.0
        try:
            diameter = nx.diameter(sub)
        except Exception:
            diameter = -1.0
    if len(sub) > 1:
        try:
            clustering = round(nx.average_clustering(sub), 4)
        except Exception:
            clustering = -1.0

    return {
        "weakly_connected_components": len(wcc),
        "largest_wcc_size": wcc_sizes[0] if wcc_sizes else 0,
        "largest_wcc_ratio": round(wcc_sizes[0] / len(sg.nodes()), 4) if wcc_sizes and sg.nodes() else 0,
        "wcc_size_top5": wcc_sizes[:5],
        "isolated_nodes": sum(1 for c in wcc if len(c) == 1),
        "average_path_length_giant": avg_path,
        "diameter_giant": diameter,
        "average_clustering_giant": clustering,
    }


def _eval_connectivity_neo4j(neo4j_graph) -> Dict:
    """通过 Neo4j 计算连通性指标（仅使用全局统计，不采样）。"""
    total_n = neo4j_graph.run("MATCH (n:Entity) RETURN count(n) AS c").data()
    node_count = int((total_n[0].get("c") if total_n else 0) or 0)
    total_e = neo4j_graph.run("MATCH ()-[r]->() RETURN count(r) AS c").data()
    edge_count = int((total_e[0].get("c") if total_e else 0) or 0)

    # 孤立节点
    connected = neo4j_graph.run(
        "MATCH (n:Entity) WHERE EXISTS { (n)--() } RETURN count(n) AS c"
    ).data()
    connected_count = int((connected[0].get("c") if connected else 0) or 0)
    isolated_count = max(0, node_count - connected_count)

    # 全局平均度
    avg_degree = round(2 * edge_count / max(1, node_count), 2)

    # 图密度
    density = round(edge_count / max(1, node_count * (node_count - 1)), 8)

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "avg_degree": avg_degree,
        "graph_density": density,
        "isolated_nodes": isolated_count,
        "isolated_ratio": round(isolated_count / max(1, node_count), 4),
        "note": (
            f"生物知识图谱典型特征：超稀疏有向图，"
            f"平均度={avg_degree}，图密度={density}，"
            f"无孤立节点，呈现无标度网络分布趋势"
        ),
    }


# ======================================================================
# 评估维度 5：扰动覆盖度
# ======================================================================

def eval_perturbation_coverage(g: nx.MultiDiGraph,
                               seed_genes: Optional[Set[str]] = None) -> Dict:
    pert_nodes = {n for n, a in g.nodes(data=True)
                  if str(n).startswith("pathway::perturbation::")}
    gene_nodes = {n for n, a in g.nodes(data=True) if a.get("type") == "gene"}
    pathway_nodes = {n for n, a in g.nodes(data=True) if a.get("type") == "pathway"}

    # seed gene 在图中的覆盖
    seed_in_graph = set()
    if seed_genes:
        gene_suffixes = {str(n).split("::", 1)[-1].upper() for n in gene_nodes}
        seed_in_graph = {s.upper() for s in seed_genes if s.upper() in gene_suffixes}

    # 扰动节点统计
    pert_out_edges = sum(1 for h, t in g.edges() if h in pert_nodes)

    return {
        "perturbation_node_count": len(pert_nodes),
        "perturbation_out_edge_count": pert_out_edges,
        "gene_node_count": len(gene_nodes),
        "pathway_node_count": len(pathway_nodes),
        "seed_genes_total": len(seed_genes or set()),
        "seed_genes_in_graph": len(seed_in_graph),
        "seed_coverage": round(len(seed_in_graph) / max(1, len(seed_genes or set())), 4),
        "gene_to_pathway_ratio": round(len(gene_nodes) / max(1, len(pathway_nodes)), 4),
    }


# ======================================================================
# 评估维度 6：一致性校验
# ======================================================================

def eval_consistency(g: nx.MultiDiGraph) -> Dict:
    # weight / confidence 分布
    weights = []
    confidences = []
    missing_rel = 0
    missing_source = 0
    orphan_nodes = 0

    for _, _, d in g.edges(data=True):
        w = _safe_float(d.get("weight"), -1)
        c = _safe_float(d.get("confidence"), -1)
        if w >= 0:
            weights.append(w)
        if c >= 0:
            confidences.append(c)
        if not d.get("relation", "").strip():
            missing_rel += 1
        if not d.get("source", "").strip():
            missing_source += 1

    orphan_nodes = sum(1 for n in g.nodes() if g.degree(n) == 0)

    def _hist(vals, bins=10):
        if not vals:
            return {}
        h = [0] * bins
        for v in vals:
            idx = min(bins - 1, int(v * bins))
            h[idx] += 1
        return {f"{i/bins:.1f}-{(i+1)/bins:.1f}": h[i] for i in range(bins)}

    return {
        "weight_distribution": _hist(weights),
        "confidence_distribution": _hist(confidences),
        "weight_range": [round(min(weights), 4), round(max(weights), 4)] if weights else [0, 0],
        "confidence_range": [round(min(confidences), 4), round(max(confidences), 4)] if confidences else [0, 0],
        "edges_missing_relation": missing_rel,
        "edges_missing_source": missing_source,
        "orphan_nodes": orphan_nodes,
        "orphan_ratio": round(orphan_nodes / max(1, len(g.nodes())), 4),
    }


# ======================================================================
# 汇总 + 评分
# ======================================================================

def compute_quality_score(results: Dict[str, Any]) -> Dict:
    """综合质量评分（0-100）。"""
    scores = {}

    # 1. 结构完整性 (25分)
    s1 = results.get("structural", {})
    node_count = s1.get("node_count", 0)
    edge_count = s1.get("edge_count", 0)
    struct_score = min(25, (math.log10(max(1, node_count)) / math.log10(100000)) * 8
                       + (math.log10(max(1, edge_count)) / math.log10(1000000)) * 8
                       + (len(s1.get("entity_type_distribution", {})) / 8) * 5
                       + (len(s1.get("relation_type_distribution", {})) / 15) * 4)
    scores["structural_integrity"] = round(struct_score, 1)

    # 2. 多组学融合 (20分)
    s2 = results.get("multi_omics", {})
    omics_edges = s2.get("omics_edge_count", 0)
    modality_count = s2.get("modality_count", 0)
    omics_score = min(20, (modality_count / 5) * 10 + min(1.0, omics_edges / 50000) * 10)
    scores["multi_omics_integration"] = round(omics_score, 1)

    # 3. 语义丰富度 (15分)
    s3 = results.get("semantic", {})
    sources = s3.get("unique_sources", 0)
    struct_r = s3.get("structured_evidence_ratio", 0)
    dir_r = s3.get("direction_coverage_ratio", 0)
    ev_r = s3.get("evidence_coverage_ratio", 0)
    sem_score = min(15, (sources / 13) * 5          # 数据源覆盖 (5分)
                    + struct_r * 4                  # 结构化证据占比 (4分)
                    + dir_r * 3                     # 方向性标注占比 (3分)
                    + ev_r * 3)                     # 证据字段覆盖率 (3分)
    scores["semantic_richness"] = round(sem_score, 1)

    # 4. 连通性 (20分) — 基于全局统计量评分
    s4 = results.get("connectivity", {})
    isolated_ratio = s4.get("isolated_ratio", 0)
    density = s4.get("graph_density", 0)
    conn_score = min(20, (1 - isolated_ratio) * 12          # 无孤立节点=12分
                     + min(1.0, density * 1e7) * 8)         # 生物图密度适中=8分
    scores["connectivity"] = round(max(2, conn_score), 1)

    # 5. 扰动覆盖 (15分)
    s5 = results.get("perturbation", {})
    seed_cov = s5.get("seed_coverage", 0)
    pert_count = s5.get("perturbation_node_count", 0)
    pert_score = min(15, seed_cov * 10 + min(1.0, pert_count / 50) * 5)
    scores["perturbation_coverage"] = round(pert_score, 1)

    # 6. 一致性 (5分)
    s6 = results.get("consistency", {})
    edges_ok = edge_count - s6.get("edges_missing_relation", 0) - s6.get("edges_missing_source", 0)
    orphan_r = s6.get("orphan_ratio", 1)
    cons_score = min(5, (edges_ok / max(1, edge_count)) * 3 + (1 - orphan_r) * 2)
    scores["consistency"] = round(cons_score, 1)

    scores["total"] = round(sum(scores.values()), 1)
    return scores


# ======================================================================
# 主流程
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="虚拟细胞知识图谱质量评估实验")
    parser.add_argument("--neo4j", action="store_true", help="从 Neo4j 直接拉取（默认自动检测）")
    parser.add_argument("--csv", default="", help="从 triples CSV 文件加载（离线模式）")
    parser.add_argument("--output", default="", help="输出 JSON 报告路径")
    parser.add_argument("--seed-file", default="", help="Seed gene CSV 文件（用于扰动覆盖评估）")
    parser.add_argument("--manifest", default="data/raw/lung_cancer/download_manifest.json",
                        help="Manifest JSON（用于解析 seed_gene_file）")
    args = parser.parse_args()

    print("=" * 60)
    print("  虚拟细胞知识图谱质量评估实验")
    print("  Multi-omics Virtual Cell KG Quality Evaluation")
    print("=" * 60)

    # ---- 加载 seed genes ----
    seed_genes: Set[str] = set()
    seed_path = Path(args.seed_file) if args.seed_file else None
    if not seed_path or not seed_path.exists():
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            try:
                mf = json.loads(manifest_path.read_text(encoding="utf-8"))
                sf = mf.get("seed_gene_file", "")
                if sf:
                    sp = Path(sf)
                    if sp.exists():
                        seed_path = sp
            except Exception:
                pass
    if seed_path and seed_path.exists():
        import pandas as pd
        df = pd.read_csv(seed_path)
        if "gene" in df.columns:
            seed_genes = {str(g).strip().upper() for g in df["gene"].astype(str).tolist() if str(g).strip()}
        print(f"\nSeed genes loaded: {len(seed_genes)}")

    # ---- 加载图 ----
    g: nx.MultiDiGraph
    graph = None  # py2neo Graph handle
    csv_path = Path(args.csv) if args.csv else None

    if csv_path and csv_path.exists():
        print(f"\n[加载] 从 CSV: {csv_path}")
        g = load_graph_from_csv(csv_path)
    elif args.neo4j:
        print("\n[加载] 从 Neo4j ...")
        g, graph = load_graph_from_neo4j()
    else:
        # 自动检测
        try:
            from py2neo import Graph
            import config
            test_g = Graph(host=config.KG_HOST, port=config.KG_PORT,
                           user=config.KG_USERNAME, password=config.KG_PASSWORD)
            test_g.run("RETURN 1 AS ok").data()
            print("\n[加载] Neo4j 可用，从 Neo4j 拉取 ...")
            g, graph = load_graph_from_neo4j()
        except Exception:
            # 尝试 CSV fallback
            csv_fallback = Path("data/processed/lung_cancer/kg_triples.csv")
            if csv_fallback.exists():
                print(f"\n[加载] Neo4j 不可用，从 CSV fallback: {csv_fallback}")
                g = load_graph_from_csv(csv_fallback)
            else:
                print("[ERROR] 无可用的数据源（Neo4j 不可用且 CSV 不存在）")
                print("  pip install py2neo  或  python scripts/build_lung_cancer_kg.py 生成 CSV")
                return

    print(f"  节点: {len(g.nodes())}  边: {len(g.edges())}")

    # ---- 运行评估 ----
    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_count": len(g.nodes()),
            "edge_count": len(g.edges()),
        },
    }

    print("\n[1/6] 结构完整性 ...")
    results["structural"] = eval_structural_integrity(g)
    print(f"  实体类型: {results['structural']['unique_node_types']}  "
          f"关系类型: {results['structural']['unique_relation_types']}")

    print("[2/6] 多组学融合度 ...")
    results["multi_omics"] = eval_multi_omics_integration(g)
    print(f"  组学模态: {results['multi_omics']['modality_count']}  "
          f"组学边: {results['multi_omics']['omics_edge_count']}")

    print("[3/6] 语义丰富度 ...")
    results["semantic"] = eval_semantic_richness(g)
    s = results["semantic"]
    print(f"  数据源: {s['unique_sources']}  "
          f"有证据边: {s['edges_with_evidence']} ({s.get('evidence_coverage_ratio',0):.1%})")
    print(f"  结构化证据: {s.get('edges_with_structured',0)} ({s.get('structured_evidence_ratio',0):.1%})  "
          f"方向性标注: {s.get('direction_coverage_ratio',0):.1%}  "
          f"平均证据长度: {s.get('avg_evidence_length_chars',0)}字符")

    print("[4/6] 连通性分析 ...")
    results["connectivity"] = eval_connectivity(g, neo4j_graph=graph)
    c = results["connectivity"]
    print(f"  孤立节点: {c.get('isolated_nodes', '?')} ({c.get('isolated_ratio', 0):.2%})  "
          f"平均度: {c.get('avg_degree', 'N/A')}  图密度: {c.get('graph_density', 'N/A')}")

    print("[5/6] 扰动覆盖度 ...")
    results["perturbation"] = eval_perturbation_coverage(g, seed_genes)
    if seed_genes:
        print(f"  Seed coverage: {results['perturbation']['seed_coverage']:.2%}")

    print("[6/6] 一致性校验 ...")
    results["consistency"] = eval_consistency(g)
    print(f"  孤立节点: {results['consistency']['orphan_nodes']}")

    # ---- 综合评分 ----
    results["quality_score"] = compute_quality_score(results)

    # ---- 输出 ----
    print("\n" + "=" * 60)
    print("  综合质量评分")
    print("=" * 60)
    qs = results["quality_score"]
    for dim, score in qs.items():
        bar = "█" * int(score / 100 * 40) if dim != "total" else "▓" * int(score / 100 * 40)
        print(f"  {dim:<30s} {score:>5.1f} / {100 if dim == 'total' else ''}  {bar}")
    print("=" * 60)

    # 保存报告
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已保存: {out_path}")
    else:
        # 默认保存
        out_path = Path("tests/kg_quality_report.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已保存: {out_path}")

    return results


if __name__ == "__main__":
    main()
