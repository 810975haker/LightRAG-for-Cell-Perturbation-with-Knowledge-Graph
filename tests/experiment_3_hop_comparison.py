"""
======================================================================
实验三：2-hop vs 1-hop 推理对比
======================================================================
分析图谱中扰动节点的直接邻域（1-hop）与共享通路扩展（2-hop）边的贡献差异。
数据来源：Neo4j 图谱中的 perturbation 节点及其出边。
======================================================================
用法：
  python tests/experiment_3_hop_comparison.py
  python tests/experiment_3_hop_comparison.py --output exp3_report.json
======================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import config

try:
    from py2neo import Graph
except Exception:
    Graph = None


def analyze_perturbation_edges(neo4j_graph) -> dict:
    """从 Neo4j 直接分析扰动节点的所有出边类型与目标分布。"""

    # 1. 扰动边语义类型分布
    sem_types = neo4j_graph.run(
        "MATCH (p:Entity)-[r]->(n:Entity) "
        "WHERE p.name STARTS WITH 'pathway::perturbation::' "
        "RETURN r.relation AS rel, count(r) AS cnt "
        "ORDER BY cnt DESC"
    ).data()

    # 2. 扰动节点数量
    pert_count = neo4j_graph.run(
        "MATCH (n:Entity) WHERE n.name STARTS WITH 'pathway::perturbation::' "
        "RETURN count(n) AS c"
    ).data()[0]["c"]

    # 3. 按边类型分类：1-hop (affects_*) vs 2-hop (affects_shared_pathway_2hop) vs 其他
    direct_edges = 0
    hop2_edges = 0
    other_edges = 0
    sem_detail = {}
    for row in sem_types:
        rel = str(row.get("rel", "associated_with"))
        cnt = int(row.get("cnt", 0))
        sem_detail[rel] = cnt
        if rel == "affects_shared_pathway_2hop":
            hop2_edges += cnt
        elif rel.startswith("affects_"):
            direct_edges += cnt
        else:
            other_edges += cnt

    # 4. 每个扰动节点按边类型的平均邻居数
    edge_stats = neo4j_graph.run(
        "MATCH (p:Entity)-[r]->(n:Entity) "
        "WHERE p.name STARTS WITH 'pathway::perturbation::' "
        "WITH p, r.relation AS rel, count(n) AS cnt "
        "RETURN rel, avg(cnt) AS avg_n, min(cnt) AS min_n, max(cnt) AS max_n "
        "ORDER BY avg_n DESC"
    ).data()

    # 5. 目标实体类型分布（区分 1-hop 和 2-hop）
    target_types = neo4j_graph.run(
        "MATCH (p:Entity)-[r]->(n:Entity) "
        "WHERE p.name STARTS WITH 'pathway::perturbation::' "
        "WITH r.relation AS rel, "
        "CASE "
        "  WHEN n.name STARTS WITH 'gene::' THEN 'gene' "
        "  WHEN n.name STARTS WITH 'pathway::' AND NOT n.name CONTAINS 'perturbation' THEN 'pathway' "
        "  WHEN n.name STARTS WITH 'protein::' THEN 'protein' "
        "  WHEN n.name STARTS WITH 'Cell::' THEN 'Cell' "
        "  ELSE 'other' "
        "END AS target_type, "
        "count(r) AS cnt "
        "RETURN rel, target_type, cnt "
        "ORDER BY rel, cnt DESC"
    ).data()

    # 6. 种子基因维度的分析
    seed_detail = neo4j_graph.run(
        "MATCH (p:Entity)-[r]->(n:Entity) "
        "WHERE p.name STARTS WITH 'pathway::perturbation::' "
        "WITH p.name AS pert, r.relation AS rel, count(n) AS cnt "
        "RETURN pert, rel, cnt "
        "ORDER BY pert, rel"
    ).data()

    # 按种子基因聚合
    gene_summary = {}
    for row in seed_detail:
        pert = str(row.get("pert", ""))
        rel = str(row.get("rel", ""))
        cnt = int(row.get("cnt", 0))
        # 提取种子基因名：pathway::perturbation::METHOD_GENE
        parts = pert.split("::")
        if len(parts) >= 3:
            gene_method = parts[-1]  # e.g., KO_EGFR
            gene = gene_method.split("_", 1)[-1] if "_" in gene_method else gene_method
        else:
            gene = pert
        if gene not in gene_summary:
            gene_summary[gene] = {"direct": 0, "hop2": 0, "total": 0}
        if rel == "affects_shared_pathway_2hop":
            gene_summary[gene]["hop2"] += cnt
        elif rel.startswith("affects_"):
            gene_summary[gene]["direct"] += cnt
        gene_summary[gene]["total"] += cnt

    return {
        "perturbation_node_count": pert_count,
        "total_perturbation_edges": sum(sem_detail.values()),
        "edge_type_breakdown": {
            "direct_1hop_edges": direct_edges,
            "shared_pathway_2hop_edges": hop2_edges,
            "other_edges": other_edges,
        },
        "hop2_edge_ratio": round(hop2_edges / max(1, direct_edges + hop2_edges), 4),
        "semantic_relation_distribution": sem_detail,
        "per_node_edge_stats": {
            row["rel"]: {
                "avg_neighbors": round(row["avg_n"], 1),
                "min": row["min_n"],
                "max": row["max_n"],
            }
            for row in edge_stats
        },
        "target_type_by_edge_type": [
            {
                "relation": row["rel"],
                "target_type": row["target_type"],
                "count": row["cnt"],
            }
            for row in target_types
        ],
        "per_seed_gene_summary": gene_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="2-hop vs 1-hop 推理对比实验")
    parser.add_argument("--output", default="tests/exp3_hop_comparison.json")
    args = parser.parse_args()

    if Graph is None:
        print("[ERROR] 需要 py2neo")
        return

    g = Graph(host=config.KG_HOST, port=config.KG_PORT,
              user=config.KG_USERNAME, password=config.KG_PASSWORD)
    g.run("RETURN 1 AS ok").data()

    print("分析图谱扰动节点结构...")
    results = analyze_perturbation_edges(g)

    # ── 输出 ──
    e = results["edge_type_breakdown"]
    print(f"\n扰动节点: {results['perturbation_node_count']}")
    print(f"扰动边总数: {results['total_perturbation_edges']}")
    print(f"  直接1-hop (affects_*):  {e['direct_1hop_edges']:>6}")
    print(f"  共享通路2-hop:          {e['shared_pathway_2hop_edges']:>6}")
    print(f"  其他:                   {e['other_edges']:>6}")
    print(f"  2-hop占比: {results['hop2_edge_ratio']:.1%}")

    print(f"\n语义关系类型分布:")
    for rel, cnt in sorted(results["semantic_relation_distribution"].items(),
                           key=lambda x: -x[1]):
        bar = "#" * min(50, cnt * 50 // max(1, results['total_perturbation_edges']))
        print(f"  {rel:<40s} {cnt:>6}  {bar}")

    print(f"\n每种边类型 每扰动节点平均邻居数:")
    for rel, stats in sorted(results["per_node_edge_stats"].items()):
        print(f"  {rel:<40s} avg={stats['avg_neighbors']:6.1f}  "
              f"[{stats['min']}, {stats['max']}]")

    print(f"\n种子基因维度（直接 vs 2-hop 贡献）:")
    for gene in sorted(results["per_seed_gene_summary"]):
        gs = results["per_seed_gene_summary"][gene]
        d, h2 = gs["direct"], gs["hop2"]
        ratio = h2 / max(1, d + h2)
        print(f"  {gene:<10s}  直接={d:>5}  +2hop={h2:>5}  "
              f"总计={gs['total']:>5}  2hop占比={ratio:.1%}")

    # 保存
    report = {
        "experiment": "2-hop vs 1-hop 推理对比",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()
