"""
======================================================================
实验四&五：消融实验 & 多组学模态贡献度分析（多跳路径版）
======================================================================
从全图谱分层抽样 200 个基因，分析 1-hop 到 3-hop 的邻居来源。
区分"知识源"和"组学模态"，追踪 STRING 通过 protein bridge 的间接贡献。

用法：
  python tests/experiment_4_ablation.py
  python tests/experiment_4_ablation.py --sample-size 100  (快速模式)
======================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import config

try:
    from py2neo import Graph
except Exception:
    Graph = None

SEED_GENES = ["EGFR", "KRAS", "TP53", "PIK3CA", "BRAF", "ERBB2", "ALK", "MET", "ROS1", "RET"]

# 知识源 vs 组学源分类
KNOWLEDGE_SOURCES = [
    "STRING", "BioGRID", "Reactome", "SIGNOR", "OmniPath", "KEGG",
    "NPInter", "ENCODE", "NCBI Gene", "Ensembl", "WikiPathways",
]
OMICS_SOURCES = ["TCGA", "GEO"]


def sample_genes_stratified(neo4j_graph, n: int) -> list:
    """按度分层抽样。"""
    pcts = neo4j_graph.run(
        "MATCH (n:Entity) WHERE n.name STARTS WITH 'gene::' "
        "WITH n, COUNT { (n)--() } AS deg "
        "RETURN percentileDisc(deg, 0.33) AS p33, percentileDisc(deg, 0.67) AS p67"
    ).data()[0]
    p33, p67 = int(pcts["p33"] or 10), int(pcts["p67"] or 50)

    genes = set(SEED_GENES)
    exclude = [f"gene::{g}" for g in genes]

    for label, deg_filter, skip in [
        ("low", f"deg <= {p33}", 100),
        ("mid", f"{p33} < deg <= {p67}", 200),
        ("high", f"deg > {p67}", 300),
    ]:
        rows = neo4j_graph.run(
            "MATCH (n:Entity) WHERE n.name STARTS WITH 'gene::' AND NOT n.name IN $ex "
            f"WITH n, COUNT {{ (n)--() }} AS deg WHERE {deg_filter} "
            "RETURN substring(n.name, 6) AS gene "
            f"ORDER BY n.name SKIP {skip} LIMIT {n // 3 + 10}",
            ex=exclude,
        ).data()
        genes.update(r["gene"] for r in rows)

    remaining = n - len(genes)
    if remaining > 0:
        rows = neo4j_graph.run(
            "MATCH (n:Entity) WHERE n.name STARTS WITH 'gene::' AND NOT n.name IN $ex "
            "RETURN substring(n.name, 6) AS gene "
            "ORDER BY n.name SKIP 500 LIMIT $r",
            ex=exclude, r=remaining,
        ).data()
        genes.update(r["gene"] for r in rows)

    return sorted(genes)[:n]


def analyze_multihop_contribution(neo4j_graph, gene_list: list) -> dict:
    """
    对每个基因追踪多跳邻居：
    - 1-hop: 直接邻居（基因/蛋白/通路）→ 按来源和类型分类
    - 2-hop: gene→X→gene 路径 → 看中间边来自什么源
    - 3-hop: gene→protein→protein→gene → 捕获 STRING 桥梁
    """

    # === 1-hop: 直接邻居 ===
    hop1 = neo4j_graph.run(
        "UNWIND $genes AS g "
        "MATCH (n:Entity {name:'gene::' + g})-[r]-(m:Entity) "
        "WHERE m.name STARTS WITH 'gene::' OR m.name STARTS WITH 'protein::' "
        "OR m.name STARTS WITH 'pathway::' "
        "RETURN g AS gene, coalesce(r.source, 'unknown') AS source, "
        "type(r) AS rel, count(DISTINCT m) AS cnt "
        "ORDER BY gene, cnt DESC",
        genes=gene_list,
    ).data()

    # === 2-hop: gene → {protein, pathway} → gene ===
    hop2 = neo4j_graph.run(
        "UNWIND $genes AS g "
        "MATCH (n:Entity {name:'gene::' + g})-[r1]-(mid:Entity)-[r2]-(target:Entity) "
        "WHERE target.name STARTS WITH 'gene::' "
        "  AND target.name <> n.name "
        "  AND (mid.name STARTS WITH 'protein::' OR mid.name STARTS WITH 'pathway::') "
        "RETURN g AS gene, "
        "coalesce(r1.source, 'unknown') + '→' + coalesce(r2.source, 'unknown') AS path, "
        "type(r1) + '→' + type(r2) AS path_type, "
        "count(DISTINCT target) AS cnt "
        "ORDER BY gene, cnt DESC",
        genes=gene_list,
    ).data()

    # === 3-hop: gene → protein → protein → gene (STRING 桥) ===
    hop3 = neo4j_graph.run(
        "UNWIND $genes AS g "
        "MATCH (n:Entity {name:'gene::' + g})-[r1]-(p1:Entity)-[r2]-(p2:Entity)-[r3]-(target:Entity) "
        "WHERE target.name STARTS WITH 'gene::' "
        "  AND target.name <> n.name "
        "  AND p1.name STARTS WITH 'protein::' "
        "  AND p2.name STARTS WITH 'protein::' "
        "RETURN g AS gene, "
        "coalesce(r2.source, 'unknown') AS bridge_source, "
        "count(DISTINCT target) AS cnt "
        "ORDER BY gene, cnt DESC",
        genes=gene_list,
    ).data()

    # === 聚合 ===
    deg_map = {}
    deg_rows = neo4j_graph.run(
        "UNWIND $genes AS g "
        "MATCH (n:Entity {name:'gene::' + g}) "
        "RETURN g AS gene, COUNT { (n)--() } AS deg",
        genes=gene_list,
    ).data()
    for r in deg_rows:
        deg_map[r["gene"]] = int(r["deg"])

    # ── 1-hop 来源和类型贡献 ──
    source_by_gene = defaultdict(lambda: defaultdict(int))
    rel_by_gene = defaultdict(lambda: defaultdict(int))
    for r in hop1:
        source_by_gene[r["gene"]][r["source"]] += r["cnt"]
        rel_by_gene[r["gene"]][r["rel"]] += r["cnt"]

    # ── 2-hop 路径来源贡献（拆出每条边独立算） ──
    hop2_source = defaultdict(lambda: defaultdict(int))
    for r in hop2:
        for src in r["path"].split("→"):
            hop2_source[r["gene"]][src] += r["cnt"]

    # ── 3-hop（STRING蛋白桥）贡献 ──
    hop3_source = defaultdict(lambda: defaultdict(int))
    for r in hop3:
        hop3_source[r["gene"]][r["bridge_source"]] += r["cnt"]

    # ── 汇总函数 ──
    def compute_contrib(gene_map, label, degs) -> dict:
        total = defaultdict(int)
        ratios = defaultdict(list)
        for gene in gene_list:
            d = max(1, degs.get(gene, 1))
            for src, cnt in gene_map.get(gene, {}).items():
                total[src] += cnt
                ratios[src].append(cnt / d)
        result = {}
        for src in total:
            rs = ratios[src]
            m = sum(rs) / len(rs)
            std = math.sqrt(sum((r - m) ** 2 for r in rs) / len(rs)) if len(rs) > 1 else 0
            ci = 1.96 * std / math.sqrt(len(rs)) if len(rs) > 1 else 0
            result[src] = {
                "total": total[src],
                "avg_ratio": round(m, 4),
                "ci95": round(ci, 4),
                "coverage": len([r for r in rs if r > 0]),
            }
        return dict(sorted(result.items(), key=lambda x: -x[1]["avg_ratio"]))

    hop1_source = compute_contrib(source_by_gene, "1-hop", deg_map)
    hop1_rel = compute_contrib(rel_by_gene, "1-hop", deg_map)
    hop2_src = compute_contrib(hop2_source, "2-hop", deg_map)
    hop3_src = compute_contrib(hop3_source, "3-hop", deg_map)

    # ── 组学覆盖分析：Sample→Gene 边统计 ──
    omics_modalities = ["EXPRESSES", "REGULATES", "HAS_METHYLATION", "HAS_CNV", "HAS_PROTEIN_ABUNDANCE"]
    omics_edges = neo4j_graph.run(
        "MATCH ()-[r]->(n:Entity) WHERE n.name STARTS WITH 'gene::' "
        "AND type(r) IN $mods "
        "RETURN type(r) AS modality, count(r) AS edge_cnt, "
        "count(DISTINCT n) AS genes_covered",
        mods=omics_modalities,
    ).data()
    omics_coverage = {}
    for row in omics_edges:
        omics_coverage[row["modality"]] = {
            "edges": int(row["edge_cnt"]),
            "genes_covered": int(row["genes_covered"]),
        }

    # 种子基因的组学覆盖
    seed_omics = {}
    for gene in SEED_GENES:
        rows = neo4j_graph.run(
            "MATCH ()-[r]->(n:Entity {name:$name}) "
            "WHERE type(r) IN $mods "
            "RETURN type(r) AS modality, count(r) AS samples",
            name=f"gene::{gene}", mods=omics_modalities,
        ).data()
        seed_omics[gene] = {r["modality"]: int(r["samples"]) for r in rows}

    # 知识源覆盖：至少有一个知识源邻居的基因比例
    k_genes = sum(1 for g in gene_list if any(
        any(ks.lower() in s.lower() for ks in KNOWLEDGE_SOURCES)
        for s in source_by_gene.get(g, {})))
    knowledge_gene_ratio = k_genes / len(gene_list)

    # 组学源覆盖：sample→gene 入向边。全量查询（不限抽样基因）取全局覆盖率
    total_gene_count = neo4j_graph.run(
        "MATCH (n:Entity) WHERE n.name STARTS WITH 'gene::' RETURN count(n) AS c"
    ).data()[0]["c"]
    omics_gene_count = neo4j_graph.run(
        "MATCH (s:Entity)-[r]->(n:Entity) WHERE n.name STARTS WITH 'gene::' "
        "AND s.name STARTS WITH 'sample::' "
        "AND type(r) IN ['EXPRESSES','HAS_CNV','HAS_METHYLATION','HAS_PROTEIN_ABUNDANCE'] "
        "RETURN count(DISTINCT n) AS c"
    ).data()
    o_genes = int((omics_gene_count[0].get("c") if omics_gene_count else 0) or 0)
    omics_gene_ratio = o_genes / max(1, total_gene_count)

    return {
        "sample_size": len(gene_list),
        "degree_stats": {
            "min": min(deg_map.values()), "max": max(deg_map.values()),
            "mean": round(sum(deg_map.values()) / len(deg_map), 1),
            "median": sorted(deg_map.values())[len(deg_map) // 2],
        },
        "hop1_source": hop1_source,
        "hop1_relation": hop1_rel,
        "hop2_source": hop2_src,
        "hop3_protein_bridge": hop3_src,
        "omics_coverage": omics_coverage,
        "seed_omics": seed_omics,
        "knowledge_vs_omics": {
            "knowledge_gene_ratio": round(knowledge_gene_ratio, 4),
            "omics_gene_ratio": round(omics_gene_ratio, 4),
            "omics_genes_covered_global": o_genes,
            "note": "知识源基因比={}（抽样{}个基因中{}个有知识源邻居），组学源基因比={}（全图{}基因中{}个有组学数据）".format(
                knowledge_gene_ratio, len(gene_list), k_genes,
                omics_gene_ratio, total_gene_count, o_genes),
        },
        "seed_detail": {
            g: {
                "degree": deg_map.get(g, 0),
                "hop1_top_sources": dict(sorted(source_by_gene[g].items(), key=lambda x: -x[1])[:5]),
                "hop2_reachable_genes": sum(hop2_source[g].values()),
                "hop3_via_protein": sum(hop3_source[g].values()),
            }
            for g in SEED_GENES
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tests/exp4_ablation.json")
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()

    if Graph is None:
        print("[ERROR] 需要 py2neo"); return

    g = Graph(host=config.KG_HOST, port=config.KG_PORT,
              user=config.KG_USERNAME, password=config.KG_PASSWORD)
    g.run("RETURN 1 AS ok").data()

    n = args.sample_size
    print(f"分层抽样 {n} 个基因 + 多跳路径分析...")
    genes = sample_genes_stratified(g, n)
    print(f"  抽样: {len(genes)} 基因")
    results = analyze_multihop_contribution(g, genes)

    d = results["degree_stats"]
    print(f"  度分布: min={d['min']} max={d['max']} mean={d['mean']} median={d['median']}")

    # ── 实验四：1-hop 来源 ──
    print("\n" + "=" * 70)
    print("实验四：数据源 1-hop 邻居贡献（直接边）")
    print("=" * 70)
    print(f"{'数据源':<23s} {'总贡献':>8s} {'平均占比':>8s} {'±CI95':>8s} {'覆盖'}")
    print("-" * 60)
    for s, info in results["hop1_source"].items():
        bar = "#" * int(info["avg_ratio"] * 40)
        print(f"  {s:<21s} {info['total']:>8,d} {info['avg_ratio']:>7.1%} "
              f"±{info['ci95']:>5.1%} {info['coverage']:>4d}  {bar}")

    # ── 实验四增强：2-hop 路径来源 ──
    print("\n" + "=" * 70)
    print("实验四增强：2-hop 路径中的边来源（gene→protein/pathway→gene）")
    print("=" * 70)
    for s, info in results["hop2_source"].items():
        bar = "#" * int(info["avg_ratio"] * 30)
        print(f"  {s:<21s} {info['total']:>8,d} {info['avg_ratio']:>7.1%} "
              f"±{info['ci95']:>5.1%} {info['coverage']:>4d}  {bar}")

    # ── STRING 蛋白桥 ──
    print("\n" + "=" * 70)
    print("STRING 蛋白桥贡献：gene→protein→protein→gene 路径中的桥来源")
    print("=" * 70)
    for s, info in results["hop3_protein_bridge"].items():
        bar = "#" * int(info["avg_ratio"] * 30)
        print(f"  {s:<21s} {info['total']:>8,d} {info['avg_ratio']:>7.1%} "
              f"±{info['ci95']:>5.1%} {info['coverage']:>4d}  {bar}")

    # ── 实验五：模态贡献 ──
    print("\n" + "=" * 70)
    print("实验五：关系类型贡献（1-hop）")
    print("=" * 70)
    modal = {
        "EXPRESSES": "mRNA/miRNA表达", "REGULATES": "miRNA靶标",
        "HAS_METHYLATION": "甲基化", "HAS_CNV": "CNV",
        "HAS_PROTEIN_ABUNDANCE": "蛋白质组", "ACTIVATES": "激活",
        "INHIBITS": "抑制", "ASSOCIATED_WITH": "通用关联",
        "PARTICIPATES": "通路参与",
    }
    for rt, info in results["hop1_relation"].items():
        bar = "#" * int(info["avg_ratio"] * 40)
        m = modal.get(rt, "")
        print(f"  {rt:<21s} {info['total']:>8,d} {info['avg_ratio']:>7.1%} "
              f"±{info['ci95']:>5.1%} [{m}]{'  ' + bar if bar else ''}")

    # ── 种子基因详情 ──
    print(f"\n{'基因':<10s} {'1-hop度':>7s} {'2-hop可达':>8s} {'蛋白桥':>7s}  Top来源")
    print("-" * 70)
    for g in SEED_GENES:
        sd = results["seed_detail"][g]
        tops = ", ".join(f"{s}({n})" for s, n in list(sd["hop1_top_sources"].items())[:3])
        print(f"  {g:<8s} {sd['degree']:>7,d} {sd['hop2_reachable_genes']:>8,d} "
              f"{sd['hop3_via_protein']:>7,d}  {tops}")

    # ── 实验五维度二：组学模态覆盖 ──
    oc = results["omics_coverage"]
    print("\n" + "=" * 70)
    print("实验五维度二：组学模态对基因的定量覆盖（Sample→Gene 边）")
    print("=" * 70)
    total_omics_edges = sum(v["edges"] for v in oc.values())
    print(f"{'模态':<25s} {'边数':>12s} {'覆盖基因数':>10s} {'占全基因比'}")
    print("-" * 60)
    for mod, info in oc.items():
        bar = "#" * int(info["edges"] / max(1, total_omics_edges) * 40)
        pct = info["genes_covered"] / max(1, 981131)  # 全图基因数
        print(f"  {mod:<23s} {info['edges']:>12,d} {info['genes_covered']:>10,d} "
              f"{pct:>7.1%}  {bar}")

    print(f"\n  组学边总计: {total_omics_edges:,}  覆盖基因: "
          f"{max(v['genes_covered'] for v in oc.values()):,} (unique union)")

    print(f"\n种子基因组学覆盖:")
    for g in SEED_GENES:
        sm = results["seed_omics"].get(g, {})
        parts = [f"{m}={s}" for m, s in sorted(sm.items(), key=lambda x: -x[1])]
        text = ", ".join(parts) if parts else "(none)"
        print(f"  {g:<8s} {text}")
    kv = results["knowledge_vs_omics"]
    print(f"\n知识源基因覆盖比: {kv['knowledge_gene_ratio']:.1%}  "
          f"（{len(genes)}个基因中有多少至少有一个知识源邻居）")
    print(f"组学源基因覆盖比: {kv['omics_gene_ratio']:.1%}  "
          f"（有多少至少有一个组学数据点）")
    print(f"注：知识源和组学源是不同维度的贡献，不直接相加，各源单独占比见上表")

    report = {"experiment": "消融 & 模态贡献（多跳版）",
              "timestamp": datetime.now(timezone.utc).isoformat(), "results": results}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
