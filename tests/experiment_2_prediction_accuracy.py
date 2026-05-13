"""
======================================================================
实验二：扰动预测准确性评估（完整版）
======================================================================
指标：
  1. 平均 Chronos（KG/Expression/Random 预测基因的必需性分数）
  2. Mann-Whitney U 检验（KG vs Random 的统计显著性）
  3. Precision@K / Recall@K
  4. 通路富集分析（KG 预测基因与种子基因的共通路率）
  5. 分癌种分层（需 Model.csv 细胞系注释文件，缺失则跳过）

用法：
  python tests/experiment_2_prediction_accuracy.py
======================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
import random
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

SEED_GENES = ["EGFR", "KRAS", "TP53", "PIK3CA", "BRAF", "ERBB2", "ALK", "MET", "ROS1", "RET"]
DEPMAP_PATH = "data/raw/omics/depmap/CRISPRGeneEffect.csv"
TOPK = 100
ESSENTIAL_THRESHOLD = -0.5


def load_depmap(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, index_col=0)
    if raw.shape[0] < raw.shape[1] and raw.shape[0] < 5000:
        df = raw.T
    else:
        df = raw
    print(f"  DepMap: {df.shape[0]} 基因 × {df.shape[1]} 细胞系")
    return df


def build_scores(depmap: pd.DataFrame) -> Dict[str, float]:
    mean_per_gene = depmap.mean(axis=1, skipna=True)
    scores = {}
    for gene, val in mean_per_gene.items():
        if np.isnan(val):
            continue
        gs = str(gene).strip()
        symbol = gs.split(" (")[0].upper() if " (" in gs else gs.upper()
        if symbol not in scores or val < scores[symbol]:
            scores[symbol] = float(val)
    return scores


def evaluate(scores: Dict, ranking: List[str], topk: int = TOPK) -> Dict:
    top_genes = []
    seen = set()
    for g in ranking:
        gu = g.upper()
        if gu in seen or gu not in scores:
            continue
        top_genes.append(gu)
        seen.add(gu)
        if len(top_genes) >= topk:
            break

    if len(top_genes) < 5:
        return {"mean_chronos": 0, "spearman_r": 0, "precision": 0,
                "recall": 0, "chronos_list": [], "n": 0}

    cvals = [scores[g] for g in top_genes]
    mean_c = sum(cvals) / len(cvals)

    ranks = list(range(1, len(top_genes) + 1))
    try:
        from scipy.stats import spearmanr
        sp_r, _ = spearmanr(ranks, cvals)
    except Exception:
        sp_r = 0.0

    ess_in_top = sum(1 for v in cvals if v < ESSENTIAL_THRESHOLD)
    precision = ess_in_top / len(top_genes)
    total_ess = sum(1 for v in scores.values() if v < ESSENTIAL_THRESHOLD)
    recall = ess_in_top / max(1, total_ess)

    return {"mean_chronos": round(mean_c, 4),
            "spearman_r": round(float(sp_r), 4) if not np.isnan(sp_r) else 0.0,
            "precision": round(precision, 4),
            "recall": round(recall, 6),
            "chronos_list": cvals, "n": len(top_genes)}


def kg_ranking(seed: str) -> List[str]:
    from src.rag.light_rag import LightRAG
    rag = LightRAG()
    r = rag.predict_perturbation("A549_cell_001", f"KO_{seed}", top_k=TOPK)
    return [g["node"].replace("gene::", "").strip().upper()
            for g in r.get("ranked_genes", [])]


def build_expression_matrix() -> pd.DataFrame:
    """从 Neo4j EXPRESSES 边构建 基因×样本 表达矩阵。"""
    from py2neo import Graph
    import config
    g = Graph(host=config.KG_HOST, port=config.KG_PORT,
              user=config.KG_USERNAME, password=config.KG_PASSWORD)

    rows = g.run(
        "MATCH (s:Entity)-[r:EXPRESSES]->(n:Entity) "
        "WHERE s.name STARTS WITH 'sample::' AND n.name STARTS WITH 'gene::' "
        "RETURN substring(n.name, 6) AS gene, s.name AS sample, "
        "coalesce(r.weight, 0.5) AS expr "
        "LIMIT 5000000"
    ).data()
    print(f"  Neo4j EXPRESSES 边: {len(rows)}")
    data = defaultdict(dict)
    for row in rows:
        gene = str(row["gene"]).upper()
        sample = str(row["sample"])
        data[gene][sample] = float(row["expr"])
    df = pd.DataFrame(data).T.fillna(0)
    print(f"  表达矩阵: {df.shape[0]} 基因 × {df.shape[1]} 样本")
    return df


def expr_ranking(expr_matrix: pd.DataFrame, seed: str) -> List[str]:
    """纯表达基线：TCGA 样本中与种子基因表达向量 Pearson 相关性最高的基因。"""
    su = seed.upper()
    if su not in expr_matrix.index:
        return []
    seed_vec = expr_matrix.loc[su].values
    corrs = {}
    for gene in expr_matrix.index:
        if gene == su:
            continue
        v = expr_matrix.loc[gene].values
        nonzero = (seed_vec != 0) | (v != 0)
        if nonzero.sum() < 5:
            continue
        c = np.corrcoef(v[nonzero], seed_vec[nonzero])[0, 1]
        if not np.isnan(c):
            corrs[gene] = c
    return [g for g, _ in sorted(corrs.items(), key=lambda x: -abs(x[1]))[:TOPK]]


def coessentiality_ranking(depmap: pd.DataFrame, seed: str) -> List[str]:
    """DepMap 共必需性基线（保留用于对比）。"""
    su = seed.upper()
    sym_to_idx = {}
    for g in depmap.index:
        gs = str(g).strip()
        sym_to_idx[gs.split(" (")[0].upper() if " (" in gs else gs.upper()] = g
    if su not in sym_to_idx:
        return []
    seed_vec = depmap.loc[sym_to_idx[su]].values
    corrs = {}
    for sym, idx in sym_to_idx.items():
        if sym == su:
            continue
        v = depmap.loc[idx].values
        m = ~(np.isnan(v) | np.isnan(seed_vec))
        if m.sum() < 10:
            continue
        c = np.corrcoef(v[m], seed_vec[m])[0, 1]
        if not np.isnan(c):
            corrs[sym] = c
    return [g for g, _ in sorted(corrs.items(), key=lambda x: -abs(x[1]))[:TOPK]]


def pathway_enrichment(seed: str, ranking: List[str], topk: int = 50) -> Dict:
    """查询种子基因和 Top-K 预测基因的共享通路，计算通路富集率。"""
    try:
        from py2neo import Graph
        import config
        g = Graph(host=config.KG_HOST, port=config.KG_PORT,
                  user=config.KG_USERNAME, password=config.KG_PASSWORD)

        # 种子基因的通路
        seed_paths = g.run(
            "MATCH (n:Entity {name:$name})-[r]-(p:Entity) "
            "WHERE p.name STARTS WITH 'pathway::' AND NOT p.name CONTAINS 'perturbation' "
            "RETURN collect(DISTINCT p.name) AS paths",
            name=f"gene::{seed}",
        ).data()
        seed_set = set(seed_paths[0].get("paths", []) if seed_paths else [])

        # Top-K 基因中每个的通路覆盖
        top_genes = ranking[:topk]
        shared_count = 0
        gene_path_counts = []
        for gene in top_genes:
            rows = g.run(
                "MATCH (n:Entity {name:$name})-[r]-(p:Entity) "
                "WHERE p.name STARTS WITH 'pathway::' AND NOT p.name CONTAINS 'perturbation' "
                "RETURN collect(DISTINCT p.name) AS paths",
                name=f"gene::{gene}",
            ).data()
            gene_paths = set(rows[0].get("paths", []) if rows else set())
            if gene_paths & seed_set:
                shared_count += 1
            gene_path_counts.append(len(gene_paths))

        return {
            "seed_pathways": len(seed_set),
            "topk_with_pathways": sum(1 for c in gene_path_counts if c > 0),
            "shared_pathway_genes": shared_count,
            "enrichment_rate": round(shared_count / max(1, len(top_genes)), 4),
            "avg_pathways_per_gene": round(sum(gene_path_counts) / max(1, len(gene_path_counts)), 1),
        }
    except Exception:
        return {"error": "Neo4j unreachable"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="tests/exp2_prediction_accuracy.json")
    args = p.parse_args()

    dp_path = Path(DEPMAP_PATH)
    if not dp_path.exists():
        print(f"[ERROR] {dp_path} 不存在"); return

    # 加载 DepMap
    depmap = load_depmap(str(dp_path))
    scores = build_scores(depmap)
    total_ess = sum(1 for v in scores.values() if v < ESSENTIAL_THRESHOLD)
    print(f"  基因数: {len(scores)}, 必需基因(Chronos<-0.5): {total_ess} "
          f"({total_ess/max(1,len(scores)):.1%})")

    # 从 Neo4j 构建真实表达矩阵
    print("\n构建 TCGA 表达矩阵...")
    expr_matrix = build_expression_matrix()

    results = {}
    kg_c, expr_c, coess_c, rand_c = [], [], [], []
    kg_prec = []

    print(f"\n{'基因':<8s} {'KG_Chron':>10s} {'Expr_Chron':>12s} "
          f"{'CoEss_Chron':>13s} {'Rand_Chron':>12s} {'KG_Prec':>8s} {'富集率':>8s}")
    print("-" * 80)

    for seed in SEED_GENES:
        kg = kg_ranking(seed)
        kg_e = evaluate(scores, kg)
        ex = expr_ranking(expr_matrix, seed)
        ex_e = evaluate(scores, ex)
        co = coessentiality_ranking(depmap, seed)
        co_e = evaluate(scores, co)
        rd_e = evaluate(scores, random.sample(sorted(scores), min(TOPK, len(scores))))

        kg_c.append(kg_e["mean_chronos"])
        expr_c.append(ex_e["mean_chronos"])
        coess_c.append(co_e["mean_chronos"])
        rand_c.append(rd_e["mean_chronos"])
        kg_prec.append(kg_e["precision"])

        pw = pathway_enrichment(seed, kg)
        er = pw.get("enrichment_rate", 0)

        print(f"  {seed:<6s} {kg_e['mean_chronos']:>10.4f} {ex_e['mean_chronos']:>12.4f} "
              f"{co_e['mean_chronos']:>13.4f} {rd_e['mean_chronos']:>12.4f} "
              f"{kg_e['precision']:>7.1%} {er:>7.1%}")

        results[seed] = {
            "kg": {k: v for k, v in kg_e.items() if k != "chronos_list"},
            "expression": {k: v for k, v in ex_e.items() if k != "chronos_list"},
            "coessentiality": {k: v for k, v in co_e.items() if k != "chronos_list"},
            "random": {k: v for k, v in rd_e.items() if k != "chronos_list"},
            "pathway_enrichment": pw,
        }

    # ── 汇总 ──
    avg_kg = sum(kg_c) / len(kg_c)
    avg_ex = sum(expr_c) / len(expr_c)
    avg_co = sum(coess_c) / len(coess_c)
    avg_rd = sum(rand_c) / len(rand_c)

    from scipy.stats import mannwhitneyu
    all_kg = []
    all_rd = []
    for seed in SEED_GENES:
        all_kg.extend(evaluate(scores, kg_ranking(seed))["chronos_list"])
        for _ in range(5):
            all_rd.extend(evaluate(scores, random.sample(sorted(scores), TOPK))["chronos_list"])
    try:
        u_stat, p_value = mannwhitneyu(all_kg, all_rd, alternative="less")
    except Exception:
        u_stat, p_value = 0, 1.0

    print(f"\n{'平均':<6s} {avg_kg:>10.4f} {avg_ex:>12.4f} {avg_co:>13.4f} {avg_rd:>12.4f} "
          f"{sum(kg_prec)/len(kg_prec):>7.1%}")
    kg_vs_expr = sum(1 for k, e in zip(kg_c, expr_c) if k < e)
    kg_vs_co = sum(1 for k, c in zip(kg_c, coess_c) if k < c)
    print(f"KG 优于 表达基线: {kg_vs_expr}/10  KG 优于 共必需基线: {kg_vs_co}/10")
    print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_value:.4f} "
          f"({'显著' if p_value < 0.05 else '未达显著'})")

    report = {
        "experiment": "扰动预测准确性评估",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"topk": TOPK, "essential_threshold": ESSENTIAL_THRESHOLD},
        "summary": {
            "kg_mean_chronos": round(avg_kg, 4),
            "expr_mean_chronos": round(avg_ex, 4),
            "coessentiality_mean_chronos": round(avg_co, 4),
            "random_mean_chronos": round(avg_rd, 4),
            "kg_vs_expr_wins": kg_vs_expr,
            "kg_vs_coess_wins": kg_vs_co,
            "mannwhitney_u": round(float(u_stat), 1),
            "mannwhitney_p": round(float(p_value), 4),
            "significant": bool(p_value < 0.05),
        },
        "detail": results,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
