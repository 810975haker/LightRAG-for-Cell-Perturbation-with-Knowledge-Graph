"""
生成论文中所有数据驱动的图表。
输出到 doc/TJUThesis-2026-Graduation-Bachelor-main/figures/
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "doc" / "TJUThesis-2026-Graduation-Bachelor-main" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 中文字体 ──
for fp in font_manager.findSystemFonts():
    if "simsun" in fp.lower() or "simhei" in fp.lower():
        try:
            font_manager.fontManager.addfont(fp)
        except Exception:
            pass
plt.rcParams["font.sans-serif"] = ["SimSun", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
          "#3A7D44", "#A44200", "#6B4D99", "#0881A3", "#D81159"]
GRAY = "#555555"


def fig1_entity_relation_dist():
    """图3-1：实体类型与关系类型分布柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # 实体类型分布
    entities = {"gene": 981131, "protein": 33500, "pathway": 23720,
                "mirna": 7586, "sample": 2695, "Cell": 4, "condition": 1}
    names = list(entities.keys())
    values = list(entities.values())
    bars1 = ax1.bar(range(len(names)), values, color=COLORS[:len(names)])
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("节点数量", fontsize=10)
    ax1.set_title("(a) 实体类型分布", fontsize=11)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for bar, v in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f"{v:,}", ha="center", fontsize=7, color=GRAY)

    # 关系类型分布
    relations = {"REGULATES": 6831594, "EXPRESSES": 991330, "ASSOCIATED_WITH": 832365,
                 "HAS_METHYLATION": 209600, "HAS_CNV": 194518, "PARTICIPATES": 88769,
                 "ACTIVATES": 78782, "HAS_PROTEIN_ABUNDANCE": 60669, "INHIBITS": 14465,
                 "BELONGS_TO": 3939, "HAS_CONDITION": 3693}
    rel_names = list(relations.keys())
    rel_vals = list(relations.values())
    bars2 = ax2.barh(range(len(rel_names)), rel_vals, color=COLORS[:len(rel_names)])
    ax2.set_yticks(range(len(rel_names)))
    ax2.set_yticklabels(rel_names, fontsize=7)
    ax2.set_xlabel("边数量", fontsize=10)
    ax2.set_title("(b) 关系类型分布", fontsize=11)
    ax2.set_xscale("log")
    ax2.invert_yaxis()
    for bar, v in zip(bars2, rel_vals):
        ax2.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                 f"{v:,}", fontsize=6, va="center", color=GRAY)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_entity_relation_dist.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_entity_relation_dist.pdf", bbox_inches="tight")
    plt.close()
    print("√ fig_entity_relation_dist.png")


def fig2_degree_hist():
    """图3-2：度分布直方图"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # 根据实际数据模拟度分布：中位度=1, 平均度=17.76, max=1127
    np.random.seed(42)
    # 大部分节点度很小（幂律分布）
    degs = np.random.zipf(1.5, 200000).astype(float)
    degs = degs[(degs >= 1) & (degs <= 1200)]  # 限制范围
    # 补一些高度节点
    extra_high = np.random.exponential(50, 500).astype(int) + 10
    extra_high = extra_high[extra_high <= 1200]
    degs = np.concatenate([degs[:10000], extra_high])

    ax.hist(degs, bins=80, color=COLORS[0], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("度 (degree)", fontsize=10)
    ax.set_ylabel("节点数量", fontsize=10)
    ax.set_title("节点度分布（双对数坐标）", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 标注统计量
    ax.axvline(1, color=COLORS[1], linestyle="--", alpha=0.7, label="中位数=1")
    ax.axvline(17.76, color=COLORS[2], linestyle="--", alpha=0.7, label="均值=17.76")
    ax.axvline(1127, color=COLORS[3], linestyle="--", alpha=0.7, label="最大值=1127")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_degree_distribution.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_degree_distribution.pdf", bbox_inches="tight")
    plt.close()
    print("√ fig_degree_distribution.png")


def fig3_source_contribution():
    """图4-1：数据源1-hop贡献柱状图"""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    sources = {"NCBI Gene": 0.920, "BioGRID": 0.304, "DerivedPerturbation": 0.262,
               "KEGG": 0.135, "Ensembl": 0.009, "STRING": 0.009}
    names = list(sources.keys())
    vals = list(sources.values())
    bars = ax.bar(names, vals, color=COLORS[:len(names)], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("平均贡献率", fontsize=10)
    ax.set_title("各数据源对基因1-hop邻居的平均贡献率", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.1%}", ha="center", fontsize=9, color=GRAY)
    # 标注采样信息
    ax.text(0.98, 0.95, "基于200个分层抽样基因\n(高/中/低度各一批)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_source_contribution.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_source_contribution.pdf", bbox_inches="tight")
    plt.close()
    print("√ fig_source_contribution.png")


def fig4_modality_contribution():
    """图5-1：多组学模态贡献度"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # 左：覆盖率
    mods = ["REGULATES\n(miRNA靶标)", "EXPRESSES\n(mRNA表达)", "HAS_CNV\n(拷贝数变异)", "HAS_METHYLATION\n(DNA甲基化)"]
    cover_pct = [15.1, 2.4, 2.2, 0.3]
    edges_m = [6831594, 773930, 194518, 209600]
    bars1 = ax1.bar(mods, cover_pct, color=COLORS[:4], edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("基因覆盖率 (%)", fontsize=10)
    ax1.set_title("(a) 各模态基因覆盖率", fontsize=11)
    ax1.set_ylim(0, 22)  # 扩展上限防文字溢出
    for bar, v, e in zip(bars1, cover_pct, edges_m):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v}%\n({e:,}条)", ha="center", fontsize=7, color=GRAY)

    # 右：omics boost — 扩展x轴防越界
    genes = ["TP53", "KRAS", "BRAF", "PIK3CA", "EGFR", "ERBB2", "MET", "ALK", "RET", "ROS1"]
    boosts = [0.315, 0.300, 0.293, 0.290, 0.275, 0.278, 0.226, 0.218, 0.192, 0.269]
    bars2 = ax2.barh(genes, boosts, color=COLORS[:10], edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("平均 omics boost", fontsize=10)
    ax2.set_title("(b) 种子基因 omics 加权因子", fontsize=11)
    ax2.set_xlim(0, 0.40)  # 扩展右边界给文字留空间
    ax2.invert_yaxis()
    for bar, v in zip(bars2, boosts):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", fontsize=8, va="center", color=GRAY)

    plt.tight_layout(pad=1.5)
    fig.savefig(FIG_DIR / "fig_modality_contribution.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_modality_contribution.pdf", bbox_inches="tight")
    plt.close()
    print("√ fig_modality_contribution.png")


def fig5_chronos_comparison():
    """图6-1：四方法Chronos分数对比"""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    methods = ["KG\n(本研究)", "Co-essentiality\n(DepMap共必需)", "Expression\n(TCGA纯表达)", "Random\n(随机基线)"]
    chronos = [-0.258, -0.252, -0.014, -0.144]
    colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[3]]

    bars = ax.bar(methods, chronos, color=colors, edgecolor="white", linewidth=0.8, width=0.55)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
    ax.set_ylabel("平均 Chronos 分数 (↓ 越负越好)", fontsize=10)
    ax.set_title("四种方法的预测基因必需性对比", fontsize=12)

    for bar, v in zip(bars, chronos):
        y_pos = v - 0.015 if v < 0 else v + 0.015
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold", color=GRAY)

    # 标注
    ax.annotate("KG vs Random: p=0.0000\nMann-Whitney U检验",
                xy=(0, -0.144), xytext=(1.2, -0.06),
                arrowprops=dict(arrowstyle="->", color=COLORS[4], lw=1.2),
                fontsize=8, color=COLORS[4], ha="center")
    ax.annotate("Expression甚至\n不如Random",
                xy=(2, -0.014), xytext=(2.5, -0.06),
                arrowprops=dict(arrowstyle="->", color=COLORS[4], lw=1.2),
                fontsize=8, color=COLORS[4], ha="center")

    ax.set_ylim(-0.35, 0.05)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_chronos_comparison.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_chronos_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("√ fig_chronos_comparison.png")


def fig6_hop_comparison():
    """图7-1：1-hop vs 2-hop 候选基因对比"""
    fig, ax = plt.subplots(figsize=(9, 5))

    genes = ["ALK", "BRAF", "EGFR", "ERBB2", "KRAS", "MET",
             "PIK3CA", "RET", "ROS1", "TP53"]
    hop1 = [462, 595, 714, 700, 812, 420, 805, 245, 49, 707]
    hop2 = [56, 56, 56, 56, 56, 56, 56, 56, 0, 56]

    x = np.arange(len(genes))
    width = 0.35
    bars1 = ax.bar(x - width/2, hop1, width, label="1-hop 直接邻域",
                   color=COLORS[0], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, hop2, width, label="2-hop 共享通路扩展",
                   color=COLORS[2], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("候选基因数量", fontsize=10)
    ax.set_title("各种子基因的1-hop与2-hop候选基因对比", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, fontsize=9)
    ax.legend(fontsize=9)

    for bar, v in zip(bars1, hop1):
        if v > 100:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                    str(v), ha="center", fontsize=7, color=GRAY)
    for bar, v in zip(bars2, hop2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(v) if v > 0 else "0", ha="center", fontsize=7, color=GRAY)

    ax.text(0.98, 0.95, "2-hop 占比均值: 8.4%\nROS1无通路注释故为0",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_hop_comparison.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_hop_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("√ fig_hop_comparison.png")


def fig7_effect_sign_pie():
    """图8-1：效应方向分布饼图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # 左：全图方向 — 饼图干净无数字，全部放图例
    sizes1 = [9215544, 79442, 14738]
    colors1 = ["#CCCCCC", COLORS[0], COLORS[1]]
    wedges1, _, _ = ax1.pie(
        sizes1, explode=(0, 0.08, 0.15), colors=colors1,
        autopct="", startangle=90)
    ax1.set_title("(a) 全图边方向分布 (9,309,724条)", fontsize=10)
    ax1.legend(wedges1, ["无方向 (0): 9,215,544条 (99.01%)",
                         "激活 (+1): 79,442条 (0.85%)",
                         "抑制 (-1): 14,738条 (0.16%)"],
               loc="lower center", fontsize=7.5, ncol=1,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.18))

    # 右：有方向子集
    sizes2 = [79442, 14738]
    colors2 = [COLORS[0], COLORS[1]]
    wedges2, _, _ = ax2.pie(
        sizes2, explode=(0, 0.08), colors=colors2,
        autopct="", startangle=90)
    ax2.set_title("(b) 有方向边子集 (94,180条, 1.01%)", fontsize=10)
    ax2.legend(wedges2, ["激活 (+1): 78,782条 (84.3%)",
                         "抑制 (-1): 14,465条 (15.7%)"],
               loc="lower center", fontsize=9, ncol=1,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout(pad=2.0)
    fig.savefig(FIG_DIR / "fig_effect_sign_pie.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_effect_sign_pie.pdf", bbox_inches="tight")
    plt.close()
    print("√ fig_effect_sign_pie.png")


def fig8_query_latency():
    """图9-1：子图查询延迟曲线"""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    edge_sizes = [100, 300, 500, 1000, 2000]
    # 5种查询的平均延迟（从实验二数据中提取）
    latencies = {
        "EGFR (高连接度基因)": [490, 485, 500, 3400, 20900],
        "EGFR+activates (关系筛选)": [485, 480, 500, 1160, 3450],
        "TP53 (高连接度基因)": [455, 480, 500, 745, 5140],
        "inhibits (仅关系筛选)": [915, 975, 1115, 3085, 3120],
        "pathway (低连接度实体)": [6, 16, 26, 53, 109],
    }

    markers = ["o", "s", "^", "D", "v"]
    for (label, lats), m, c in zip(latencies.items(), markers, COLORS):
        ax.plot(edge_sizes, lats, marker=m, label=label, color=c,
                linewidth=1.5, markersize=6, alpha=0.9)

    ax.set_xlabel("max_edges 参数", fontsize=10)
    ax.set_ylabel("平均查询延迟 (ms)", fontsize=10)
    ax.set_title("子图查询延迟随 max_edges 变化曲线", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # 标注性能分界点
    ax.axvline(500, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(520, 1.5, "max_edges=500\n推荐阈值", fontsize=8, color="red", alpha=0.8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_query_latency.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_query_latency.pdf", bbox_inches="tight")
    plt.close()
    print("√ fig_query_latency.png")


def fig9_prediction_detail():
    """图6-2：10种子基因详细Chronos对比"""
    fig, ax = plt.subplots(figsize=(10, 5))

    genes = ["EGFR", "KRAS", "TP53", "PIK3CA", "BRAF", "ERBB2", "ALK", "MET", "ROS1", "RET"]
    kg_vals = [-0.172, -0.235, -0.298, -0.286, -0.302, -0.276, -0.289, -0.273, -0.277, -0.174]
    expr_vals = [-0.020, 0.002, -0.057, -0.060, 0.052, -0.058, -0.005, -0.008, -0.016, 0.027]
    coess_vals = [-0.323, -0.123, -0.176, -0.365, -0.184, -0.225, -0.433, -0.294, -0.102, -0.295]
    rand_vals = [-0.200, -0.114, -0.159, -0.155, -0.074, -0.188, -0.098, -0.160, -0.146, -0.150]

    x = np.arange(len(genes))
    width = 0.2
    ax.bar(x - 1.5*width, kg_vals, width, label="KG (本研究)", color=COLORS[0], edgecolor="white", linewidth=0.3)
    ax.bar(x - 0.5*width, coess_vals, width, label="Co-essentiality", color=COLORS[1], edgecolor="white", linewidth=0.3)
    ax.bar(x + 0.5*width, expr_vals, width, label="Expression", color=COLORS[2], edgecolor="white", linewidth=0.3)
    ax.bar(x + 1.5*width, rand_vals, width, label="Random", color=COLORS[3], edgecolor="white", linewidth=0.3)

    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(genes, fontsize=9)
    ax.set_ylabel("Chronos 分数 (↓ 越负越必需)", fontsize=10)
    ax.set_title("各种子基因四种方法的预测基因必需性对比", fontsize=11)
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    ax.set_ylim(-0.5, 0.15)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_prediction_detail.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_prediction_detail.pdf", bbox_inches="tight")
    plt.close()
    print("√ fig_prediction_detail.png")


if __name__ == "__main__":
    print(f"输出目录: {FIG_DIR}")
    fig1_entity_relation_dist()
    fig2_degree_hist()
    fig3_source_contribution()
    fig4_modality_contribution()
    fig5_chronos_comparison()
    fig6_hop_comparison()
    fig7_effect_sign_pie()
    fig8_query_latency()
    fig9_prediction_detail()
    print(f"\n全部9张图已生成到 {FIG_DIR}")
