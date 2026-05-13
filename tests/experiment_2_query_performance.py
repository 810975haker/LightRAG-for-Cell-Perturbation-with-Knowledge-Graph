"""
======================================================================
实验二：子图查询性能评估
======================================================================
测量不同 (max_nodes, max_edges) 参数下的子图查询响应延迟和结果完整性。
对比 Neo4j vs NetworkX 双后端性能。
======================================================================
用法：
  python tests/experiment_2_query_performance.py
  python tests/experiment_2_query_performance.py --output exp2_report.json
======================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.knowledge_graph.graph_store import KnowledgeGraphStore


# ── 测试参数网格 ──
NODE_SIZES = [50, 100, 200, 500, 1000]
EDGE_SIZES = [100, 300, 500, 1000, 2000]
REPEAT = 5  # 每组参数重复次数

# ── 测试查询关键词（覆盖不同匹配密度） ──
QUERIES = [
    {"node_keyword": "EGFR", "relation_keyword": ""},
    {"node_keyword": "EGFR", "relation_keyword": "activates"},
    {"node_keyword": "TP53", "relation_keyword": ""},
    {"node_keyword": "", "relation_keyword": "inhibits"},
    {"node_keyword": "pathway::", "relation_keyword": ""},
]


def measure_subgraph(store, node_kw: str, rel_kw: str,
                     max_nodes: int, max_edges: int) -> Dict[str, Any]:
    """单次测量并返回延迟(ms)与结果计数。"""
    t0 = time.perf_counter()
    result = store.subgraph_data(
        node_keyword=node_kw, relation_keyword=rel_kw,
        query_mode="or", view_mode="replace",
        max_nodes=max_nodes, max_edges=max_edges,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "latency_ms": round(elapsed_ms, 2),
        "nodes_returned": len(result.get("nodes", [])),
        "edges_returned": len(result.get("edges", [])),
        "match_stats": result.get("meta", {}).get("match_stats", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="子图查询性能评估实验")
    parser.add_argument("--output", default="tests/exp2_query_performance.json")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：仅测试主要参数组合")
    args = parser.parse_args()

    store = KnowledgeGraphStore()
    backend = store.backend

    print(f"后端: {backend}")
    print(f"测试参数: max_nodes={NODE_SIZES}, max_edges={EDGE_SIZES}")
    print(f"查询关键词: {len(QUERIES)} 组")
    print(f"每组重复: {REPEAT} 次")
    print()

    results: List[Dict] = []

    if args.quick:
        node_sizes = [100, 500, 1000]
        edge_sizes = [300, 1000, 2000]
        queries = QUERIES[:2]
    else:
        node_sizes = NODE_SIZES
        edge_sizes = EDGE_SIZES
        queries = QUERIES

    total = len(queries) * len(node_sizes) * len(edge_sizes)
    idx = 0

    for q in queries:
        for mn in node_sizes:
            for me in edge_sizes:
                idx += 1
                latencies = []
                sample_result = None
                for r in range(REPEAT):
                    res = measure_subgraph(store, q["node_keyword"], q["relation_keyword"], mn, me)
                    latencies.append(res["latency_ms"])
                    if r == 0:
                        sample_result = res

                avg_lat = round(sum(latencies) / len(latencies), 2)
                mn_lat = round(min(latencies), 2)
                mx_lat = round(max(latencies), 2)

                entry = {
                    "node_keyword": q["node_keyword"],
                    "relation_keyword": q["relation_keyword"],
                    "max_nodes": mn,
                    "max_edges": me,
                    "latency_avg_ms": avg_lat,
                    "latency_min_ms": mn_lat,
                    "latency_max_ms": mx_lat,
                    "nodes_returned": sample_result["nodes_returned"],
                    "edges_returned": sample_result["edges_returned"],
                    "match_stats": sample_result["match_stats"],
                }
                results.append(entry)

                bar = "!" if sample_result["edges_returned"] >= me else " "
                print(f"  [{idx:3d}/{total}] n={mn:4d} e={me:4d} | "
                      f"avg={avg_lat:8.2f}ms | "
                      f"nodes={sample_result['nodes_returned']:4d} edges={sample_result['edges_returned']:5d} {bar}",
                      flush=True)

    # ── 汇总统计 ──
    all_lat = [r["latency_avg_ms"] for r in results]
    report = {
        "experiment": "子图查询性能评估",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "parameters": {
            "node_sizes": node_sizes,
            "edge_sizes": edge_sizes,
            "queries": [f"{q['node_keyword']}|{q['relation_keyword']}" for q in queries],
            "repeat": REPEAT,
        },
        "summary": {
            "total_tests": len(results),
            "latency_min_ms": round(min(all_lat), 2),
            "latency_max_ms": round(max(all_lat), 2),
            "latency_mean_ms": round(sum(all_lat) / len(all_lat), 2),
            "latency_p99_ms": round(sorted(all_lat)[int(len(all_lat) * 0.99)], 2),
        },
        "results": sorted(results, key=lambda x: (x["max_edges"], x["max_nodes"])),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n报告已保存: {out_path}")
    print(f"延迟范围: {report['summary']['latency_min_ms']} - {report['summary']['latency_max_ms']} ms")
    print(f"平均延迟: {report['summary']['latency_mean_ms']} ms")
    print(f"P99 延迟: {report['summary']['latency_p99_ms']} ms")


if __name__ == "__main__":
    main()
