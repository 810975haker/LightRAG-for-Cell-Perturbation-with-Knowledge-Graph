from __future__ import annotations

import argparse
from typing import Dict, List

try:
    from py2neo import Graph
except Exception:  # pragma: no cover
    Graph = None

import config


def _connect() -> Graph:
    if Graph is None:
        raise RuntimeError("py2neo is not available; install requirements.txt dependencies")
    return Graph(
        host=config.KG_HOST,
        port=config.KG_PORT,
        user=config.KG_USERNAME,
        password=config.KG_PASSWORD,
    )


def _merge_source_counts(rel_rows: List[Dict], node_rows: List[Dict]) -> List[Dict]:
    rel_map = {str(row.get("source", "unknown")): int(row.get("rel_count", 0)) for row in rel_rows}
    node_map = {str(row.get("source", "unknown")): int(row.get("entity_count", 0)) for row in node_rows}
    sources = sorted(set(rel_map) | set(node_map))
    return [
        {
            "source": source,
            "entity_count": node_map.get(source, 0),
            "rel_count": rel_map.get(source, 0),
        }
        for source in sources
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Neo4j KG counts by source/type")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows per section")
    args = parser.parse_args()

    graph = _connect()

    rel_rows = graph.run(
        "MATCH ()-[r]->() "
        "RETURN coalesce(r.source, 'unknown') AS source, count(r) AS rel_count "
        "ORDER BY rel_count DESC"
    ).data()

    node_rows = graph.run(
        "MATCH (a)-[r]->(b) "
        "WITH coalesce(r.source, 'unknown') AS source, collect(DISTINCT a) + collect(DISTINCT b) AS nodes "
        "UNWIND nodes AS n "
        "RETURN source, count(DISTINCT n) AS entity_count "
        "ORDER BY entity_count DESC"
    ).data()

    type_rows = graph.run(
        "MATCH (n:Entity) RETURN coalesce(n.type, 'unknown') AS type, count(*) AS count ORDER BY count DESC"
    ).data()

    rel_type_rows = graph.run(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
    ).data()

    source_counts = _merge_source_counts(rel_rows, node_rows)

    print("数据源: 实体数, 关系数")
    for row in (source_counts[: args.limit] if args.limit else source_counts):
        print("{}\t{}\t{}".format(row["source"], row["entity_count"], row["rel_count"]))

    print("\n实体类型: 数量")
    for row in (type_rows[: args.limit] if args.limit else type_rows):
        print("{}\t{}".format(row.get("type", "unknown"), row.get("count", 0)))

    print("\n关系类型: 数量")
    for row in (rel_type_rows[: args.limit] if args.limit else rel_type_rows):
        print("{}\t{}".format(row.get("type", "UNKNOWN"), row.get("count", 0)))


if __name__ == "__main__":
    main()

