from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.knowledge_graph.graph_store import KnowledgeGraphStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantify perturbation-node sparsity against non-seed targets")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    store = KnowledgeGraphStore()
    if store.backend != "neo4j" or store.graph is None:
        print("Neo4j backend unavailable")
        return 1

    query = (
        "MATCH (p:Entity)-[r]->(n:Entity) "
        "WHERE p.name STARTS WITH 'pathway::perturbation::' "
        "WITH p, "
        "sum(CASE WHEN n.name STARTS WITH 'gene::' AND coalesce(n.is_seed,false)=false THEN 1 ELSE 0 END) AS non_seed_gene_edges, "
        "sum(CASE WHEN n.name STARTS WITH 'protein::' THEN 1 ELSE 0 END) AS protein_edges, "
        "count(DISTINCT CASE WHEN n.name STARTS WITH 'gene::' AND coalesce(n.is_seed,false)=false THEN n END) AS non_seed_genes, "
        "count(DISTINCT CASE WHEN n.name STARTS WITH 'protein::' THEN n END) AS proteins "
        "RETURN p.name AS perturbation, non_seed_genes, proteins, non_seed_gene_edges, protein_edges "
        "ORDER BY non_seed_genes ASC, proteins ASC, perturbation "
        "LIMIT $limit"
    )
    rows = store.graph.run(query, limit=max(1, int(args.limit))).data()

    print("[Perturbation sparsity: low to high coverage]")
    for row in rows:
        print(
            "{} | non_seed_genes={} proteins={} gene_edges={} protein_edges={}".format(
                row.get("perturbation", ""),
                int(row.get("non_seed_genes", 0) or 0),
                int(row.get("proteins", 0) or 0),
                int(row.get("non_seed_gene_edges", 0) or 0),
                int(row.get("protein_edges", 0) or 0),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
