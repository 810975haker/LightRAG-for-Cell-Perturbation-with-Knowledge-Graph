from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.knowledge_graph.graph_store import KnowledgeGraphStore
from src.rag.light_rag import LightRAG


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate top-gene overlap among perturbations in the same cell")
    parser.add_argument("--cell-id", default="A549")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--perturb-limit", type=int, default=8)
    args = parser.parse_args()

    store = KnowledgeGraphStore()
    if store.backend != "neo4j" or store.graph is None:
        print("Neo4j backend unavailable")
        return 1

    rag = LightRAG(graph_store=store)
    rows = store.query_graph(
        "MATCH (p:Entity) "
        "WHERE p.name STARTS WITH 'pathway::perturbation::' "
        "RETURN p.name AS name ORDER BY p.name LIMIT {}".format(max(2, int(args.perturb_limit)))
    )
    perturbations = [str(row.get("name", "")).replace("pathway::perturbation::", "") for row in rows]
    if len(perturbations) < 2:
        print("Not enough perturbation nodes for overlap evaluation")
        return 0

    gene_sets = {}
    for p in perturbations:
        result = rag.predict_perturbation(
            cell_id=args.cell_id,
            perturbation=p,
            top_k=max(1, int(args.top_k)),
        )
        genes = set(result.get("top_genes", []))
        gene_sets[p] = genes

    print("[Top-gene overlap by perturbation]")
    for p, genes in gene_sets.items():
        print("{} -> {}".format(p, sorted(list(genes))))

    print("\n[Pairwise Jaccard]")
    for a, b in combinations(perturbations, 2):
        score = jaccard(gene_sets.get(a, set()), gene_sets.get(b, set()))
        print("{} vs {} => {:.4f}".format(a, b, score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

