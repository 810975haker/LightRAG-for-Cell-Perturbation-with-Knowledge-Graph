from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import KG_HOST, KG_PASSWORD, KG_PORT, KG_USERNAME
from src.knowledge_graph.neo4j_validator import load_seed_genes, run_all_checks


try:
    from py2neo import Graph
except Exception:
    Graph = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Neo4j KG integrity for lung-cancer pipeline")
    parser.add_argument("--seed-file", default="", help="CSV file with gene column")
    parser.add_argument(
        "--manifest",
        default="data/raw/lung_cancer/download_manifest.json",
        help="Manifest path used to resolve seed_gene_file when --seed-file is empty",
    )
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    args = parser.parse_args()

    if Graph is None:
        print("py2neo is not available; cannot validate Neo4j graph.")
        return 2

    try:
        graph = Graph(host=KG_HOST, port=KG_PORT, user=KG_USERNAME, password=KG_PASSWORD)
    except Exception as exc:
        print("Cannot connect to Neo4j: {}".format(exc))
        return 2

    seed_genes = load_seed_genes(seed_file=args.seed_file or None, manifest_file=args.manifest or None)
    report = run_all_checks(graph, seed_genes)

    print(json.dumps(report, ensure_ascii=True, indent=2))

    if report.get("passed", False):
        return 0
    if args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

