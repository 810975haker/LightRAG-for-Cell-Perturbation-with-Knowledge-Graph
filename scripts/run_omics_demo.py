from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.knowledge_graph.omics_etl import build_omics_triples, load_omics_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build demo omics triples from a manifest")
    parser.add_argument(
        "--manifest",
        default="data/processed/lung_cancer/omics_manifest.sample.json",
        help="Path to omics manifest JSON",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError("Manifest not found: {}".format(manifest_path))

    specs = load_omics_manifest(manifest_path)
    triples = build_omics_triples(specs)

    print("Omics datasets:", len(specs))
    print("Omics triples:", len(triples))
    for triple in triples[:8]:
        print("{} {} {} [{}]".format(triple.head, triple.relation, triple.tail, triple.source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

