from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.knowledge_graph.graph_store import KnowledgeGraphStore
from src.knowledge_graph.lung_cancer_etl import (
    canonical_relation,
    canonicalize_entity_id,
    deduplicate_triples,
    derive_perturbation_triples,
    infer_effect_sign_from_relation,
    infer_node_type,
    is_seed_entity,
    load_seed_genes,
    parse_manifest_downloads,
    write_triples_csv,
)
from src.knowledge_graph.omics_etl import build_omics_triples, load_omics_manifest


def _build_gene_symbol_dict(triples, seed_genes):
    symbols = {str(g).upper() for g in (seed_genes or set()) if str(g).strip()}
    for triple in triples:
        for entity in (triple.head, triple.tail):
            text = str(entity or "")
            if text.startswith("gene::"):
                symbols.add(text.split("::", 1)[-1].upper())
    return symbols


def _extract_pathway_name(raw_tail: str) -> str:
    text = str(raw_tail or "")
    if "::" in text:
        text = text.split("::", 1)[-1]
    return str(text).replace("_", " ").strip()


def _normalize_relation_by_types(relation: str, head_type: str, tail_type: str) -> str:
    rel = str(relation or "associated_with")
    if rel == "belongs_to":
        if tail_type == "Cell" and head_type in {"gene", "protein", "pathway", "sample", "mirna"}:
            return "belongs_to"
        return "associated_with"
    if rel == "participates":
        if head_type in {"gene", "protein"} and tail_type == "pathway":
            return "participates"
        return "associated_with"
    if rel in {"activates", "inhibits"}:
        if head_type in {"gene", "protein"} and tail_type in {"gene", "protein"}:
            return rel
        return "associated_with"
    if rel in {"expresses", "has_protein_abundance", "has_cnv", "has_methylation"}:
        if head_type in {"sample", "Cell"} and tail_type in {"gene", "protein", "mirna"}:
            return rel
        return "associated_with"
    if rel == "has_condition":
        if head_type in {"sample", "Cell"} and tail_type == "condition":
            return rel
        return "associated_with"
    if rel == "regulates":
        if head_type == "mirna" and tail_type in {"gene", "protein"}:
            return rel
        return "associated_with"
    # 扰动推导产生的关系类型
    if rel.startswith("affects_") or rel.startswith("has_perturbation") or rel.startswith("targets_"):
        return rel
    if rel == "associated_with":
        return "associated_with"
    return "associated_with"


def build_nx_graph(triples, seed_genes):
    graph = nx.MultiDiGraph()
    gene_symbols = _build_gene_symbol_dict(triples, seed_genes)
    for triple in triples:
        head = canonicalize_entity_id(triple.head, gene_symbols=gene_symbols)
        tail = canonicalize_entity_id(triple.tail, gene_symbols=gene_symbols)
        relation = canonical_relation(triple.relation)
        if not head or not tail:
            continue

        if not graph.has_node(head):
            graph.add_node(
                head,
                type=infer_node_type(head),
                is_seed=is_seed_entity(head, seed_genes),
            )
        if not graph.has_node(tail):
            graph.add_node(
                tail,
                type=infer_node_type(tail),
                is_seed=is_seed_entity(tail, seed_genes),
            )
        head_type = str(graph.nodes[head].get("type", "Cell"))
        tail_type = str(graph.nodes[tail].get("type", "Cell"))

        # Normalize cell context as explicit belongs_to edges: entity -> Cell.
        if head_type == "Cell" and tail_type in {"gene", "protein", "pathway"}:
            head, tail = tail, head
            head_type, tail_type = tail_type, "Cell"
            relation = "belongs_to"
        elif tail_type == "Cell" and head_type in {"gene", "protein", "pathway"}:
            relation = "belongs_to"

        relation = _normalize_relation_by_types(relation, head_type, tail_type)

        if str(triple.relation) == "pathway_name" and head.startswith("pathway::"):
            pathway_name = _extract_pathway_name(triple.tail)
            if pathway_name:
                graph.nodes[head]["pathway_name"] = pathway_name
            continue

        graph.add_edge(
            head,
            tail,
            relation=relation,
            source=triple.source,
            version=triple.version,
            evidence=triple.evidence,
            weight=float(getattr(triple, "weight", 1.0)),
            confidence=float(getattr(triple, "confidence", 0.5)),
            effect_sign=float(
                getattr(triple, "effect_sign", 0.0)
                if abs(float(getattr(triple, "effect_sign", 0.0) or 0.0)) > 0.001
                else infer_effect_sign_from_relation(triple.relation)
            ),
        )
    return graph


def iter_normalized_rows(triples, seed_genes):
    gene_symbols = _build_gene_symbol_dict(triples, seed_genes)
    for triple in triples:
        head = canonicalize_entity_id(triple.head, gene_symbols=gene_symbols)
        tail = canonicalize_entity_id(triple.tail, gene_symbols=gene_symbols)
        relation = canonical_relation(triple.relation)
        if not head or not tail:
            continue

        head_type = infer_node_type(head)
        tail_type = infer_node_type(tail)

        # Normalize cell context as explicit belongs_to edges: entity -> Cell.
        if head_type == "Cell" and tail_type in {"gene", "protein", "pathway"}:
            head, tail = tail, head
            head_type, tail_type = tail_type, "Cell"
            relation = "belongs_to"
        elif tail_type == "Cell" and head_type in {"gene", "protein", "pathway"}:
            relation = "belongs_to"

        relation = _normalize_relation_by_types(relation, head_type, tail_type)

        if str(triple.relation) == "pathway_name" and head.startswith("pathway::"):
            pathway_name = _extract_pathway_name(triple.tail)
            if pathway_name:
                yield {
                    "node_update": True,
                    "node_name": head,
                    "pathway_name": pathway_name,
                }
            continue

        yield {
            "head": head,
            "tail": tail,
            "relation": relation,
            "source": triple.source,
            "version": triple.version,
            "evidence": triple.evidence,
            "weight": float(getattr(triple, "weight", 1.0)),
            "confidence": float(getattr(triple, "confidence", 0.5)),
            "effect_sign": float(getattr(triple, "effect_sign", infer_effect_sign_from_relation(triple.relation)) or 0.0),
            "head_type": head_type,
            "tail_type": tail_type,
            "head_is_seed": is_seed_entity(head, seed_genes),
            "tail_is_seed": is_seed_entity(tail, seed_genes),
        }


def main():
    parser = argparse.ArgumentParser(description="Build mixed-source lung cancer KG and load into Neo4j")
    parser.add_argument(
        "--manifest",
        default="data/raw/lung_cancer/download_manifest.json",
        help="Path to download manifest JSON",
    )
    parser.add_argument(
        "--out-triples",
        default="data/processed/lung_cancer/kg_triples.csv",
        help="Output CSV for normalized triples",
    )
    parser.add_argument(
        "--omics-manifest",
        default="",
        help="Optional JSON manifest describing omics matrices to ingest",
    )
    parser.add_argument("--replace", action="store_true", help="Replace existing graph before loading")
    parser.add_argument("--streaming-neo4j", action="store_true", help="Stream triples directly into Neo4j")
    parser.add_argument("--skip-csv", action="store_true", help="Skip writing triples CSV file")
    parser.add_argument("--max-rows-per-source", type=int, default=200000)
    parser.add_argument("--min-string-score", type=int, default=700)
    parser.add_argument("--disable-perturbation-augmentation", action="store_true")
    parser.add_argument("--max-derived-affected-per-seed", type=int, default=120)
    parser.add_argument("--max-shared-pathway-2hop-per-seed", type=int, default=80)
    parser.add_argument("--min-pathway-edge-strength", type=float, default=0.45)
    parser.add_argument("--min-shared-pathway-score", type=float, default=0.5)
    parser.add_argument("--shared-pathway-keep-all", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError("Manifest not found: {}".format(manifest_path))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed_file = Path(manifest.get("seed_gene_file", "")) if manifest.get("seed_gene_file") else None
    seed_genes = load_seed_genes(seed_file)

    triples = parse_manifest_downloads(
        manifest,
        max_rows_per_source=args.max_rows_per_source,
        min_string_score=args.min_string_score,
    )

    omics_count = 0
    if args.omics_manifest:
        omics_path = Path(args.omics_manifest)
        if omics_path.exists():
            omics_specs = load_omics_manifest(omics_path)
            omics_triples = build_omics_triples(omics_specs)
            omics_count = len(omics_triples)
            triples = deduplicate_triples(list(triples) + list(omics_triples))

    derived_count = 0
    if not args.disable_perturbation_augmentation:
        derived_triples = derive_perturbation_triples(
            triples,
            seed_genes=seed_genes,
            max_affected_per_seed=args.max_derived_affected_per_seed,
            max_shared_pathway_2hop_per_seed=args.max_shared_pathway_2hop_per_seed,
            min_pathway_edge_strength=args.min_pathway_edge_strength,
            min_shared_pathway_score=args.min_shared_pathway_score,
            shared_pathway_keep_best_per_gene=(not args.shared_pathway_keep_all),
        )
        derived_count = len(derived_triples)
        triples = deduplicate_triples(list(triples) + list(derived_triples))

    if not triples:
        print("No triples generated from manifest. Check downloads and parsing settings.")
        return

    out_path = Path(args.out_triples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_csv:
        write_triples_csv(triples, out_path)

    store = KnowledgeGraphStore()
    if args.streaming_neo4j and store.backend == "neo4j":
        store.save_triples_streaming(iter_normalized_rows(triples, seed_genes), replace=args.replace)
        seed_count = len(seed_genes)
        graph_nodes = "streaming"
        graph_edges = "streaming"
    else:
        graph = build_nx_graph(triples, seed_genes)
        store.save_graph(graph, replace=args.replace)
        seed_count = len([node for node, data in graph.nodes(data=True) if data.get("is_seed")])
        graph_nodes = graph.number_of_nodes()
        graph_edges = graph.number_of_edges()

    print("Triples written:", len(triples))
    print("Derived perturbation triples:", derived_count)
    print("Omics triples:", omics_count)
    print("Triples CSV:", out_path if not args.skip_csv else "(skipped)")
    print("Graph nodes:", graph_nodes)
    print("Graph edges:", graph_edges)
    print("Seed-marked nodes:", seed_count)
    print("Configured backend:", store.backend)


if __name__ == "__main__":
    main()

