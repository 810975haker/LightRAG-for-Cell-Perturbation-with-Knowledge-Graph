from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

ALLOWED_RELATIONS = {
    "activates",
    "associated_with",
    "belongs_to",
    "expresses",
    "has_cnv",
    "has_condition",
    "has_methylation",
    "has_protein_abundance",
    "inhibits",
    "participates",
    "regulates",
}
ALLOWED_EDGE_KEYS = {"relation", "source", "version", "evidence", "weight", "confidence", "effect_sign"}


def normalize_seed(value: str) -> str:
    return str(value or "").strip().upper()


def entity_suffix(entity_name: str) -> str:
    if "::" in entity_name:
        return normalize_seed(entity_name.split("::", 1)[1])
    return normalize_seed(entity_name)


def load_seed_genes(seed_file: Optional[str] = None, manifest_file: Optional[str] = None) -> Set[str]:
    if seed_file:
        path = Path(seed_file)
    elif manifest_file:
        manifest_path = Path(manifest_file)
        if not manifest_path.exists():
            return set()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed_path = manifest.get("seed_gene_file")
        path = Path(seed_path) if seed_path else Path()
    else:
        return set()

    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "gene" not in df.columns:
        return set()
    return {normalize_seed(v) for v in df["gene"].astype(str).tolist() if normalize_seed(v)}


def check_seed_consistency(seed_genes: Set[str], seed_node_names: Iterable[str]) -> Dict:
    suffixes = {entity_suffix(name) for name in seed_node_names if entity_suffix(name)}
    missing = sorted(seed_genes - suffixes)
    extras = sorted(suffixes - seed_genes)
    return {
        "expected_seed_count": len(seed_genes),
        "db_seed_unique_count": len(suffixes),
        "missing_seeds": missing,
        "extra_seeds": extras,
        "passed": len(missing) == 0 and len(extras) == 0,
    }


def _relation_to_type(relation: str) -> str:
    text = str(relation or "").strip()
    if not text:
        return "RELATED_TO"
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", text).upper()
    normalized = normalized.strip("_") or "RELATED_TO"
    if normalized[0].isdigit():
        normalized = "REL_{}".format(normalized)
    return normalized


def check_relation_type(graph) -> Dict:
    allowed_types = sorted({_relation_to_type(rel) for rel in ALLOWED_RELATIONS} | {"RELATED_TO"})
    data = graph.run(
        "MATCH ()-[r]->() "
        "WHERE NOT type(r) IN $allowed_types "
        "RETURN count(r) AS invalid_count, collect(DISTINCT type(r))[0..10] AS invalid_types",
        allowed_types=allowed_types,
    ).data()
    row = data[0] if data else {"invalid_count": 0, "invalid_types": []}
    invalid_count = int(row.get("invalid_count", 0) or 0)
    return {
        "invalid_count": invalid_count,
        "invalid_types": row.get("invalid_types", []),
        "passed": invalid_count == 0,
    }


def check_edge_required_props(graph) -> Dict:
    data = graph.run(
        "MATCH (a)-[r]->(b) "
        "WHERE r.relation IS NULL OR trim(toString(r.relation)) = '' "
        "OR r.source IS NULL OR trim(toString(r.source)) = '' "
        "OR r.version IS NULL OR trim(toString(r.version)) = '' "
        "OR r.evidence IS NULL OR trim(toString(r.evidence)) = '' "
        "OR r.weight IS NULL "
        "OR r.confidence IS NULL "
        "RETURN count(r) AS missing_count, collect({head:a.name, tail:b.name})[0..10] AS samples"
    ).data()
    row = data[0] if data else {"missing_count": 0, "samples": []}
    missing_count = int(row.get("missing_count", 0) or 0)
    return {
        "missing_count": missing_count,
        "samples": row.get("samples", []),
        "passed": missing_count == 0,
    }


def check_relation_values(graph) -> Dict:
    data = graph.run(
        "MATCH ()-[r]->() "
        "WHERE NOT coalesce(r.relation, '') IN $allowed "
        "RETURN count(r) AS invalid_count, collect(DISTINCT r.relation)[0..20] AS invalid_values",
        allowed=sorted(ALLOWED_RELATIONS),
    ).data()
    row = data[0] if data else {"invalid_count": 0, "invalid_values": []}
    invalid_count = int(row.get("invalid_count", 0) or 0)
    return {
        "invalid_count": invalid_count,
        "invalid_values": row.get("invalid_values", []),
        "allowed_values": sorted(ALLOWED_RELATIONS),
        "passed": invalid_count == 0,
    }


def check_edge_property_keys(graph) -> Dict:
    data = graph.run(
        "MATCH ()-[r]->() "
        "WHERE any(k IN keys(r) WHERE NOT k IN $allowed_keys) "
        "RETURN count(r) AS invalid_count, collect(keys(r))[0..10] AS samples",
        allowed_keys=sorted(ALLOWED_EDGE_KEYS),
    ).data()
    row = data[0] if data else {"invalid_count": 0, "samples": []}
    invalid_count = int(row.get("invalid_count", 0) or 0)
    return {
        "invalid_count": invalid_count,
        "allowed_keys": sorted(ALLOWED_EDGE_KEYS),
        "samples": row.get("samples", []),
        "passed": invalid_count == 0,
    }


def fetch_seed_nodes(graph) -> List[str]:
    data = graph.run(
        "MATCH (n:Entity) WHERE coalesce(n.is_seed, false) = true "
        "RETURN collect(n.name) AS names"
    ).data()
    if not data:
        return []
    return [str(x) for x in (data[0].get("names") or [])]


def run_all_checks(graph, seed_genes: Set[str]) -> Dict:
    relation = check_relation_type(graph)
    edge_props = check_edge_required_props(graph)
    relation_values = check_relation_values(graph)
    edge_keys = check_edge_property_keys(graph)

    seed = {"skipped": True, "passed": True}
    if seed_genes:
        seed_nodes = fetch_seed_nodes(graph)
        seed = check_seed_consistency(seed_genes, seed_nodes)
        seed["skipped"] = False
        seed["seed_node_count"] = len(seed_nodes)

    return {
        "relation_type": relation,
        "relation_values": relation_values,
        "edge_props": edge_props,
        "edge_keys": edge_keys,
        "seed_consistency": seed,
        "passed": relation["passed"] and relation_values["passed"] and edge_props["passed"] and edge_keys["passed"] and seed.get("passed", True),
    }

