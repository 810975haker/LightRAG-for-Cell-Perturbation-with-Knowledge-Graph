from __future__ import annotations

import csv
import gzip
import json
import re
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd


@dataclass(frozen=True)
class KGTriple:
    head: str
    relation: str
    tail: str
    source: str
    version: str
    evidence: str
    weight: float = 1.0
    confidence: float = 0.5
    effect_sign: float = 0.0


def build_evidence(raw: str, structured: Dict) -> str:
    return json.dumps({"raw": raw, "structured": structured}, ensure_ascii=True)


def normalize_entity(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return "_".join(text.split())


def canonicalize_entity_id(entity_id: str, gene_symbols: Optional[Set[str]] = None) -> str:
    text = str(entity_id or "").strip()
    if not text:
        return ""
    prefix, value = (text.split("::", 1) + [""])[:2] if "::" in text else ("", text)
    base = normalize_entity(value or text)
    pfx = prefix.lower()
    gene_dict = gene_symbols or set()
    upper = base.upper()

    if pfx in {"cell", "cell_context"}:
        return "Cell::{}".format(base)
    if pfx in {"sample"}:
        return "sample::{}".format(base)
    if pfx in {"condition"}:
        return "condition::{}".format(base)
    if pfx in {"mirna", "mi_rna", "micro_rna"}:
        return "mirna::{}".format(base.upper())
    if pfx in {"protein"}:
        return "protein::{}".format(base)
    if pfx in {"pathway", "term"}:
        return "pathway::{}".format(base)
    if pfx in {"gene", "ncbi_gene", "ensembl_gene"}:
        return "gene::{}".format(base.upper())
    if pfx in {"rna_or_protein"}:
        if upper in gene_dict:
            return "gene::{}".format(upper)
        if upper.startswith("ENSP") or base.startswith("9606."):
            return "protein::{}".format(base)
        return "gene::{}".format(upper)

    if upper in gene_dict:
        return "gene::{}".format(upper)
    if upper.startswith("ENSP") or base.startswith("9606."):
        return "protein::{}".format(base)
    if upper.startswith("R-HSA") or upper.startswith("WP") or upper.startswith("HSA"):
        return "pathway::{}".format(base)
    return "gene::{}".format(upper)


def canonical_relation(relation: str) -> str:
    raw = normalize_entity(relation).lower()
    if not raw:
        return "associated_with"
    # 扰动推导的专用关系名（affects_*, targets_*, has_perturbation_*）直接透传
    if raw.startswith("affects_") or raw.startswith("targets_") or raw.startswith("has_perturbation"):
        return raw
    if any(token in raw for token in ["activate", "activates", "upreg", "induce", "promote", "enhance", "positive"]):
        return "activates"
    if any(token in raw for token in ["inhibit", "suppres", "repress", "downreg", "decrease", "negative", "knockdown", "knockout"]):
        return "inhibits"
    if any(token in raw for token in ["belong", "belongs_to", "member_of"]):
        return "belongs_to"
    if any(token in raw for token in ["pathway", "participat", "member", "pathway_child", "pathway_name"]):
        return "participates"
    if any(token in raw for token in ["express", "expression", "transcript"]):
        return "expresses"
    if any(token in raw for token in ["methyl", "meth"]):
        return "has_methylation"
    if any(token in raw for token in ["cnv", "copy_number", "copy-number"]):
        return "has_cnv"
    if any(token in raw for token in ["protein_abundance", "proteomic", "rppa", "cptac"]):
        return "has_protein_abundance"
    if "condition" in raw:
        return "has_condition"
    if any(token in raw for token in ["regulat", "target"]):
        return "regulates"
    return "associated_with"


def infer_effect_sign_from_relation(relation: str) -> float:
    raw = normalize_entity(relation).lower()
    if any(token in raw for token in ["inhibit", "suppress", "repress", "down", "decrease", "negative"]):
        return -1.0
    if any(token in raw for token in ["activ", "enhanc", "induce", "up", "increase", "positive", "promote"]):
        return 1.0
    return 0.0


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: float, default: float = 0.5) -> float:
    x = _safe_float(value, default)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@contextmanager
def _open_text(path: Path):
    if path.suffix == ".gz":
        fh = gzip.open(path, "rt", encoding="utf-8", errors="ignore")
        try:
            yield fh
        finally:
            fh.close()
        return
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path, "r")
        try:
            text_members = [
                name
                for name in archive.namelist()
                if not name.endswith("/") and name.lower().endswith((".txt", ".tab", ".tsv", ".sif", ".csv"))
            ]
            # 优先选择人类相关文件（BioGRID ZIP 包含 82 个物种）
            human_members = [m for m in text_members if "homo_sapiens" in m.lower() or "9606" in m.lower()]
            member = human_members[0] if human_members else (text_members[0] if text_members else archive.namelist()[0])
            with archive.open(member, "r") as raw:
                with raw:
                    yield (line.decode("utf-8", errors="ignore") for line in raw)
        finally:
            archive.close()
        return
    fh = path.open("r", encoding="utf-8", errors="ignore")
    try:
        yield fh
    finally:
        fh.close()


def _as_pathway(pathway_id: str) -> str:
    return "pathway::{}".format(normalize_entity(pathway_id))


def _as_gene(gene: str) -> str:
    return "gene::{}".format(normalize_entity(gene).upper())


def _as_protein(protein: str) -> str:
    return "protein::{}".format(normalize_entity(protein))


def parse_reactome_pathway_relations(path: Path, source: str, version: str) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            parent_id = normalize_entity(row[0])
            child_id = normalize_entity(row[1])
            if not parent_id or not child_id:
                continue
            triples.append(
                KGTriple(
                    head=_as_pathway(child_id),
                    relation="pathway_child_of",
                    tail=_as_pathway(parent_id),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw="{}\t{}".format(parent_id, child_id),
                        structured={"parent": parent_id, "child": child_id},
                    ),
                    weight=0.6,
                    confidence=0.65,
                )
            )
    return triples


def parse_reactome_pathways(path: Path, source: str, version: str) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            pathway_id = normalize_entity(row[0])
            pathway_name = normalize_entity(row[1])
            species = row[2].strip().lower()
            if species != "homo sapiens" or not pathway_id:
                continue
            triples.append(
                KGTriple(
                    head=_as_pathway(pathway_id),
                    relation="pathway_name",
                    tail="term::{}".format(pathway_name),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw="\t".join(row[:3]),
                        structured={"pathway_id": pathway_id, "pathway_name": pathway_name, "species": row[2]},
                    ),
                    weight=0.4,
                    confidence=0.6,
                )
            )
    return triples


def parse_reactome_uniprot_pathway(
    path: Path,
    source: str,
    version: str,
    max_rows: int = 400000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            uniprot_id = normalize_entity(row[0])
            pathway_id = normalize_entity(row[1])
            species = str(row[5] or "").strip().lower()
            if species != "homo sapiens":
                continue
            if not uniprot_id or not pathway_id:
                continue

            triples.append(
                KGTriple(
                    head=_as_protein(uniprot_id),
                    relation="participates_in_pathway",
                    tail=_as_pathway(pathway_id),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw="\t".join(row[:6])[:500],
                        structured={
                            "uniprot": uniprot_id,
                            "pathway_id": pathway_id,
                            "pathway_name": normalize_entity(row[3]),
                            "species": row[5],
                        },
                    ),
                    weight=0.74,
                    confidence=0.82,
                )
            )

            # Preserve readable pathway label as node property triple.
            pathway_name = normalize_entity(row[3])
            if pathway_name:
                triples.append(
                    KGTriple(
                        head=_as_pathway(pathway_id),
                        relation="pathway_name",
                        tail="term::{}".format(pathway_name),
                        source=source,
                        version=version,
                        evidence=build_evidence(
                            raw="\t".join(row[:6])[:500],
                            structured={"pathway_id": pathway_id, "pathway_name": pathway_name},
                        ),
                        weight=0.45,
                        confidence=0.7,
                    )
                )

            if len(triples) >= max_rows:
                break
    return triples


def parse_string_ppi(
    path: Path,
    source: str,
    version: str,
    min_score: int = 700,
    max_rows: int = 200000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        header_skipped = False
        for idx, line in enumerate(fh):
            if not header_skipped:
                header_skipped = True
                if "protein1" in line and "protein2" in line:
                    continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            p1, p2, score_text = parts[0], parts[1], parts[2]
            try:
                score = int(score_text)
            except ValueError:
                continue
            if score < min_score:
                continue
            triples.append(
                KGTriple(
                    head=_as_protein(p1),
                    relation="protein_interacts_with",
                    tail=_as_protein(p2),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw="{} {} {}".format(p1, p2, score),
                        structured={"string_score": score},
                    ),
                    weight=round(min(1.0, score / 1000.0), 4),
                    confidence=round(min(1.0, score / 1000.0), 4),
                )
            )
            if len(triples) >= max_rows:
                break
            if idx >= max_rows * 5:
                break
    return triples


def parse_string_protein_info(
    path: Path,
    source: str,
    version: str,
    max_rows: int = 300000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            protein_id = normalize_entity(row.get("#string_protein_id") or row.get("string_protein_id") or "")
            preferred_name = normalize_entity(row.get("preferred_name") or "").upper()
            if not protein_id or not preferred_name:
                continue
            triples.append(
                KGTriple(
                    head=_as_gene(preferred_name),
                    relation="has_string_protein_id",
                    tail=_as_protein(protein_id),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw="{}\t{}".format(protein_id, preferred_name),
                        structured={"string_protein_id": protein_id, "preferred_name": preferred_name},
                    ),
                    weight=0.96,
                    confidence=0.92,
                )
            )
            if len(triples) >= max_rows:
                break
    return triples


def parse_string_protein_aliases(
    path: Path,
    source: str,
    version: str,
    max_rows: int = 500000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        header_skipped = False
        for row in reader:
            if not header_skipped:
                header_skipped = True
                continue
            if len(row) < 2:
                continue
            protein_id = normalize_entity(row[0])
            alias = normalize_entity(row[1]).upper()
            source_text = normalize_entity(row[2]).lower() if len(row) >= 3 else ""
            if not protein_id or not alias:
                continue
            if not any(ch.isalpha() for ch in alias):
                continue
            if len(alias) > 18:
                continue

            confidence = 0.82
            if "hgnc" in source_text or "ensembl_hgnc" in source_text:
                confidence = 0.9
            elif "uniprot" in source_text:
                confidence = 0.86

            triples.append(
                KGTriple(
                    head=_as_gene(alias),
                    relation="has_string_protein_id",
                    tail=_as_protein(protein_id),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw="\t".join(row[:3]),
                        structured={"string_protein_id": protein_id, "alias": alias, "alias_source": source_text},
                    ),
                    weight=0.88,
                    confidence=confidence,
                )
            )
            if len(triples) >= max_rows:
                break
    return triples


def parse_pathway_commons_sif(
    path: Path,
    source: str,
    version: str,
    max_rows: int = 200000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        for line in fh:
            row = line.strip()
            if not row or row.startswith("#"):
                continue
            parts = row.split("\t")
            if len(parts) < 3:
                continue
            left = normalize_entity(parts[0]).upper()
            relation = normalize_entity(parts[1]).lower()
            right = normalize_entity(parts[2]).upper()
            triples.append(
                KGTriple(
                    head=_as_gene(left),
                    relation=relation or "gene_related_to",
                    tail=_as_gene(right),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw=row[:500],
                        structured={"left": left, "relation": relation, "right": right},
                    ),
                    weight=0.7,
                    confidence=0.7,
                )
            )
            if len(triples) >= max_rows:
                break
    return triples


def parse_npinter(
    path: Path,
    source: str,
    version: str,
    max_rows: int = 150000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            left = normalize_entity(row[0])
            relation = normalize_entity(row[1]).lower() or "rna_protein_regulates"
            right = normalize_entity(row[2])
            if left.lower() in {"rna", "rna_interactor", "interactor1"}:
                continue
            left_u = left.upper()
            right_u = right.upper()
            triples.append(
                KGTriple(
                    head="rna_or_protein::{}".format(left),
                    relation=relation,
                    tail="rna_or_protein::{}".format(right),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw="|".join(row[:6])[:500],
                        structured={"left": left, "relation": relation, "right": right},
                    ),
                    weight=0.65,
                    confidence=0.68,
                )
            )
            if len(triples) >= max_rows:
                break
    return triples


def parse_kegg_gene_pathway(
    path: Path,
    source: str,
    version: str,
    query_gene: str = "",
    max_rows: int = 5000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    gene_fallback = normalize_entity(query_gene).upper()
    with _open_text(path) as fh:
        for line in fh:
            row = line.strip()
            if not row or "\t" not in row:
                continue
            left, right = row.split("\t", 1)
            left_token = left.split(":", 1)[-1]
            right_token = right.split(":", 1)[-1]
            parsed_gene = normalize_entity(left_token).upper()
            # Keep symbol in graph when query provides one, avoiding numeric-only labels (e.g., 1956).
            gene_token = gene_fallback or parsed_gene
            pathway_token = normalize_entity(right_token)
            if not gene_token or not pathway_token:
                continue
            triples.append(
                KGTriple(
                    head=_as_gene(gene_token),
                    relation="participates_in_pathway",
                    tail=_as_pathway("kegg_{}".format(pathway_token)),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw=row,
                        structured={
                            "gene": gene_token,
                            "kegg_gene_id": parsed_gene,
                            "pathway": pathway_token,
                            "query_gene": gene_fallback,
                        },
                    ),
                    weight=0.72,
                    confidence=0.75,
                )
            )
            if len(triples) >= max_rows:
                break
    return triples


def parse_wikipathways_query_json(
    path: Path,
    source: str,
    version: str,
    query_gene: str = "",
    max_rows: int = 3000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    gene_token = normalize_entity(query_gene).upper()
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return triples

    result_sets = []
    if isinstance(payload, dict):
        for key in ("result", "results"):
            if isinstance(payload.get(key), list):
                result_sets = payload[key]
                break

    for row in result_sets:
        if not isinstance(row, dict):
            continue
        pathway_id = normalize_entity(row.get("id") or row.get("wpid") or row.get("identifier"))
        if not pathway_id:
            continue
        pathway_name = normalize_entity(row.get("name") or row.get("title") or "")
        triples.append(
            KGTriple(
                head=_as_gene(gene_token or "UNKNOWN"),
                relation="participates_in_pathway",
                tail=_as_pathway("wp_{}".format(pathway_id)),
                source=source,
                version=version,
                evidence=build_evidence(
                    raw=json.dumps(row, ensure_ascii=True)[:500],
                    structured={"query_gene": gene_token, "pathway_id": pathway_id, "pathway_name": pathway_name},
                ),
                weight=0.66,
                confidence=0.7,
            )
        )
        if len(triples) >= max_rows:
            break

    return triples


def parse_encode_eclip_report(
    path: Path,
    source: str,
    version: str,
    max_rows: int = 120000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        # 跳过首行元数据（日期+URL），真正的表头在第二行
        first_line = fh.readline()
        if not first_line.strip():
            return triples
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames:
            return triples
        for row in reader:
            target = normalize_entity(
                row.get("Target of assay")
                or row.get("target.label")
                or row.get("target")
                or row.get("Target")
                or ""
            ).upper()
            biosample = normalize_entity(
                row.get("Biosample summary")
                or row.get("Biosample term name")
                or row.get("biosample_ontology.term_name")
                or row.get("Biosample")
                or ""
            )
            accession = normalize_entity(row.get("Accession") or row.get("accession") or "")
            if not target:
                continue
            tail = "cell_context::{}".format(biosample or accession or "unknown")
            triples.append(
                KGTriple(
                    head=_as_gene(target),
                    relation="supported_by_eclip",
                    tail=tail,
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw=json.dumps({"target": target, "biosample": biosample, "accession": accession}, ensure_ascii=True),
                        structured={"target": target, "biosample": biosample, "accession": accession},
                    ),
                    weight=0.74,
                    confidence=0.78,
                )
            )
            if len(triples) >= max_rows:
                break
    return triples


def parse_biogrid_tab3(
    path: Path,
    source: str,
    version: str,
    max_rows: int = 250000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            left = normalize_entity(row.get("Official Symbol Interactor A") or row.get("Systematic Name Interactor A") or "").upper()
            right = normalize_entity(row.get("Official Symbol Interactor B") or row.get("Systematic Name Interactor B") or "").upper()
            organism_a = str(row.get("Organism ID Interactor A") or "")
            organism_b = str(row.get("Organism ID Interactor B") or "")
            if organism_a and organism_a != "9606":
                continue
            if organism_b and organism_b != "9606":
                continue
            if not left or not right:
                continue
            method = normalize_entity(row.get("Experimental System") or "")
            triples.append(
                KGTriple(
                    head=_as_gene(left),
                    relation="biogrid_interacts_with",
                    tail=_as_gene(right),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw=json.dumps(row, ensure_ascii=True)[:500],
                        structured={"left": left, "right": right, "method": method},
                    ),
                    weight=0.76,
                    confidence=0.74,
                )
            )
            if len(triples) >= max_rows:
                break
    return triples


def _entity_from_type(name: str, type_hint: str) -> str:
    value = normalize_entity(name)
    hint = normalize_entity(type_hint).lower()
    if not value:
        return ""
    if "protein" in hint:
        return _as_protein(value)
    if "mirna" in hint or "micro" in hint:
        return "mirna::{}".format(normalize_entity(value).upper())
    if "sample" in hint:
        return "sample::{}".format(normalize_entity(value))
    if "condition" in hint:
        return "condition::{}".format(normalize_entity(value))
    if "gene" in hint:
        return _as_gene(value)
    return _as_gene(value)


def parse_signor_signed_relations(
    path: Path,
    source: str,
    version: str,
    max_rows: int = 300000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        # SIGNOR 导出为 TSV（无表头），列顺序：entity_a, type_a, id_a, db_a, entity_b, type_b, id_b, db_b, effect, mechanism, ...
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 9:
                continue
            a_name = str(row[0] or "").strip()
            a_type = str(row[1] or "").strip().lower() if len(row) > 1 else "gene"
            b_name = str(row[4] or "").strip() if len(row) > 4 else ""
            b_type = str(row[5] or "").strip().lower() if len(row) > 5 else "gene"
            effect = normalize_entity(row[8] if len(row) > 8 else "").lower()
            organism = str(row[11] or "").strip() if len(row) > 11 else ""
            pmid = str(row[17] or "").strip() if len(row) > 17 else ""

            # 仅保留人类数据
            if organism and organism != "9606":
                continue
            if not a_name or not b_name:
                continue

            head = _entity_from_type(a_name, a_type)
            tail = _entity_from_type(b_name, b_type)
            if not head or not tail:
                continue

            relation = "associated_with"
            if any(token in effect for token in ["up-regulates", "upregulates", "activat", "stimul", "expression"]):
                relation = "activates"
            elif any(token in effect for token in ["down-regulates", "downregulates", "inhibit", "repress", "suppress"]):
                relation = "inhibits"

            if relation in {"activates", "inhibits"}:
                weight = 0.88
                confidence = 0.9 if pmid else 0.8
            else:
                weight = 0.7
                confidence = 0.72

            triples.append(
                KGTriple(
                    head=head,
                    relation=relation,
                    tail=tail,
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw="\t".join(row[:10])[:500],
                        structured={
                            "effect": effect,
                            "pmid": pmid,
                        },
                    ),
                    weight=weight,
                    confidence=confidence,
                )
            )
            if len(triples) >= max_rows:
                break
    return triples


def parse_omnipath_signed_relations(
    path: Path,
    source: str,
    version: str,
    max_rows: int = 400000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            left = normalize_entity(row.get("source") or row.get("source_genesymbol") or "").upper()
            right = normalize_entity(row.get("target") or row.get("target_genesymbol") or "").upper()
            if not left or not right:
                continue

            stim = str(row.get("is_stimulation") or row.get("stimulation") or "").lower() in {"1", "true", "yes"}
            inhib = str(row.get("is_inhibition") or row.get("inhibition") or "").lower() in {"1", "true", "yes"}
            consensus = str(row.get("consensus_direction") or "").lower() in {"1", "true", "yes"}

            relation = "associated_with"
            if stim and not inhib:
                relation = "activates"
            elif inhib and not stim:
                relation = "inhibits"

            weight = 0.86 if consensus else 0.78
            confidence = 0.87 if consensus else 0.78

            triples.append(
                KGTriple(
                    head=_as_gene(left),
                    relation=relation,
                    tail=_as_gene(right),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw=json.dumps(row, ensure_ascii=True)[:500],
                        structured={
                            "is_stimulation": stim,
                            "is_inhibition": inhib,
                            "consensus_direction": consensus,
                        },
                    ),
                    weight=weight,
                    confidence=confidence,
                )
            )
            if len(triples) >= max_rows:
                break
    return triples


def parse_ncbi_gene_info(
    path: Path,
    source: str,
    version: str,
    max_rows: int = 250000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    with _open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            tax_id = str(row.get("#tax_id") or row.get("tax_id") or "")
            if tax_id and tax_id != "9606":
                continue
            gene_id = normalize_entity(row.get("GeneID") or "")
            symbol = normalize_entity(row.get("Symbol") or "").upper()
            if not gene_id or not symbol:
                continue
            triples.append(
                KGTriple(
                    head=_as_gene(symbol),
                    relation="has_ncbi_gene_id",
                    tail="ncbi_gene::{}".format(gene_id),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw="{}\t{}".format(gene_id, symbol),
                        structured={"gene_id": gene_id, "symbol": symbol},
                    ),
                    weight=1.0,
                    confidence=0.95,
                )
            )
            if len(triples) >= max_rows:
                break
    return triples


def parse_ensembl_gtf(
    path: Path,
    source: str,
    version: str,
    max_rows: int = 300000,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    attr_gene_id = re.compile(r'gene_id "([^"]+)"')
    attr_gene_name = re.compile(r'gene_name "([^"]+)"')
    with _open_text(path) as fh:
        for line in fh:
            row = line.strip()
            if not row or row.startswith("#"):
                continue
            parts = row.split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != "gene":
                continue
            attrs = parts[8]
            gene_id_match = attr_gene_id.search(attrs)
            gene_name_match = attr_gene_name.search(attrs)
            if not gene_id_match or not gene_name_match:
                continue
            ensembl_gene = normalize_entity(gene_id_match.group(1))
            symbol = normalize_entity(gene_name_match.group(1)).upper()
            if not ensembl_gene or not symbol:
                continue
            triples.append(
                KGTriple(
                    head=_as_gene(symbol),
                    relation="has_ensembl_gene_id",
                    tail="ensembl_gene::{}".format(ensembl_gene),
                    source=source,
                    version=version,
                    evidence=build_evidence(
                        raw=row[:500],
                        structured={"symbol": symbol, "ensembl_gene": ensembl_gene},
                    ),
                    weight=1.0,
                    confidence=0.95,
                )
            )
            if len(triples) >= max_rows:
                break
    return triples


def _perturbation_node(method: str, target_gene: str) -> str:
    return "pathway::perturbation::{}_{}".format(normalize_entity(method).upper(), normalize_entity(target_gene).upper())


def _perturbation_method_node(method: str) -> str:
    return "pathway::perturb_method::{}".format(normalize_entity(method).upper())


def derive_perturbation_triples(
    base_triples: Sequence[KGTriple],
    seed_genes: Set[str],
    methods: Optional[Sequence[str]] = None,
    max_affected_per_seed: int = 120,
    max_shared_pathway_2hop_per_seed: int = 80,
    min_pathway_edge_strength: float = 0.45,
    min_shared_pathway_score: float = 0.5,
    shared_pathway_keep_best_per_gene: bool = True,
) -> List[KGTriple]:
    """Derive perturbation-centric edges for better intervention reasoning.

    All derived entities still map into the existing 4-type schema via canonicalization
    (`pathway::...` for perturbation concepts, `gene/protein/pathway/Cell` for targets).
    """
    seeds = {normalize_entity(g).upper() for g in (seed_genes or set()) if normalize_entity(g)}
    if not seeds:
        return []

    perturb_methods = [m.upper() for m in (methods or ["KO", "KD", "OE", "CRISPRI", "CRISPRA", "RNAI", "INHIBIT"])]
    gene_neighbors: Dict[str, List[KGTriple]] = {}
    gene_to_pathways: Dict[str, Set[str]] = {}
    pathway_to_genes: Dict[str, Set[str]] = {}
    gene_pathway_strength: Dict[tuple, float] = {}

    for triple in base_triples:
        head = canonicalize_entity_id(triple.head)
        tail = canonicalize_entity_id(triple.tail)
        if head.startswith("gene::"):
            g = head.split("::", 1)[-1]
            gene_neighbors.setdefault(g, []).append(triple)
        if tail.startswith("gene::"):
            g = tail.split("::", 1)[-1]
            gene_neighbors.setdefault(g, []).append(triple)

        rel = canonical_relation(triple.relation)
        if rel == "participates":
            if head.startswith("gene::") and tail.startswith("pathway::"):
                g = head.split("::", 1)[-1]
                gene_to_pathways.setdefault(g, set()).add(tail)
                pathway_to_genes.setdefault(tail, set()).add(head)
                edge_strength = _clamp01((float(getattr(triple, "weight", 0.5)) + float(getattr(triple, "confidence", 0.5))) / 2.0)
                key = (g, tail)
                gene_pathway_strength[key] = max(edge_strength, gene_pathway_strength.get(key, 0.0))
            elif tail.startswith("gene::") and head.startswith("pathway::"):
                g = tail.split("::", 1)[-1]
                gene_to_pathways.setdefault(g, set()).add(head)
                pathway_to_genes.setdefault(head, set()).add(tail)
                edge_strength = _clamp01((float(getattr(triple, "weight", 0.5)) + float(getattr(triple, "confidence", 0.5))) / 2.0)
                key = (g, head)
                gene_pathway_strength[key] = max(edge_strength, gene_pathway_strength.get(key, 0.0))

    derived: List[KGTriple] = []
    for seed in sorted(seeds):
        linked = gene_neighbors.get(seed, [])
        linked_sorted = sorted(linked, key=lambda t: float(getattr(t, "weight", 0.0) or 0.0), reverse=True)

        for method in perturb_methods:
            pert_node = _perturbation_node(method, seed)
            method_node = _perturbation_method_node(method)

            derived.append(
                KGTriple(
                    head=pert_node,
                    relation="has_perturbation_method",
                    tail=method_node,
                    source="DerivedPerturbation",
                    version="v1",
                    evidence=build_evidence(
                        raw="{}->{}".format(pert_node, method_node),
                        structured={"seed_gene": seed, "method": method},
                    ),
                    weight=0.95,
                    confidence=0.9,
                )
            )
            derived.append(
                KGTriple(
                    head=pert_node,
                    relation="targets_gene",
                    tail="gene::{}".format(seed),
                    source="DerivedPerturbation",
                    version="v1",
                    evidence=build_evidence(
                        raw="{} targets {}".format(pert_node, seed),
                        structured={"seed_gene": seed, "method": method},
                    ),
                    weight=1.0,
                    confidence=0.92,
                )
            )

            affected_count = 0
            for origin in linked_sorted:
                h = canonicalize_entity_id(origin.head, gene_symbols=seeds)
                t = canonicalize_entity_id(origin.tail, gene_symbols=seeds)
                if h == "gene::{}".format(seed):
                    affected = t
                elif t == "gene::{}".format(seed):
                    affected = h
                else:
                    continue

                if not affected or affected == "gene::{}".format(seed):
                    continue
                if not (
                    affected.startswith("gene::")
                    or affected.startswith("protein::")
                    or affected.startswith("pathway::")
                    or affected.startswith("Cell::")
                ):
                    continue

                rel = canonical_relation(origin.relation)
                base_weight = float(getattr(origin, "weight", 0.6) or 0.6)
                base_conf = float(getattr(origin, "confidence", 0.6) or 0.6)
                derived.append(
                    KGTriple(
                        head=pert_node,
                        relation="affects_{}".format(rel),
                        tail=affected,
                        source="DerivedPerturbation",
                        version="v1",
                        evidence=build_evidence(
                            raw="derived from {} {} {}".format(origin.head, origin.relation, origin.tail),
                            structured={
                                "seed_gene": seed,
                                "method": method,
                                "derived_from": {
                                    "head": origin.head,
                                    "relation": origin.relation,
                                    "tail": origin.tail,
                                    "source": origin.source,
                                },
                            },
                        ),
                        weight=round(max(0.2, min(1.0, base_weight * 0.85)), 4),
                        confidence=round(max(0.2, min(1.0, base_conf * 0.85)), 4),
                    )
                )
                affected_count += 1
                if affected_count >= max_affected_per_seed:
                    break

            # Shared-pathway 2-hop expansion:
            # seed_gene -> pathway <- other_gene  => perturbation affects other_gene
            shared_paths = sorted(gene_to_pathways.get(seed, set()))
            hop2_count = 0
            hop2_candidates: List[Dict] = []
            for pathway_node in shared_paths:
                seed_path_strength = _clamp01(gene_pathway_strength.get((seed, pathway_node), 0.0), default=0.0)
                if seed_path_strength < min_pathway_edge_strength:
                    continue
                for other_gene_node in sorted(pathway_to_genes.get(pathway_node, set())):
                    if other_gene_node == "gene::{}".format(seed):
                        continue
                    other_gene = other_gene_node.split("::", 1)[-1]
                    other_path_strength = _clamp01(gene_pathway_strength.get((other_gene, pathway_node), 0.0), default=0.0)
                    if other_path_strength < min_pathway_edge_strength:
                        continue

                    pathway_score = _clamp01((seed_path_strength * other_path_strength) ** 0.5)
                    if pathway_score < min_shared_pathway_score:
                        continue

                    dynamic_weight = _clamp01(0.58 * (0.7 + 0.6 * pathway_score), default=0.58)
                    dynamic_confidence = _clamp01(0.62 * (0.7 + 0.6 * pathway_score), default=0.62)

                    hop2_candidates.append(
                        {
                            "tail": other_gene_node,
                            "via_pathway": pathway_node,
                            "pathway_score": pathway_score,
                            "seed_path_strength": seed_path_strength,
                            "other_path_strength": other_path_strength,
                            "weight": dynamic_weight,
                            "confidence": dynamic_confidence,
                        }
                    )

            if shared_pathway_keep_best_per_gene:
                best_by_gene: Dict[str, Dict] = {}
                for cand in hop2_candidates:
                    tail = cand["tail"]
                    if tail not in best_by_gene or cand["pathway_score"] > best_by_gene[tail]["pathway_score"]:
                        best_by_gene[tail] = cand
                selected_candidates = sorted(best_by_gene.values(), key=lambda x: x["pathway_score"], reverse=True)
            else:
                selected_candidates = sorted(hop2_candidates, key=lambda x: x["pathway_score"], reverse=True)

            for cand in selected_candidates:
                derived.append(
                    KGTriple(
                        head=pert_node,
                        relation="affects_shared_pathway_2hop",
                        tail=cand["tail"],
                        source="DerivedPerturbation",
                        version="v1",
                        evidence=build_evidence(
                            raw="{}->{}<-{}".format(seed, cand["via_pathway"], cand["tail"]),
                            structured={
                                "seed_gene": seed,
                                "method": method,
                                "via_pathway": cand["via_pathway"],
                                "hop": 2,
                                "pathway_score": round(float(cand["pathway_score"]), 6),
                                "seed_path_strength": round(float(cand["seed_path_strength"]), 6),
                                "other_path_strength": round(float(cand["other_path_strength"]), 6),
                            },
                        ),
                        weight=round(float(cand["weight"]), 4),
                        confidence=round(float(cand["confidence"]), 4),
                    )
                )
                hop2_count += 1
                if hop2_count >= max_shared_pathway_2hop_per_seed:
                    break

    return deduplicate_triples(derived)


def deduplicate_triples(triples: Sequence[KGTriple]) -> List[KGTriple]:
    seen = set()
    deduped: List[KGTriple] = []
    for triple in triples:
        key = (triple.head, triple.relation, triple.tail, triple.source, triple.version)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(triple)
    return deduped


def triples_to_dataframe(triples: Sequence[KGTriple]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "head": t.head,
                "relation": canonical_relation(t.relation),
                "raw_relation": t.relation,
                "tail": t.tail,
                "source": t.source,
                "version": t.version,
                "evidence": t.evidence,
                "weight": _clamp01(t.weight, default=1.0),
                "confidence": _clamp01(t.confidence, default=0.5),
                "effect_sign": float(getattr(t, "effect_sign", infer_effect_sign_from_relation(t.relation)) or 0.0),
            }
            for t in triples
        ]
    )


def write_triples_csv(triples: Sequence[KGTriple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "head",
        "relation",
        "raw_relation",
        "tail",
        "source",
        "version",
        "evidence",
        "weight",
        "confidence",
        "effect_sign",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for t in triples:
            writer.writerow(
                {
                    "head": t.head,
                    "relation": canonical_relation(t.relation),
                    "raw_relation": t.relation,
                    "tail": t.tail,
                    "source": t.source,
                    "version": t.version,
                    "evidence": t.evidence,
                    "weight": _clamp01(t.weight, default=1.0),
                    "confidence": _clamp01(t.confidence, default=0.5),
                    "effect_sign": float(getattr(t, "effect_sign", infer_effect_sign_from_relation(t.relation)) or 0.0),
                }
            )


def load_seed_genes(seed_gene_file: Optional[Path]) -> Set[str]:
    if not seed_gene_file or not seed_gene_file.exists():
        return set()
    df = pd.read_csv(seed_gene_file)
    if "gene" not in df.columns:
        return set()
    return {normalize_entity(g).upper() for g in df["gene"].astype(str).tolist()}


def parse_manifest_downloads(
    manifest: Dict,
    max_rows_per_source: int = 200000,
    min_string_score: int = 700,
) -> List[KGTriple]:
    triples: List[KGTriple] = []
    for item in manifest.get("downloads", []):
        if item.get("status") != "ok":
            continue
        path = Path(item.get("path", ""))
        if not path.exists():
            continue
        name = item.get("name", "")
        parser_name = item.get("parser", "")
        source = item.get("source", "unknown")
        version = item.get("version", "unknown")
        query_gene = item.get("query_gene", "")

        if parser_name == "reactome_pathway_relations" or name == "reactome_pathway_relations":
            triples.extend(parse_reactome_pathway_relations(path, source=source, version=version))
        elif parser_name == "reactome_pathways" or name == "reactome_pathways":
            triples.extend(parse_reactome_pathways(path, source=source, version=version))
        elif parser_name == "reactome_uniprot_pathway" or name == "reactome_uniprot_pathway":
            triples.extend(
                parse_reactome_uniprot_pathway(
                    path,
                    source=source,
                    version=version,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "string_ppi" or name == "string_human_ppi":
            triples.extend(
                parse_string_ppi(
                    path,
                    source=source,
                    version=version,
                    min_score=min_string_score,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "pathway_commons_hgnc_sif" or name == "pathway_commons_hgnc_sif":
            triples.extend(
                parse_pathway_commons_sif(
                    path,
                    source=source,
                    version=version,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "string_protein_info":
            triples.extend(
                parse_string_protein_info(
                    path,
                    source=source,
                    version=version,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "string_protein_aliases":
            triples.extend(
                parse_string_protein_aliases(
                    path,
                    source=source,
                    version=version,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "npinter" or name == "npinter_rna_interaction":
            triples.extend(
                parse_npinter(
                    path,
                    source=source,
                    version=version,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "kegg_gene_pathway" or name.startswith("kegg_gene_pathway_"):
            triples.extend(
                parse_kegg_gene_pathway(
                    path,
                    source=source,
                    version=version,
                    query_gene=query_gene,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "wikipathways_query_json" or name.startswith("wikipathways_gene_query_"):
            triples.extend(
                parse_wikipathways_query_json(
                    path,
                    source=source,
                    version=version,
                    query_gene=query_gene,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "encode_eclip_report":
            triples.extend(
                parse_encode_eclip_report(
                    path,
                    source=source,
                    version=version,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "biogrid_tab3":
            triples.extend(
                parse_biogrid_tab3(
                    path,
                    source=source,
                    version=version,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "signor_signed_relations":
            triples.extend(
                parse_signor_signed_relations(
                    path,
                    source=source,
                    version=version,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "omnipath_signed_relations":
            triples.extend(
                parse_omnipath_signed_relations(
                    path,
                    source=source,
                    version=version,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "ncbi_gene_info":
            triples.extend(
                parse_ncbi_gene_info(
                    path,
                    source=source,
                    version=version,
                    max_rows=max_rows_per_source,
                )
            )
        elif parser_name == "ensembl_gtf":
            triples.extend(
                parse_ensembl_gtf(
                    path,
                    source=source,
                    version=version,
                    max_rows=max_rows_per_source,
                )
            )
    return deduplicate_triples(triples)


def infer_node_type(entity_id: str) -> str:
    canonical = canonicalize_entity_id(entity_id)
    if canonical.startswith("gene::"):
        return "gene"
    if canonical.startswith("protein::"):
        return "protein"
    if canonical.startswith("pathway::"):
        return "pathway"
    if canonical.startswith("mirna::"):
        return "mirna"
    if canonical.startswith("sample::"):
        return "sample"
    if canonical.startswith("condition::"):
        return "condition"
    return "Cell"


def is_seed_entity(entity_id: str, seed_genes: Set[str]) -> bool:
    if not seed_genes:
        return False
    canonical = canonicalize_entity_id(entity_id)
    if not canonical.startswith("gene::"):
        return False
    value = canonical.split("::", 1)[-1]
    return normalize_entity(value).upper() in seed_genes

