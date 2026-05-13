from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

import config
from src.data.matrix_loader import MatrixLoader
from src.knowledge_graph.lung_cancer_etl import KGTriple, build_evidence, normalize_entity


@dataclass
class OmicsMatrixSpec:
    name: str
    modality: str
    matrix_path: str
    matrix_format: str
    genes_path: Optional[str] = None
    cells_path: Optional[str] = None
    sample_metadata_path: Optional[str] = None
    sample_id_column: str = "sample_id"
    cell_line_column: str = "cell_line"
    condition_column: str = "condition"
    source: str = "unknown"
    version: str = "unknown"
    min_value: float = 0.0
    top_k_per_sample: int = 0
    confidence: float = 0.7
    target_map_path: Optional[str] = None
    target_map_delimiter: str = "\t"

    @staticmethod
    def from_dict(payload: Dict) -> "OmicsMatrixSpec":
        return OmicsMatrixSpec(
            name=str(payload.get("name", "omics_dataset")),
            modality=str(payload.get("modality", "expression")),
            matrix_path=str(payload.get("matrix_path", "")),
            matrix_format=str(payload.get("matrix_format", "csv")),
            genes_path=payload.get("genes_path"),
            cells_path=payload.get("cells_path"),
            sample_metadata_path=payload.get("sample_metadata_path"),
            sample_id_column=str(payload.get("sample_id_column", "sample_id")),
            cell_line_column=str(payload.get("cell_line_column", "cell_line")),
            condition_column=str(payload.get("condition_column", "condition")),
            source=str(payload.get("source", "unknown")),
            version=str(payload.get("version", "unknown")),
            min_value=float(payload.get("min_value", 0.0) or 0.0),
            top_k_per_sample=int(payload.get("top_k_per_sample", 0) or 0),
            confidence=float(payload.get("confidence", 0.7) or 0.7),
            target_map_path=payload.get("target_map_path"),
            target_map_delimiter=str(payload.get("target_map_delimiter", "\t")),
        )


def load_omics_manifest(path: Path) -> List[OmicsMatrixSpec]:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    items = payload.get("omics", []) if isinstance(payload, dict) else []
    return [OmicsMatrixSpec.from_dict(item) for item in items if isinstance(item, dict)]


def _load_sample_metadata(spec: OmicsMatrixSpec) -> Dict[str, Dict[str, str]]:
    if not spec.sample_metadata_path:
        return {}
    meta_path = Path(spec.sample_metadata_path)
    if not meta_path.exists():
        return {}
    df = pd.read_csv(meta_path)
    if spec.sample_id_column not in df.columns:
        return {}
    mapping: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        sample_id = str(row.get(spec.sample_id_column, "") or "").strip()
        if not sample_id:
            continue
        mapping[sample_id] = {
            "cell_line": str(row.get(spec.cell_line_column, "") or "").strip(),
            "condition": str(row.get(spec.condition_column, "") or "").strip(),
        }
    return mapping


def _load_expression_frame(spec: OmicsMatrixSpec) -> pd.DataFrame:
    loader = MatrixLoader()
    return loader._load_expression(
        matrix_path=spec.matrix_path,
        matrix_format=spec.matrix_format,
        genes_path=spec.genes_path,
        cells_path=spec.cells_path,
        delimiter=str(getattr(config, "DEFAULT_MATRIX_DELIMITER", ",")),
    )


def _relation_for_modality(modality: str) -> str:
    text = str(modality or "").strip().lower()
    if text in {"expression", "rna", "transcriptome"}:
        return "expresses"
    if text in {"proteomics", "protein", "rppa", "cptac"}:
        return "has_protein_abundance"
    if text in {"methylation", "meth"}:
        return "has_methylation"
    if text in {"cnv", "copy_number", "copy-number"}:
        return "has_cnv"
    if text in {"mirna", "mi_rna", "micro_rna"}:
        return "expresses"
    return "associated_with"


def _entity_prefix_for_modality(modality: str) -> str:
    text = str(modality or "").strip().lower()
    if text in {"proteomics", "protein", "rppa", "cptac"}:
        return "protein"
    if text in {"mirna", "mi_rna", "micro_rna"}:
        return "mirna"
    return "gene"


def _sample_node(sample_id: str) -> str:
    return "sample::{}".format(normalize_entity(sample_id))


def _entity_node(prefix: str, value: str) -> str:
    return "{}::{}".format(prefix, normalize_entity(value).upper())


def _condition_node(condition: str) -> str:
    return "condition::{}".format(normalize_entity(condition))


def build_omics_triples(specs: Iterable[OmicsMatrixSpec]) -> List[KGTriple]:
    """全量构建组学三元组列表（大数据集可能内存溢出，大规模构建请用 iter_omics_triples_chunked）。"""
    triples: List[KGTriple] = []
    for batch in iter_omics_triples_chunked(specs, chunk_size=0):
        triples.extend(batch)
    return triples


def iter_omics_triples_chunked(
    specs: Iterable[OmicsMatrixSpec],
    chunk_size: int = 10000,
) -> Iterator[List[KGTriple]]:
    """按 chunk_size 分批 yield 组学三元组，避免全量物化在内存中。

    每个 chunk 是一个独立的 List[KGTriple]，调用方可逐批写入 Neo4j。
    chunk_size=0 时退化为按 spec 逐样本 yield 全量。
    """
    for spec in specs:
        if not spec.matrix_path:
            continue
        expr = _load_expression_frame(spec)
        sample_meta = _load_sample_metadata(spec)
        relation = _relation_for_modality(spec.modality)
        prefix = _entity_prefix_for_modality(spec.modality)

        batch: List[KGTriple] = []
        limit = max(0, int(chunk_size))

        def _flush():
            nonlocal batch
            if batch:
                yield batch
                batch = []

        for sample_id in expr.columns:
            sample_node = _sample_node(sample_id)
            series = expr[sample_id]
            if spec.top_k_per_sample and spec.top_k_per_sample > 0:
                series = series.sort_values(ascending=False).head(spec.top_k_per_sample)
            for entity_id, value in series.items():
                try:
                    numeric = float(value)
                except (ValueError, TypeError):
                    continue  # 跳过非数值行（如甲基化数据中的 'ZZZ3' 标记）
                if numeric < float(spec.min_value or 0.0):
                    continue
                entity_node = _entity_node(prefix, str(entity_id))
                batch.append(KGTriple(
                    head=sample_node,
                    relation=relation,
                    tail=entity_node,
                    source=spec.source,
                    version=spec.version,
                    evidence=build_evidence(
                        raw="{} {} {}".format(sample_id, entity_id, numeric),
                        structured={
                            "sample": sample_id,
                            "entity": str(entity_id),
                            "value": numeric,
                            "modality": spec.modality,
                            "dataset": spec.name,
                        },
                    ),
                    weight=min(1.0, max(0.0, numeric)),
                    confidence=float(spec.confidence or 0.7),
                ))
                if limit > 0 and len(batch) >= limit:
                    yield from _flush()

            meta = sample_meta.get(str(sample_id), {})
            cell_line = meta.get("cell_line", "")
            condition = meta.get("condition", "")
            if cell_line:
                batch.append(KGTriple(
                    head=sample_node,
                    relation="belongs_to",
                    tail="cell::{}".format(normalize_entity(cell_line)),
                    source=spec.source,
                    version=spec.version,
                    evidence=build_evidence(
                        raw="{} -> {}".format(sample_id, cell_line),
                        structured={"sample": sample_id, "cell_line": cell_line},
                    ),
                    weight=1.0,
                    confidence=0.9,
                ))
            if condition:
                batch.append(KGTriple(
                    head=sample_node,
                    relation="has_condition",
                    tail=_condition_node(condition),
                    source=spec.source,
                    version=spec.version,
                    evidence=build_evidence(
                        raw="{} -> {}".format(sample_id, condition),
                        structured={"sample": sample_id, "condition": condition},
                    ),
                    weight=1.0,
                    confidence=0.9,
                ))

        if spec.target_map_path and str(spec.modality).lower() in {"mirna", "mi_rna", "micro_rna"}:
            mirna_triples = _build_mirna_target_triples(spec)
            for mt in mirna_triples:
                batch.append(mt)
                if limit > 0 and len(batch) >= limit:
                    yield from _flush()

        yield from _flush()


def _build_mirna_target_triples(spec: OmicsMatrixSpec) -> List[KGTriple]:
    target_path = Path(spec.target_map_path or "")
    if not target_path.exists():
        return []
    df = pd.read_csv(target_path, sep=spec.target_map_delimiter)
    mirna_col = "mirna" if "mirna" in df.columns else df.columns[0]
    gene_col = "target" if "target" in df.columns else ("gene" if "gene" in df.columns else df.columns[1])
    triples: List[KGTriple] = []
    for _, row in df.iterrows():
        mirna = str(row.get(mirna_col, "") or "").strip()
        target = str(row.get(gene_col, "") or "").strip()
        if not mirna or not target:
            continue
        triples.append(
            KGTriple(
                head="mirna::{}".format(normalize_entity(mirna).upper()),
                relation="regulates",
                tail="gene::{}".format(normalize_entity(target).upper()),
                source=spec.source,
                version=spec.version,
                evidence=build_evidence(
                    raw="{} -> {}".format(mirna, target),
                    structured={"mirna": mirna, "target": target, "dataset": spec.name},
                ),
                weight=0.7,
                confidence=float(spec.confidence or 0.7),
            )
        )
    return triples

