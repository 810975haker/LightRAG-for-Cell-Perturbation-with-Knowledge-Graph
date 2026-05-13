from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from scipy.io import mmread


MatrixFormat = Literal["csv", "tsv", "mtx"]


@dataclass
class CellPerturbRecord:
    cell_id: str
    perturbation: str


@dataclass
class CellPerturbDataset:
    dataset_name: str
    expression: pd.DataFrame
    perturbations: List[CellPerturbRecord]


class MatrixLoader:
    """Load bio datasets from matrix-like files into a normalized in-memory structure."""

    def load(
        self,
        matrix_path: str,
        matrix_format: MatrixFormat,
        dataset_name: str,
        genes_path: Optional[str] = None,
        cells_path: Optional[str] = None,
        perturbation_path: Optional[str] = None,
        delimiter: str = ",",
    ) -> CellPerturbDataset:
        expression = self._load_expression(
            matrix_path=matrix_path,
            matrix_format=matrix_format,
            genes_path=genes_path,
            cells_path=cells_path,
            delimiter=delimiter,
        )
        perturbations = self._load_perturbations(perturbation_path, expression.columns)
        return CellPerturbDataset(
            dataset_name=dataset_name,
            expression=expression,
            perturbations=perturbations,
        )

    def _load_expression(
        self,
        matrix_path: str,
        matrix_format: MatrixFormat,
        genes_path: Optional[str],
        cells_path: Optional[str],
        delimiter: str,
    ) -> pd.DataFrame:
        matrix_file = Path(matrix_path)
        if not matrix_file.exists():
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

        if matrix_format == "csv":
            df = pd.read_csv(matrix_file, sep=delimiter, index_col=0)
            return df

        if matrix_format == "tsv":
            df = pd.read_csv(matrix_file, sep="\t", index_col=0)
            return df

        if matrix_format == "mtx":
            sparse_matrix = mmread(matrix_file).tocsr()
            gene_names = self._load_labels(genes_path, sparse_matrix.shape[0], "gene")
            cell_ids = self._load_labels(cells_path, sparse_matrix.shape[1], "cell")
            dense = sparse_matrix.toarray()
            return pd.DataFrame(dense, index=gene_names, columns=cell_ids)

        raise ValueError(f"Unsupported matrix format: {matrix_format}")

    def _load_labels(self, path: Optional[str], expected_len: int, prefix: str) -> List[str]:
        if not path:
            return [f"{prefix}_{idx}" for idx in range(expected_len)]

        labels = pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()
        if len(labels) != expected_len:
            raise ValueError(
                f"Label count mismatch for {prefix}: expected {expected_len}, got {len(labels)}"
            )
        return labels

    def _load_perturbations(
        self,
        perturbation_path: Optional[str],
        cell_ids: List[str],
    ) -> List[CellPerturbRecord]:
        if not perturbation_path:
            return [CellPerturbRecord(cell_id=cell_id, perturbation="unknown") for cell_id in cell_ids]

        mapping_df = pd.read_csv(perturbation_path)
        if "cell_id" not in mapping_df.columns or "perturbation" not in mapping_df.columns:
            raise ValueError("perturbation file must include columns: cell_id, perturbation")

        mapping: Dict[str, str] = dict(zip(mapping_df["cell_id"].astype(str), mapping_df["perturbation"].astype(str)))
        return [
            CellPerturbRecord(cell_id=cell_id, perturbation=mapping.get(cell_id, "unknown"))
            for cell_id in cell_ids
        ]

