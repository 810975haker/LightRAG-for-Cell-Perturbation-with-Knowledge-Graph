from __future__ import annotations

from typing import Iterable

import networkx as nx

from config import KG_EXPR_THRESHOLD, KG_TOP_GENES_PER_CELL
from src.data.matrix_loader import CellPerturbDataset


class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def build_from_dataset(self, dataset: CellPerturbDataset) -> nx.MultiDiGraph:
        """Build a bio knowledge graph from expression matrix and perturbation labels."""
        self.clear_graph()

        for record in dataset.perturbations:
            cell_node = f"cell::{record.cell_id}"
            pert_node = f"pathway::perturbation::{record.perturbation}"

            self.graph.add_node(cell_node, type="Cell", cell_id=record.cell_id)
            self.graph.add_node(pert_node, type="pathway", name=record.perturbation)
            self.graph.add_edge(pert_node, cell_node, relation="belongs_to", weight=1.0, confidence=0.9)
            # Keep a forward edge from cell for neighborhood retrieval while staying in canonical relation set.
            self.graph.add_edge(cell_node, pert_node, relation="associated_with", weight=0.9, confidence=0.8)

            self._add_expression_edges(cell_node, dataset.expression[record.cell_id])

        return self.graph

    def _add_expression_edges(self, cell_node: str, expression_series) -> None:
        top_genes = expression_series.sort_values(ascending=False).head(KG_TOP_GENES_PER_CELL)
        for gene_name, value in top_genes.items():
            if float(value) < KG_EXPR_THRESHOLD:
                continue
            gene_node = f"gene::{gene_name}"
            self.graph.add_node(gene_node, type="gene", gene_id=str(gene_name))
            self.graph.add_edge(
                gene_node,
                cell_node,
                relation="belongs_to",
                weight=float(value),
                confidence=0.9,
            )
            self.graph.add_edge(
                cell_node,
                gene_node,
                relation="associated_with",
                weight=float(value),
                confidence=0.85,
            )

    def build_graph(self, text: str) -> nx.MultiDiGraph:
        """Fallback text graph builder to keep compatibility with earlier text ingestion."""
        self.clear_graph()
        tokens = [token.strip() for token in text.split() if token.strip()]
        unique_tokens = list(dict.fromkeys(tokens[:100]))
        for token in unique_tokens:
            self.graph.add_node(token, type=self._infer_type(token))
        for head, tail in self._pairwise(unique_tokens):
            self.graph.add_edge(head, tail, relation="associated_with", weight=0.2, confidence=0.3)
        return self.graph

    def _infer_type(self, token: str) -> str:
        text = str(token or "")
        lower = text.lower()
        if lower.startswith("gene::"):
            return "gene"
        if lower.startswith("protein::") or text.startswith("9606.") or text.upper().startswith("ENSP"):
            return "protein"
        if lower.startswith("pathway::"):
            return "pathway"
        if lower.startswith("cell::") or "a549" in lower or "cell" in lower:
            return "Cell"
        return "gene"

    def _pairwise(self, values: Iterable[str]):
        values = list(values)
        for idx in range(len(values) - 1):
            yield values[idx], values[idx + 1]

    def clear_graph(self):
        self.graph.clear()

    def get_graph(self):
        return self.graph