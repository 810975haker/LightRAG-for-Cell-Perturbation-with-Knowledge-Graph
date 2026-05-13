from __future__ import annotations

from typing import Dict


def build_smw_template_payload(result: Dict) -> str:
    """Generate MediaWiki template text so the front-end can write semantic facts."""
    genes = ",".join(result.get("top_genes", [])) or "N/A"
    deltas = ";".join(
        [
            "{}:{}".format(item.get("gene", ""), item.get("predicted_delta", ""))
            for item in result.get("gene_expression_changes", [])
        ]
    ) or "N/A"
    return (
        "{{CellPerturbationPrediction"
        f"|CellID={result.get('cell_id', '')}"
        f"|Perturbation={result.get('perturbation', '')}"
        f"|TaskType={result.get('task_type', '')}"
        f"|Confidence={result.get('confidence', '')}"
        f"|TopGenes={genes}"
        f"|GeneExpressionDeltas={deltas}"
        f"|PredictedEffect={result.get('predicted_effect', '').replace('|', '/') }"
        "}}"
    )

