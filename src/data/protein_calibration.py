from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


class ProteinCalibrator:
    """Loads optional protein abundance priors (e.g., CCLE RPPA / CPTAC)."""

    def __init__(self, file_path: str = "", enabled: bool = True):
        self.file_path = str(file_path or "").strip()
        self.enabled = bool(enabled)
        self._index: Dict[str, Dict] = {}
        if self.enabled and self.file_path:
            self._load()

    def _normalize(self, value: str) -> str:
        return str(value or "").strip().upper()

    def _load(self) -> None:
        path = Path(self.file_path)
        if not path.exists():
            return
        try:
            df = pd.read_csv(path)
        except Exception:
            return

        # Supported column aliases.
        protein_col = None
        for cand in ["protein", "protein_symbol", "gene", "symbol", "target"]:
            if cand in df.columns:
                protein_col = cand
                break
        if protein_col is None:
            return

        log2fc_col = None
        for cand in ["log2_fc", "log2fc", "fold_change_log2", "delta"]:
            if cand in df.columns:
                log2fc_col = cand
                break
        if log2fc_col is None:
            return

        source_col = "source" if "source" in df.columns else None

        for _, row in df.iterrows():
            protein = self._normalize(row.get(protein_col, ""))
            if not protein:
                continue
            try:
                log2_fc = float(row.get(log2fc_col, 0.0) or 0.0)
            except Exception:
                log2_fc = 0.0
            source = str(row.get(source_col, "")) if source_col else ""
            prev = self._index.get(protein)
            payload = {
                "protein": protein,
                "log2_fc": log2_fc,
                "source": source,
            }
            if prev is None or abs(log2_fc) > abs(float(prev.get("log2_fc", 0.0) or 0.0)):
                self._index[protein] = payload

    def get(self, protein: str) -> Dict:
        return self._index.get(self._normalize(protein), {})

    def calibrate_delta(self, protein: str, raw_delta: float) -> Dict:
        meta = self.get(protein)
        if not meta:
            return {
                "delta": float(raw_delta),
                "factor": 1.0,
                "prior_sign": 0,
                "source": "",
            }

        prior_log2_fc = float(meta.get("log2_fc", 0.0) or 0.0)
        factor = max(0.7, min(2.0, 1.0 + abs(prior_log2_fc) * 0.35))
        calibrated = float(raw_delta) * factor
        prior_sign = 1 if prior_log2_fc > 0 else (-1 if prior_log2_fc < 0 else 0)

        return {
            "delta": calibrated,
            "factor": round(factor, 4),
            "prior_sign": prior_sign,
            "source": str(meta.get("source", "")),
        }

