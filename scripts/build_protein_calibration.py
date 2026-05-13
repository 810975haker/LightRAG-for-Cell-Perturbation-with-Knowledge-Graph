from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pick_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Build protein calibration table from CCLE RPPA/CPTAC-like files")
    parser.add_argument("--input", required=True, help="Raw CSV/TSV path")
    parser.add_argument("--output", default="data/processed/lung_cancer/protein_calibration.csv")
    parser.add_argument("--source", default="CCLE_RPPA")
    parser.add_argument("--sep", default=",", help="Delimiter for input file, e.g. , or \t")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError("Input not found: {}".format(in_path))

    df = pd.read_csv(in_path, sep=args.sep)

    protein_col = _pick_column(df.columns, ["protein", "protein_symbol", "target", "gene", "symbol"])
    value_col = _pick_column(df.columns, ["log2_fc", "log2fc", "fold_change_log2", "delta"])
    if protein_col is None or value_col is None:
        raise ValueError("Input must include protein/gene column and log2_fc-like column")

    out = pd.DataFrame(
        {
            "protein": df[protein_col].astype(str).str.strip().str.upper(),
            "log2_fc": pd.to_numeric(df[value_col], errors="coerce").fillna(0.0),
            "source": str(args.source),
        }
    )
    out = out[out["protein"] != ""]
    out = out.groupby("protein", as_index=False)["log2_fc"].median().merge(
        out[["protein", "source"]].drop_duplicates(subset=["protein"]), on="protein", how="left"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("Calibration table written:", out_path)
    print("Rows:", len(out))


if __name__ == "__main__":
    main()

