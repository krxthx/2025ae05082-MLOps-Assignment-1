"""Download Heart Disease UCI dataset (reproducible, offline-safe).

Tries (in order):
 1. Local cached CSV (data/heart.csv)
 2. Remote mirrors (UCI-processed + widely used GitHub mirrors)
 3. sklearn-free synthetic fallback is NOT used -- fails loudly instead.

Usage:
    python src/data_download.py [--out data/heart.csv]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

URLS = [
    # Well-known clean mirror (303 rows, 14 cols, header present)
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv",
    # UCI archive processed file (no header, '?' for missing)
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
]

DATA_README = """# Dataset
Title: Heart Disease UCI (Cleveland, processed)
Source: UCI ML Repository — https://archive.ics.uci.edu/dataset/45/heart+disease
Rows: 303 | Features: 13 + binary target (0 = no disease, 1 = disease)
Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
         oldpeak, slope, ca, thal, target
Download: `python src/data_download.py` (see script for mirrors).
"""


def load_from_url(url: str) -> pd.DataFrame:
    if "processed.cleveland" in url:
        df = pd.read_csv(url, header=None, names=COLUMNS, na_values="?")
        # Cleveland file uses 0-4 severity; binarize to 0/1
        df["target"] = (df["target"] > 0).astype(int)
        return df
    df = pd.read_csv(url)
    # normalise common variants
    if "target" not in df.columns and "output" in df.columns:
        df = df.rename(columns={"output": "target"})
    return df


def main(out: str = "data/heart.csv") -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        df = pd.read_csv(out_path)
        print(f"[data] using cached {out_path} shape={df.shape}")
        return
    last_err = None
    for url in URLS:
        try:
            print(f"[data] trying {url} ...")
            df = load_from_url(url)
            assert set(COLUMNS).issubset(df.columns), f"unexpected cols {df.columns.tolist()}"
            df.to_csv(out_path, index=False)
            print(f"[data] saved {out_path} shape={df.shape}")
            (out_path.parent / "README.md").write_text(DATA_README)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[data] failed: {e}", file=sys.stderr)
    raise SystemExit(f"All dataset mirrors failed. Last error: {last_err}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/heart.csv")
    args = p.parse_args()
    main(args.out)
