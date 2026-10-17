"""EDA script: prints summary stats + writes professional plots to screenshots/.

Usage: python src/eda.py  (requires data/heart.csv)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import pandas as pd  # noqa: E402

sns.set_theme(style="whitegrid")

DATA = Path("data/heart.csv")
OUT = Path("screenshots")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(DATA)
    print(f"shape={df.shape}")
    print(df.describe().round(2).to_string())
    print("\nmissing:\n", df.isna().sum().to_string())
    print("\nclass balance:\n", df["target"].value_counts(normalize=True).round(3).to_string())

    # 1. histograms for numeric cols
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.flat, num_cols):
        sns.histplot(df[c], kde=True, ax=ax)
        ax.set_title(f"Distribution: {c}")
    axes.flat[-1].axis("off")
    fig.suptitle("Heart Disease — Numeric Feature Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "eda_histograms.png", dpi=150)
    plt.close(fig)

    # 2. class balance
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="target", data=df, ax=ax)
    ax.set_title("Class balance (0 = no disease, 1 = disease)")
    ax.set_xlabel("target")
    fig.tight_layout()
    fig.savefig(OUT / "eda_class_balance.png", dpi=150)
    plt.close(fig)

    # 3. correlation heatmap (numeric only)
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[num_cols + ["target"]].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap (numeric features + target)")
    fig.tight_layout()
    fig.savefig(OUT / "eda_correlation.png", dpi=150)
    plt.close(fig)

    # 4. chest-pain type vs target
    fig, ax = plt.subplots(figsize=(7, 4))
    pd.crosstab(df["cp"], df["target"], normalize="index").plot(kind="bar", ax=ax)
    ax.set_title("Disease rate by chest-pain type (cp)")
    ax.set_xlabel("cp (0-3)")
    fig.tight_layout()
    fig.savefig(OUT / "eda_cp_vs_target.png", dpi=150)
    plt.close(fig)

    print(f"[eda] plots written to {OUT}/eda_*.png")


if __name__ == "__main__":
    main()
