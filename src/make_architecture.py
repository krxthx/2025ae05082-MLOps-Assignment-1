"""Render the architecture diagram used in REPORT.md -> screenshots/architecture.png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

COLORS = {"data": "#dbeafe", "ml": "#dcfce7", "ci": "#fef9c3", "serve": "#fae8ff", "obs": "#ffedd5"}


def box(ax, x, y, w, h, text, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=COLORS[kind], ec="#334155", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(ax, p, q, label=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color="#334155", lw=1.2))
    if label:
        dx, dy = (0.7, 0) if p[0] == q[0] else (0, 0.12)
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label, ha="center", fontsize=7, color="#475569")


def main(out="screenshots/architecture.png"):
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    ax.text(6, 6.35, "Heart Disease MLOps - end-to-end architecture", ha="center", fontsize=12, weight="bold")

    # Row 1: data -> model
    box(ax, 0.2, 5.0, 2.0, 0.9, "UCI Heart Disease\n(download script)", "data")
    box(ax, 2.9, 5.0, 2.0, 0.9, "EDA + cleaning\nsrc/eda.py", "data")
    box(ax, 5.6, 5.0, 2.4, 0.9, "Train: sklearn Pipeline\nLR vs RF + GridSearchCV", "ml")
    box(ax, 9.0, 5.0, 2.6, 0.9, "heart_pipeline.joblib\n(preprocess + model)", "ml")
    arrow(ax, (2.2, 5.45), (2.9, 5.45))
    arrow(ax, (4.9, 5.45), (5.6, 5.45))
    arrow(ax, (8.0, 5.45), (9.0, 5.45))

    # Row 2: CI + tracking
    box(ax, 0.2, 3.3, 1.8, 0.9, "GitHub\npush / PR", "ci")
    box(ax, 2.6, 3.3, 3.2, 0.9, "GitHub Actions: flake8 -> train\n-> pytest -> docker build", "ci")
    box(ax, 6.4, 3.3, 2.0, 0.9, "MLflow tracking\nmlflow.db", "ml")
    arrow(ax, (2.0, 3.75), (2.6, 3.75))
    arrow(ax, (6.8, 5.0), (7.0, 4.2), "params, metrics, plots")

    # Row 3: serving
    box(ax, 0.2, 1.7, 2.0, 0.9, "Client\nPOST /predict (JSON)", "serve")
    box(ax, 3.0, 1.7, 2.6, 0.9, "Kubernetes (Minikube)\nDeployment x2 + LoadBalancer", "serve")
    box(ax, 6.4, 1.7, 2.4, 0.9, "Docker image heart-api\nFastAPI: /predict\n/health /metrics", "serve")
    arrow(ax, (2.2, 2.15), (3.0, 2.15))
    arrow(ax, (6.4, 2.15), (5.6, 2.15), "image")
    arrow(ax, (10.3, 5.0), (8.2, 2.6), "baked into image")

    # Row 4: observability
    box(ax, 3.0, 0.2, 2.6, 0.9, "Prometheus -> Grafana\nrates, latency, label-drift gauge", "obs")
    arrow(ax, (4.3, 1.1), (4.3, 1.7), "scrape /metrics")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
