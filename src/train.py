"""Train ≥2 classifiers with CV + MLflow tracking; save best pipeline.

Usage: python src/train.py [--data data/heart.csv --out models/heart_pipeline.joblib]
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402
from preprocessing import TARGET, build_pipeline  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "roc_auc": "roc_auc",  # from predict_proba (not hard labels)
}

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1
    ),
}

# Hyper-parameter grids searched with the same stratified 5-fold CV (scored on ROC-AUC).
GRIDS = {
    "logistic_regression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "random_forest": {"clf__n_estimators": [100, 200, 400], "clf__max_depth": [None, 4, 8]},
}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def tune(name, pipe, X, y):
    gs = GridSearchCV(pipe, GRIDS[name], scoring="roc_auc", cv=CV, n_jobs=-1)
    gs.fit(X, y)
    return gs.best_params_, float(gs.best_score_), gs.cv_results_


def log_plots(name, pipe, X, y):
    """Log out-of-fold ROC curve and confusion matrix as MLflow figures."""
    proba = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba")[:, 1]
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name=name)
    ax.set_title(f"{name}: out-of-fold ROC")
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y, (proba >= 0.5).astype(int), ax=ax, colorbar=False)
    ax.set_title(f"{name}: confusion matrix (CV)")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


def evaluate(pipe, X, y):
    cv = CV
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {k.replace("test_", ""): float(res[k].mean()) for k in res if k.startswith("test_")}


def main(data="data/heart.csv", out="models/heart_pipeline.joblib"):
    df = pd.read_csv(data)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    print(f"[train] data shape={df.shape} positive_rate={y.mean():.3f}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-disease")
    best_name, best_score, best_pipe = None, -1.0, None
    summary = {}

    for name, clf in CANDIDATES.items():
        pipe = build_pipeline(clf)
        t0 = time.time()
        with mlflow.start_run(run_name=name):
            best_params, tuned_auc, cvr = tune(name, pipe, X, y)
            pipe.set_params(**best_params)
            print(f"[train] {name} tuned: {best_params} cv_roc_auc={tuned_auc:.4f}")
            mlflow.log_params({f"{name}__{k}": v for k, v in clf.get_params().items()})
            mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})
            mlflow.log_metric("grid_best_roc_auc", tuned_auc)
            pd.DataFrame(cvr)[["params", "mean_test_score", "rank_test_score"]].to_csv("grid_results.csv", index=False)
            mlflow.log_artifact("grid_results.csv")
            Path("grid_results.csv").unlink()
            metrics = evaluate(pipe, X, y)
            log_plots(name, pipe, X, y)
            metrics["fit_seconds"] = time.time() - t0
            mlflow.log_metrics(metrics)
            pipe.fit(X, y)  # refit on full data for packaging
            try:
                mlflow.sklearn.log_model(
                    pipe,
                    artifact_path=name,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
            except Exception as e:  # noqa: BLE001 - skops trust issue on some versions
                print(f"[train] mlflow log_model skipped ({name}): {e}")
                mlflow.log_param("model_artifact_skipped", name)
            print(f"[train] {name}: " + json.dumps(metrics, indent=2))
            summary[name] = metrics
            if metrics["roc_auc"] > best_score:
                best_name, best_score, best_pipe = name, metrics["roc_auc"], pipe

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, out)
    Path("models/metrics.json").write_text(json.dumps(summary, indent=2))
    Path("models/best_model.txt").write_text(best_name)
    print(f"[train] BEST={best_name} roc_auc={best_score:.4f} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heart.csv")
    ap.add_argument("--out", default="models/heart_pipeline.joblib")
    a = ap.parse_args()
    main(a.data, a.out)
