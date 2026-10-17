"""Train ≥2 classifiers with CV + MLflow tracking; save best pipeline.

Usage: python src/train.py [--data data/heart.csv --out models/heart_pipeline.joblib]
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402
from preprocessing import TARGET, build_pipeline  # noqa: E402

SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "roc_auc": make_scorer(roc_auc_score),
}

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1
    ),
}


def evaluate(pipe, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
            mlflow.log_params({f"{name}__{k}": v for k, v in clf.get_params().items()})
            metrics = evaluate(pipe, X, y)
            metrics["fit_seconds"] = time.time() - t0
            mlflow.log_metrics(metrics)
            pipe.fit(X, y)  # refit on full data for packaging
            try:
                mlflow.sklearn.log_model(pipe, artifact_path=name)
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
