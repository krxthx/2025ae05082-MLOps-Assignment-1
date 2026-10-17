"""Unit tests: data contract + preprocessing + model quality + API."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import FEATURE_ORDER, TARGET, build_preprocessor  # noqa: E402


def test_dataset_contract():
    df = pd.read_csv(ROOT / "data" / "heart.csv")
    assert df.shape[0] >= 200, "dataset unexpectedly small"
    assert set(FEATURE_ORDER + [TARGET]).issubset(df.columns)
    assert df[TARGET].isin([0, 1]).all()
    assert df.isna().sum().sum() < len(df) * len(df.columns) * 0.2


def test_preprocessor_transforms():
    df = pd.read_csv(ROOT / "data" / "heart.csv")
    X = df.drop(columns=[TARGET])
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] > len(FEATURE_ORDER)


def test_model_quality():
    import joblib
    from sklearn.metrics import accuracy_score
    model = joblib.load(ROOT / "models" / "heart_pipeline.joblib")
    df = pd.read_csv(ROOT / "data" / "heart.csv")
    acc = accuracy_score(df[TARGET], model.predict(df.drop(columns=[TARGET])))
    assert acc >= 0.75, f"train accuracy too low: {acc:.3f}"


def test_api_predict():
    from fastapi.testclient import TestClient
    from app import app
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    payload = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
               "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
               "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}
    r = c.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert c.get("/metrics").status_code == 200


def test_preprocessor_has_no_nans():
    import numpy as np
    df = pd.read_csv(ROOT / "data" / "heart.csv")
    X = df.drop(columns=[TARGET])
    X.loc[X.index[:10], "chol"] = np.nan  # imputation must absorb missing values
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(np.asarray(Xt, dtype=float)).any()


def test_pipeline_predicts_with_missing_value():
    import joblib
    import numpy as np
    model = joblib.load(ROOT / "models" / "heart_pipeline.joblib")
    row = pd.read_csv(ROOT / "data" / "heart.csv").drop(columns=[TARGET]).iloc[[0]].copy()
    row["chol"] = np.nan
    proba = model.predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_api_rejects_invalid_input():
    from fastapi.testclient import TestClient
    from app import app
    c = TestClient(app)
    bad = {"age": 63, "sex": 7, "cp": 3, "trestbps": 145, "chol": 233,
           "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
           "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}
    assert c.post("/predict", json=bad).status_code == 422
    assert c.post("/predict", json={"age": 63}).status_code == 422
