"""FastAPI model-serving API with request logging + Prometheus metrics."""
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("heart-api")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_pipeline.joblib"
MODEL_PATH_ALT = Path("models/heart_pipeline.joblib")

PREDICTIONS = Counter("heart_predictions_total", "Total predictions", ["label"])
LATENCY = Histogram("heart_predict_latency_seconds", "Predict latency (s)")

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")


class Patient(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=50, le=300)
    chol: float = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)


_model = None


def get_model():
    global _model
    if _model is None:
        path = MODEL_PATH if MODEL_PATH.exists() else MODEL_PATH_ALT
        if not path.exists():
            raise RuntimeError(f"Model file not found at {path}. Run `python src/train.py` first.")
        _model = joblib.load(path)
        log.info("loaded model from %s", path)
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(p: Patient):
    t0 = time.time()
    try:
        model = get_model()
        X = pd.DataFrame([p.model_dump()])
        proba = float(model.predict_proba(X)[0][1])
        pred = int(proba >= 0.5)
        PREDICTIONS.labels(str(pred)).inc()
        LATENCY.observe(time.time() - t0)
        log.info("predict input=%s -> pred=%d proba=%.4f", p.model_dump(), pred, proba)
        return {"prediction": pred, "probability": round(proba, 4),
                "label": "disease" if pred else "no_disease"}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
