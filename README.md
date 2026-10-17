# Heart Disease MLOps — End-to-End (Assignment 1)

Predict heart-disease risk from 13 clinical features, served as a monitored
FastAPI API, containerised with Docker, deployed on Kubernetes (Minikube),
with MLflow tracking and GitHub Actions CI.

## Quickstart (clean setup)

```bash
pip install -r requirements.txt
python src/data_download.py        # -> data/heart.csv (303 x 14, UCI)
python src/eda.py                  # stats + screenshots/eda_*.png
python src/train.py                # 5-fold CV, MLflow -> models/heart_pipeline.joblib
pytest -q
python -m flake8 src tests --max-line-length=120 --extend-ignore=E203,W503
uvicorn src.app:app --port 8000    # /health /predict /metrics
python src/inference.py --url http://localhost:8000
```

## Docker (isolated serving test)

```bash
docker build -t heart-api:latest .
docker run -d --name heart-test -p 8001:8000 heart-api:latest
curl localhost:8001/health
curl -X POST localhost:8001/predict -H 'Content-Type: application/json' \
  -d @data/sample_input.json
# -> {"prediction":1,"probability":0.8471,"label":"disease"}
```

`requirements-serve.txt` is the slim serving image (no mlflow/matplotlib).

## Kubernetes (Minikube)

```bash
minikube start --driver=docker
minikube image load heart-api:latest
kubectl apply -f k8s/               # deployment (2 replicas) + LoadBalancer svc
kubectl rollout status deployment/heart-api
kubectl port-forward svc/heart-api 8002:80
curl localhost:8002/health
```

Verified: 2/2 pods `Running`, live `/predict` returns 200 (see `screenshots/k8s_pods.txt`).

## Monitoring & logging (full stack)

```bash
docker compose up --build
# API        → http://localhost:8000
# Prometheus → http://localhost:9090
# Grafana    → http://localhost:3000  (admin / admin)
```

Grafana auto-provisions the **Heart Disease API** dashboard on first boot
(no manual datasource or import needed). Panels: prediction rate by label,
latency p50/p95/p99, disease-rate gauge (for label-drift detection), CPU usage.

- Structured request logs on every `/predict` (input, pred, probability).
- Prometheus metrics at `/metrics`: `heart_predictions_total{label}`,
  `heart_predict_latency_seconds_*` (+ default process metrics).
- Scrape config: `monitoring/prometheus.yml` (15 s interval).

## Results (5-fold stratified CV, 303 rows)

| model | accuracy | precision | recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (**selected**) | 0.865 | 0.862 | 0.909 | **0.860** |
| Random Forest (200 trees) | 0.822 | 0.830 | 0.861 | 0.818 |

Tracking: `mlflow.db` (SQLite backend, experiment `heart-disease`; MLflow ≥2.19
retired the flat-file store, so SQLite is the supported default).
Artifacts: `models/heart_pipeline.joblib` (full preprocess+classifier pipeline),
`models/metrics.json`, `models/best_model.txt`.

## Layout

```
src/ data_download.py, eda.py, preprocessing.py, train.py, app.py, inference.py
tests/test_all.py  .github/workflows/ci.yml  Dockerfile
k8s/deployment.yaml, k8s/service.yaml  monitoring/prometheus.yml
data/heart.csv, data/sample_input.json  models/  screenshots/  mlflow.db
notebooks/eda_training.ipynb  REPORT.md  VIDEO_SCRIPT.md
```

## Video

Record ≤5 min following `VIDEO_SCRIPT.md` (data → train → MLflow → pytest →
docker run → minikube deploy → curl). Suggested: `peek`/`obs` + voiceover,
save as `screenshots/demo.mp4` (not committed; upload link in report).
