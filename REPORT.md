# MLOps Assignment 1 — Report: Heart-Disease Risk API (End-to-End)

**Course:** AIMLCZG523 Machine Learning Operations · **Marks:** 50
**Problem:** binary classification of heart-disease risk (UCI Cleveland, 303×14)
**Stack:** sklearn pipelines · MLflow · FastAPI · Pytest · GitHub Actions ·
Docker · Kubernetes (Minikube) · Prometheus metrics

---

## 1. Setup / install (reproducible from clean checkout)

```bash
pip install -r requirements.txt        # pinned; tested on Python 3.11/3.13
python src/data_download.py           # downloads UCI mirror -> data/heart.csv
python src/eda.py                     # -> screenshots/eda_*.png
python src/train.py                   # CV + MLflow -> models/heart_pipeline.joblib
pytest -q && python -m flake8 src tests --max-line-length=120 --extend-ignore=E203,W503
uvicorn src.app:app --port 8000
```

Serving-only image uses `requirements-serve.txt` (no mlflow/matplotlib/seaborn).
Verified clean-room path: download → train → test → `docker build` → `docker run`
→ `kubectl apply` all succeed without manual steps (CI does exactly this).

## 2. Data acquisition & EDA (Task 1)

- **Source:** UCI ML Repository, Heart Disease (Cleveland processed). `src/data_download.py`
  tries two mirrors (GitHub clean CSV, then UCI `processed.cleveland.data` with
  `?`→NaN handling and target binarisation 0–4 → 0/1), caches to `data/heart.csv`,
  aborts loudly if both fail. Columns: `age sex cp trestbps chol fbs restecg
  thalach exang oldpeak slope ca thal | target`.
- **Cleaning:** no missing values in the 303-row snapshot (verified
  `df.isna().sum()`); pipeline still imputes (median / most-frequent) so serving
  is robust to partial JSON. No duplicates; `ca`/`thal` kept as categorical codes.
- **EDA findings** (`python src/eda.py`, plots in `screenshots/`):
  - Class balance 54.5% disease / 45.5% healthy — mild imbalance, no resampling needed;
    stratified CV used throughout.
  - `eda_histograms.png`: `chol` right-skewed (max 564), `oldpeak` zero-inflated —
    motivates scaling (StandardScaler) rather than raw distances.
  - `eda_correlation.png`: `thalach` negatively correlated with disease/target and
    age; `exang`/`oldpeak`/`cp` strongest positive signals — consistent with
    clinical intuition (exercise-induced angina, ST depression).
  - `eda_cp_vs_target.png`: typical angina (cp=0) has the *lowest* disease rate in
    this cohort while non-anginal/asymptomatic types are higher — a known
    quirk of the Cleveland sample, captured automatically by one-hot encoding.
  - `eda_class_balance.png`: visual confirmation of the 165/138 split.

## 3. Feature engineering & model development (Task 2)

- **Pipeline** (`src/preprocessing.py`, single `ColumnTransformer` reused in
  training *and* serving — no train/serve skew):
  - numeric `age trestbps chol thalach oldpeak` → median-impute + StandardScaler;
  - categorical `sex cp fbs restecg exang slope ca thal` → most-frequent-impute +
    OneHotEncoder(`handle_unknown="ignore"`).
- **Models:** (a) Logistic Regression (`C=1.0`, `max_iter=2000`) — linear baseline,
  calibrated probabilities; (b) Random Forest (200 trees, `random_state=42`) —
  non-linear comparator. Deliberately no gradient boosting: with n=303, tuned
  forests/logreg generalise better and stay auditable for a clinical-adjacent task.
- **Evaluation:** stratified 5-fold CV, metrics accuracy / precision / recall /
  ROC-AUC (`src/train.py`):

| model | acc | prec | rec | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression ✅ shipped | **0.865** | 0.862 | **0.909** | **0.860** |
| Random Forest | 0.822 | 0.830 | 0.861 | 0.818 |

  LR wins on all four metrics; high recall (0.909) suits screening (false negatives
  cost more than false positives). Winner refit on full data and saved as one
  artifact `models/heart_pipeline.joblib` (preprocess + classifier), so the API
  cannot drift from training. Full table in `models/metrics.json`.

## 4. Experiment tracking (Task 3)

`train.py` logs to MLflow (SQLite backend `mlflow.db`, experiment `heart-disease`;
SQLite because MLflow ≥2.19 retired the flat-file store — `MLFLOW_ALLOW_FILE_STORE`
is the only opt-out). Each run records all hyperparameters, the four CV metrics +
`fit_seconds`, and the fitted pipeline artifact (artifact upload gracefully skipped
with a warning on sklearn/skops trust-policy mismatches; metrics/params always land).
`mlflow ui --backend-store-uri sqlite:///mlflow.db` reproduces the comparison table.

## 5. Packaging & reproducibility (Task 4)

- `models/heart_pipeline.joblib` (joblib, 4.7 KB) + `models/metrics.json` +
  `models/best_model.txt`; `data/heart.csv` + `data/sample_input.json` + download
  script pin the data lineage.
- `requirements.txt` (full, pinned) and `requirements-serve.txt` (slim image);
  verified serving env `sklearn==1.9.1 / pandas==2.3.3 / numpy==2.1.3` matches
  training env exactly — an earlier 1.9.1-vs-1.5.2 pickle mismatch
  (`LogisticRegression.multi_class`) was caught by the Docker smoke test and fixed
  by aligning pins and retraining (see §9 lessons).
- `src/inference.py` smoke client; `notebooks/eda_training.ipynb` mirrors the
  scripts interactively.

## 6. CI/CD & automated testing (Task 5)

- `tests/test_all.py` (4 tests, all passing): dataset contract (shape/columns/
  binary target), preprocessor output dimensionality, model quality gate
  (train accuracy ≥ 0.75), and live API test (`/health`, `/predict` schema +
  probability bounds, `/metrics` reachable) via FastAPI TestClient.
- `.github/workflows/ci.yml`: setup-python 3.11 → pip install → flake8 →
  download → train → pytest → `docker build` → upload `metrics.json`/screenshots.
  Any lint/test/train failure fails the workflow with logs; the Docker step proves
  the serving image builds on every push/PR.

## 7. Containerisation (Task 6)

`Dockerfile` (python:3.11-slim, slim requirements, `HEALTHCHECK` on `/health`,
2 replicas' worth of readiness/liveness in k8s): **builds clean** and **runs
isolated** — verified:

```
{"status":"ok"}                                                      # /health
{"prediction":1,"probability":0.8471,"label":"disease"}              # /predict (sample_input.json)
heart_predictions_total{label="1"} 1.0                               # /metrics
```

Request/response logs captured in `screenshots/docker_run.log`.

## 8. Production deployment (Task 7)

`k8s/deployment.yaml` (2 replicas, probes, CPU/mem requests+limits,
`imagePullPolicy: IfNotPresent`) + `k8s/service.yaml` (LoadBalancer → :80).
Deployed on **Minikube** (Docker driver):

```
minikube image load heart-api:latest
kubectl apply -f k8s/
kubectl rollout status deployment/heart-api   # successfully rolled out
NAME  READY  heart-api-xxx  1/1 Running (x2)
kubectl port-forward svc/heart-api 8002:80
curl /health -> {"status":"ok"}
curl /predict -> {"prediction":0,"probability":0.119,"label":"no_disease"}
```

Pod/service listing saved at `screenshots/k8s_pods.txt`. Same manifests apply to
GKE/EKS/AKS (swap image to a pushed tag, e.g. `ghcr.io/<user>/heart-api:<sha>`).

## 9. Monitoring & logging (Task 8)

- Every `/predict` emits a structured log line (timestamp, full input, pred,
  probability) — see `docker_run.log`; in Kubernetes via `kubectl logs deploy/heart-api`.
- Prometheus exposition at `/metrics`: `heart_predictions_total{label}`,
  `heart_predict_latency_seconds_*` histograms + standard process metrics.
- Full monitoring stack (heart-api + Prometheus + Grafana) wired with
  `docker-compose.yml`. One command brings up the entire observability layer:

  ```bash
  docker compose up --build
  # Prometheus → http://localhost:9090
  # Grafana    → http://localhost:3000  (admin / admin)
  ```

- Grafana is **auto-provisioned** — no manual setup needed:
  - Datasource: `monitoring/grafana/provisioning/datasources/prometheus.yml`
    (Prometheus at `http://prometheus:9090`).
  - Dashboard: `monitoring/grafana/dashboards/heart-api.json` — loads
    automatically and shows:
    - **Stat panels:** total predictions, disease count, no-disease count, p95 latency.
    - **Time-series:** prediction rate by label (req/s), latency p50/p95/p99.
    - **Donut chart:** prediction label distribution (disease vs no-disease ratio).
    - **Gauge:** live disease-rate % — useful for detecting label drift.
    - **CPU panel:** process CPU usage from default Prometheus process metrics.
- `monitoring/prometheus.yml` scrapes `heart-api:8000` at 15 s intervals;
  same config works for the Kubernetes deployment (swap target to the pod IP or
  service DNS).
- **Lesson learned:** sklearn/pandas/numpy must be pinned identically for train
  and serve; container smoke test caught the incompatibility before k8s.

## 10. Architecture

```
            ┌─────────────┐   CI (lint→train→test→docker build)
  UCI CSV → │ data_download│─────────────┐
            └──────┬──────┘             ▼
                   ▼              mlflow.db (params/metrics)
            ┌─────────────┐   ┌──────────────────┐
            │ eda / train │──▶│ heart_pipeline   │
            │ (sklearn    │   │ .joblib          │
            │  pipeline)  │   └────────┬─────────┘
            └─────────────┘            │ baked into image
                                       ▼
  JSON ──▶ FastAPI (/predict./health./metrics) ──▶ logs + Prometheus ──▶ Grafana
              │ Docker (heart-api:latest)
              ▼
        Kubernetes Deployment(2) + LoadBalancer Service (Minikube; portable to cloud)
```

## 11. Deliverables & links

Code/Dockerfile/requirements, cleaned data + download script, scripts + notebook,
`tests/`, Actions workflow, k8s manifests, screenshots, this report — all in the
submission folder, ready to `git init && git add . && git commit` and push as the
assignment repo. Deployed API: Minikube-local (`kubectl port-forward svc/heart-api
8002:80`); public URL on request after pushing image to GHCR. Demo video: record
per `VIDEO_SCRIPT.md` (data → MLflow → pytest → docker → minikube → curl).
