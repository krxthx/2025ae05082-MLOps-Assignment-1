# Demo video script (≤5 min) — record, save link in REPORT

1. (30s) Problem + repo tour: `ls`, README, architecture (REPORT §10).
2. (45s) Data & EDA: `python src/data_download.py`, `python src/eda.py`; show
   `screenshots/eda_*.png`, class balance 54.5/45.5.
3. (60s) Training: `python src/train.py`; show metrics table
   (tuned: RF ROC-AUC 0.918 vs LR 0.917, near-tie; show the GridSearch lines); `mlflow ui --backend-store-uri sqlite:///mlflow.db`
   runs comparison.
4. (30s) Tests: `pytest -q` (7 passed) + flake8 clean.
5. (60s) Docker: `docker build -t heart-api .`, `docker run -p 8001:8000`,
   curl `/health` + `/predict` (0.652 disease) + `/metrics`.
6. (60s) K8s: `minikube image load`, `kubectl apply -f k8s/`,
   `kubectl get pods,svc`, port-forward + live curl (both labels).
7. (15s) Monitoring: `/metrics` output + `monitoring/prometheus.yml` in Grafana.

Tools: OBS/QuickTime + mic; 1080p terminal font ≥16pt; upload unlisted YouTube /
Drive; add link to REPORT §11.
