# Narrated pipeline walkthrough

Target: about 5½–6 minutes at a relaxed pace, including pauses. The assignment asks for a short video; it does not specify a five-minute maximum. Read the quoted paragraphs aloud. Everything else is a recording cue.

Record your own narration after rehearsing the demo. Use the actual output on screen if a fresh run differs from the values below. Capture real terminal/browser activity, trim long waits, and keep enough of each command and its output visible to establish what ran. Record the scenes separately if that makes delivery easier.

## 1. Problem and architecture · 0:00–0:30

**Screen:** Open `screenshots/architecture.png` large enough to read. Keep the repository name visible in the editor or browser.

> This assignment builds an end-to-end MLOps pipeline for predicting the presence of heart disease from thirteen patient features. I use the UCI Heart Disease dataset, compare two classifiers, track experiments with MLflow, and serve the selected model through FastAPI. I will show the path from data and training to automated checks, Docker, Kubernetes, and monitoring.

Point along the architecture as you describe the flow. Avoid spending this scene scrolling through folders.

## 2. Data and EDA · 0:30–1:00

**Screen:** Show the downloader's output, then `screenshots/eda_class_balance.png` and `screenshots/eda_correlation.png`. Let each plot remain visible while discussing it.

> The downloaded dataset contains three hundred and three rows, thirteen input features, and one binary target. There are one hundred and sixty-five positive cases and one hundred and thirty-eight negative cases, so the imbalance is mild. I checked missing values and duplicates and examined feature distributions and correlations. Evaluation uses shuffled, stratified folds to preserve class proportions and avoid bias from the dataset's ordering.

## 3. Preprocessing and model selection · 1:00–1:45

**Screen:** Show `src/preprocessing.py`, the tuning grids in `src/train.py`, then `models/metrics.json` and `models/best_model.txt`. Use a readable split view rather than rapid scrolling.

> Preprocessing is part of the model pipeline. Numeric features receive median imputation and standard scaling. Categorical features receive most-frequent imputation and one-hot encoding, with unseen categories handled safely. Keeping preprocessing inside the pipeline fits it within each training fold and reuses it at inference.
>
> I tune Logistic Regression and Random Forest with five-fold stratified cross-validation. Random Forest is selected by the highest mean ROC-AUC: about zero point nine one eight, compared with zero point nine one seven for Logistic Regression. That margin is tiny, and Logistic Regression has higher recall. Tuning and evaluation reuse the same folds, so these estimates may be optimistic.

## 4. MLflow evidence · 1:45–2:15

**Screen:** In the live MLflow UI, open `heart-disease`, select both model runs, and compare them. Then open one run's artifacts and display its ROC curve. If using an existing run, say so rather than implying it was just trained.

> These are the tracked runs for the two candidate models. Each run records model parameters, tuned settings, accuracy, precision, recall, and ROC-AUC. The artifacts include grid-search results, an out-of-fold ROC curve, a confusion matrix, and the fitted model. This lets me inspect how a result was produced and compare experiments using the same evaluation setup.

## 5. Packaging, tests, and CI · 2:15–2:55

**Screen:** Briefly show `models/heart_pipeline.joblib` and the requirements files. Run the test and lint commands below. Then show the successful GitHub Actions run with its steps expanded.

> The packaged joblib file contains both preprocessing and the selected classifier. Training and serving dependencies are pinned, with a smaller requirements file for the serving image. The seven tests cover the dataset contract, preprocessing, missing-value handling, model inference, and API validation. Here are the local test and lint results.
>
> GitHub Actions repeats linting, data acquisition, training, tests, and a Docker build. A failing step fails the workflow, and the artifact step preserves available metrics and screenshots.

Pause while the results appear. Do not claim the tests passed until the output confirms it. The workflow builds the image; Kubernetes deployment is performed separately.

## 6. Docker and a real prediction · 2:55–3:40

**Screen:** Show a successful Docker build and the running container. Run the health check and prediction request. Keep the submitted JSON and response readable together.

> The Docker image contains the saved pipeline and starts the FastAPI server. This health request confirms the API is reachable. I now submit the sample patient's thirteen features to the prediction endpoint. The saved model returns class one, labelled disease, with a positive-class probability of zero point six five two.
>
> The probability field always refers to the disease class. Input validation rejects invalid or incomplete requests before they reach the model. This is an educational model, rather than a tool for clinical decisions.

If your actual response differs, read that response. `/health` checks API reachability; the successful `/predict` request demonstrates model loading and inference.

## 7. Kubernetes deployment · 3:40–4:25

**Screen:** Show `k8s/deployment.yaml` and `k8s/service.yaml`, then real rollout status and `kubectl get pods,svc`. Through the port-forward, run the same prediction request on port 8002.

> The same image is deployed to local Minikube using these manifests. The deployment requests two replicas and defines resource limits, readiness checks, and liveness checks. Here, both pods are ready, and the rollout has completed.
>
> The service is configured as a LoadBalancer. On this local setup, I access it through port-forwarding; a pending external IP does not prevent this demonstration. This request goes through the Kubernetes service and returns a prediction from the deployed model.

Keep the separate port-forward terminal running while demonstrating the request. Do not describe this as a publicly hosted API.

## 8. Monitoring and request logs · 4:25–5:15

**Screen:** Generate traffic against the Compose API, then show the Prometheus target as UP, the provisioned Grafana dashboard, and a recent request log. Use the commands below. Keep the dashboard visible for most of this scene.

> The monitoring stack runs the API, Prometheus, and Grafana. Prometheus scrapes the API every fifteen seconds; this target is up. After sending prediction requests, the Grafana dashboard displays prediction counts, request rates, latency, class distribution, and process CPU usage.
>
> The request logs connect an input with its prediction and probability. The disease-rate gauge can flag changes in predicted class mix, but it cannot establish model accuracy or confirm drift without further evidence and ground-truth labels.

The Compose API and Kubernetes API are separate instances. These dashboard metrics come from Compose's `heart-api:8000` scrape target. Generate traffic on port 8000, not only on port 8002. Default host ports can be overridden; match the commands to the chosen ports.

## 9. Close · 5:15–5:35

**Screen:** Return to the repository README and report. Show the local API access instructions and deliverables.

> The repository includes the download and training scripts, reusable pipeline, tests, CI workflow, deployment manifests, monitoring configuration, and report. This demonstration connects a reproducible experiment to a tested, containerised, locally deployed API with operational visibility. The main limitations are the small dataset, shared tuning and evaluation folds, and the absence of live ground-truth labels.

## Preparation and commands

Run from the repository root in the prepared Python environment. Start Docker Desktop first. Do a rehearsal before recording; use terminal text around 20–24 points at 1080p. Close unrelated windows and notifications.

### Data, training, and MLflow

```bash
python src/data_download.py
python src/eda.py
python src/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Training generates new MLflow runs and rewrites model artifacts. Finish training before building either serving image. Open `http://localhost:5000`. Keep the process in a separate terminal. Record the training command and completed output; trim the wait if necessary.

### Tests and lint

```bash
pytest -q
python -m flake8 src tests --max-line-length=120 --extend-ignore=E203,W503
```

A successful flake8 run is silent. Leave the returned prompt visible so the viewer can see completion.

### Isolated Docker serving

Use an unused container name; if `heart-video` already exists, inspect it and choose another name or reuse the correct container.

```bash
docker build -t heart-api:latest .
docker run -d --name heart-video -p 8001:8000 heart-api:latest
docker ps --filter name=heart-video
curl -fsS http://localhost:8001/health
cat data/sample_input.json
curl -fsS http://localhost:8001/predict \
  -H 'Content-Type: application/json' -d @data/sample_input.json
```

Wait for startup before the requests. The current saved evidence returns `{"prediction":1,"probability":0.652,"label":"disease"}`; verify it during rehearsal.

### Kubernetes

```bash
minikube status
minikube image load heart-api:latest
kubectl apply -f k8s/
kubectl rollout status deployment/heart-api
kubectl get pods,svc
kubectl port-forward svc/heart-api 8002:80
```

If Minikube is stopped, start it with `minikube start --driver=docker` before recording. If the existing cluster has an older `heart-api:latest`, refresh the cached image and recreate the pods during preparation, then verify the deployed prediction matches the current Docker model. Applying an unchanged manifest alone does not refresh running pods.

In a second terminal:

```bash
curl -fsS http://localhost:8002/health
curl -fsS http://localhost:8002/predict \
  -H 'Content-Type: application/json' -d @data/sample_input.json
kubectl logs -l app=heart-api --tail=3 --prefix=true
```

### Monitoring

```bash
docker compose up -d --build
```

Wait for `http://localhost:8000/health` to respond, then generate mixed traffic against that API. This replays dataset features; it does not measure production accuracy.

```bash
python - <<'PY'
import json
import time
import urllib.request
from itertools import islice

import pandas as pd

rows = pd.read_csv('data/heart.csv').drop(columns='target')
rows = rows.sample(frac=1, random_state=42)
for row in islice(rows.to_dict(orient='records'), 90):
    request = urllib.request.Request(
        'http://localhost:8000/predict',
        data=json.dumps(row).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        response.read()
    time.sleep(1)
PY
```

Traffic runs for about ninety seconds. Allow at least two Prometheus scrapes before checking rate panels. Open `http://localhost:9090/targets` and `http://localhost:3000` (local demo login: admin/admin), then the **Heart Disease API** dashboard. Select a recent time range and enable refresh. If the charts are sparse, send more traffic and inspect the scrape status before recording.

```bash
curl -fsS http://localhost:8000/metrics | rg 'heart_predictions_total|heart_predict_latency_seconds_count'
docker compose logs --tail=5 heart-api
```

## Recording and hand-in

- Capture a ten-second microphone test and play it back before recording the full narration. Keep the microphone level below clipping.
- Keep each command, result, chart, or browser view on screen while its narration discusses it. Trim builds and page-load waits; retain real success/failure output.
- Use existing screenshots only as clearly identified evidence where needed. Prefer fresh screen capture for the API, rollout, and monitoring demonstrations.
- Listen to the exported MP4 from beginning to end and check text readability, timing, and audible speech. H.264 video with AAC audio at 1080p is a practical export choice.
- Save the new export as `demo/pipeline_walkthrough_narrated.mp4` until reviewed.
- After the new video exists, update the video references in `README.md`, `REPORT.md`, and the regenerated `REPORT.pdf`. If using a hosted link, verify it opens for the evaluator without requesting access.
