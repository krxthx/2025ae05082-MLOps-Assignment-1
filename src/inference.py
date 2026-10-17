"""Smoke-test a running API: python src/inference.py [--url http://localhost:8001]."""
import argparse
import json
import sys
import urllib.request

PAYLOAD = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
           "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
           "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}


def main(url: str) -> None:
    for path in ("health", "predict"):
        req = urllib.request.Request(
            f"{url}/{path}",
            data=json.dumps(PAYLOAD).encode() if path == "predict" else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"[{r.status}] /{path}: {r.read().decode()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    a = ap.parse_args()
    try:
        main(a.url)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"inference failed: {e}")
