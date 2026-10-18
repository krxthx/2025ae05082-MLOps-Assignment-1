import json
import os
from pathlib import Path
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
STATE = {"output": "", "done": True, "code": 0}
COMMANDS = {
    "python3 src/data_download.py": ["python3", "src/data_download.py"],
    "python3 src/eda.py": ["python3", "src/eda.py"],
    "python3 src/train.py": ["python3", "src/train.py"],
    "cat models/metrics.json": ["cat", "models/metrics.json"],
    "cat models/best_model.txt": ["cat", "models/best_model.txt"],
    "pytest -q": ["python3", "-m", "pytest", "-q", "--color=no"],
    "python3 -m flake8 src tests --max-line-length=120 --extend-ignore=E203,W503": ["python3", "-m", "flake8", "src", "tests", "--max-line-length=120", "--extend-ignore=E203,W503"],
    "docker build -t heart-api:latest .": ["docker", "build", "-t", "heart-api:latest", "."],
    "docker run -d --name heart-video -p 8001:8000 heart-api:latest": ["docker", "run", "-d", "--name", "heart-video", "-p", "8001:8000", "heart-api:latest"],
    "docker ps --filter name=heart-video": ["docker", "ps", "--filter", "name=heart-video"],
    "curl -sS localhost:8001/health": ["curl", "-fsS", "localhost:8001/health"],
    "cat data/sample_input.json": ["python3", "-m", "json.tool", "data/sample_input.json"],
    "curl -sS localhost:8001/predict -H 'Content-Type: application/json' -d @data/sample_input.json": ["curl", "-fsS", "localhost:8001/predict", "-H", "Content-Type: application/json", "-d", "@data/sample_input.json"],
    "kubectl apply -f k8s/": ["kubectl", "apply", "-f", "k8s/"],
    "kubectl rollout status deployment/heart-api": ["kubectl", "rollout", "status", "deployment/heart-api"],
    "kubectl get pods,svc": ["kubectl", "get", "pods,svc"],
    "curl -sS localhost:8002/predict -H 'Content-Type: application/json' -d @data/sample_input.json": ["curl", "-fsS", "localhost:8002/predict", "-H", "Content-Type: application/json", "-d", "@data/sample_input.json"],
    "docker compose logs --tail=3 heart-api": ["docker", "compose", "logs", "--tail=3", "heart-api"],
    "ls models/": ["ls", "models/"],
}
SOURCES = {
    "preprocessing": ("src/preprocessing.py", 8, None),
    "training": ("src/train.py", 40, 60),
    "ci": (".github/workflows/ci.yml", 0, None),
    "docker": ("Dockerfile", 0, None),
    "kubernetes": ("k8s/deployment.yaml", 0, None),
    "readme": ("README.md", 0, 32),
}

def execute(argv):
    STATE.update(output="", done=False, code=None)
    env = dict(os.environ, PYTHONUNBUFFERED="1", NO_COLOR="1", API_PORT="8010", PROM_PORT="9091", GRAFANA_PORT="3001")
    try:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while chunk := process.stdout.read1(4096):
            STATE["output"] += re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", chunk.decode(errors="replace"))
        STATE["code"] = process.wait()
    except Exception as exc:
        STATE["output"] += str(exc)
        STATE["code"] = 1
    STATE["done"] = True

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, body, content_type="text/plain", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        route = urlparse(self.path)
        if route.path == "/state":
            self.send(json.dumps(STATE), "application/json")
        elif route.path == "/source":
            name = parse_qs(route.query).get("name", [""])[0]
            if name not in SOURCES:
                self.send("Unknown source", status=404)
                return
            path, start, end = SOURCES[name]
            lines = (ROOT / path).read_text().splitlines()[start:end]
            self.send(path + "\n\n" + "\n".join(f"{i + start + 1:2}  {line}" for i, line in enumerate(lines)))
        else:
            self.send((Path(__file__).parent / "console.html").read_text(), "text/html")

    def do_POST(self):
        command = json.loads(self.rfile.read(int(self.headers["Content-Length"]))).get("command")
        if self.path != "/run" or command not in COMMANDS or not STATE["done"]:
            self.send("Command unavailable", status=400)
            return
        STATE["done"] = False
        threading.Thread(target=execute, args=(COMMANDS[command],), daemon=True).start()
        self.send("Started")

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8766), Handler).serve_forever()
