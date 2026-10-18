import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, "/private/tmp/assignment-video-tools")
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[2]
WORK = Path("/private/tmp/assignment-recording")
BROWSER = "/Users/krithika/Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
CONSOLE = "http://127.0.0.1:8766"
MLFLOW = "http://127.0.0.1:5001"
GITHUB = "https://github.com/krxthx/2025ae05082-MLOps-Assignment-1"
CI_RUN_TITLE = "Docs: point README and report at the narrated walkthrough"

POINTER = """
window.addEventListener('DOMContentLoaded',()=>{
if(location.hostname==='github.com')document.documentElement.style.zoom='1.25';
if(location.port==='8001')document.documentElement.style.zoom='1.35';
if(location.port==='5001')document.documentElement.style.zoom='1.15';
if(location.port==='9091')document.documentElement.style.zoom='1.25';
const cursor=document.createElement('div');cursor.style.cssText='position:fixed;width:22px;height:22px;border:2px solid #25ba96;background:#25ba9633;border-radius:50%;pointer-events:none;z-index:2147483647;transform:translate(-50%,-50%)';document.body.appendChild(cursor);
document.addEventListener('mousemove',e=>{cursor.style.left=e.clientX+'px';cursor.style.top=e.clientY+'px'});
const badge=document.createElement('div');badge.textContent='Live browser demo · Synthetic voice-over';badge.style.cssText='position:fixed;bottom:12px;right:18px;background:#132333ed;color:#c4d7e9;padding:8px 13px;border:1px solid #456174;border-radius:5px;font:14px system-ui;pointer-events:none;z-index:2147483646';document.body.appendChild(badge);
});
"""

EVENTS = []
SCENE_START = 0

async def pause(page, seconds):
    await page.wait_for_timeout(seconds * 1000)
    text = (await page.locator("body").inner_text()).lower()
    if any(term in text for term in ("claude", "co-authored-by:", "noreply@anthropic.com")):
        raise RuntimeError("Unexpected attribution or assistant label in the recording view")

async def point(page, locator):
    await locator.scroll_into_view_if_needed()
    box = await locator.bounding_box()
    if box:
        await page.mouse.move(box["x"] + min(box["width"] / 2, 160), box["y"] + box["height"] / 2, steps=22)

async def click(page, locator):
    await point(page, locator)
    await pause(page, .35)
    await locator.click()

async def terminal(page, command, hold=2):
    if not page.url.rstrip("/") == CONSOLE:
        await page.goto(CONSOLE)
    field = page.locator("#command")
    await click(page, field)
    await field.press_sequentially(command, delay=25)
    await pause(page, .5)
    entered = time.monotonic() - SCENE_START
    await field.press("Enter")
    await page.wait_for_function("document.querySelector('#state').textContent.startsWith('Completed')", timeout=180000)
    state = await (await page.request.get(CONSOLE + "/state")).json()
    if state["code"] != 0:
        raise RuntimeError(f"{command} failed: {state['output']}")
    EVENTS.append({"command": command, "start": entered, "completed": time.monotonic() - SCENE_START})
    print(f"  executed: {command}", flush=True)
    await pause(page, hold)
    return state["output"]

async def source(page, name, hold=4):
    await page.goto(CONSOLE + "/file?name=" + name)
    await pause(page, hold)

async def scene1(page):
    await page.goto(GITHUB + "/commits/main/")
    await pause(page, 3)
    await click(page, page.get_by_role("link", name=CI_RUN_TITLE, exact=True).first)
    await pause(page, 3)
    await page.goto(GITHUB + "/blob/main/README.md")
    await page.get_by_role("heading", name="Heart Disease MLOps — End-to-End (Assignment 1)").scroll_into_view_if_needed()
    await pause(page, 5)
    await page.mouse.wheel(0, 490)
    await pause(page, 5)
    await page.mouse.wheel(0, 420)
    await pause(page, 4)

async def scene2(page):
    await page.goto(CONSOLE)
    await page.locator("#chapter").evaluate("e=>e.textContent='Data acquisition and exploratory analysis'")
    await terminal(page, "python3 src/data_download.py", 3)
    await terminal(page, "python3 src/eda.py", 2)
    await page.goto((ROOT / "screenshots/eda_class_balance.png").as_uri())
    await pause(page, 5)
    await page.goto((ROOT / "screenshots/eda_correlation.png").as_uri())
    await pause(page, 5)
    await page.mouse.move(1100, 450, steps=20)

async def scene3(page):
    await source(page, "preprocessing", 12)
    await page.mouse.wheel(0, 350)
    await pause(page, 5)
    await source(page, "training", 6)
    await terminal(page, "python3 src/train.py", 2)
    await terminal(page, "cat models/metrics.json", 7)
    await terminal(page, "cat models/best_model.txt", 3)

async def scene4(page):
    await page.goto(MLFLOW + "/#/experiments/1/runs?workflowType=machine_learning")
    await pause(page, 3)
    await click(page, page.get_by_role("link", name="random_forest", exact=True).first)
    await pause(page, 5)
    await page.mouse.move(900, 430, steps=20)
    await page.mouse.wheel(0, 260)
    await pause(page, 3)
    await click(page, page.get_by_role("tab", name="Artifacts", exact=True))
    await pause(page, 2)
    await click(page, page.get_by_text("roc_curve.png", exact=True).first)
    await pause(page, 7)
    await click(page, page.get_by_text("confusion_matrix.png", exact=True).first)
    await pause(page, 3)

async def scene5(page):
    await terminal(page, "ls models/", 2)
    await terminal(page, "pytest -q", 3)
    await terminal(page, "python3 -m flake8 src tests --max-line-length=120 --extend-ignore=E203,W503", 2)
    await page.goto(GITHUB + "/actions")
    await pause(page, 3)
    await click(page, page.get_by_role("link", name=CI_RUN_TITLE, exact=True))
    await pause(page, 3)
    await click(page, page.get_by_role("link", name="build-test", exact=True).first)
    await pause(page, 6)

async def scene6(page):
    await source(page, "docker", 3)
    await terminal(page, "docker build -t heart-api:latest .", 1)
    await terminal(page, "docker run -d --name heart-video -p 8001:8000 heart-api:latest", 4)
    await terminal(page, "curl -sS localhost:8001/health", 2)
    await page.goto("http://127.0.0.1:8001/docs")
    await click(page, page.get_by_role("button", name="POST /predict Predict"))
    await pause(page, 1)
    await click(page, page.get_by_role("button", name="Try it out"))
    payload = json.loads((ROOT / "data/sample_input.json").read_text())
    await page.locator("textarea").fill(json.dumps(payload, indent=2))
    await pause(page, 2)
    await click(page, page.get_by_role("button", name="Execute", exact=True))
    await page.get_by_text('"probability": 0.652', exact=False).first.wait_for(timeout=30000)
    await page.get_by_text("Response body", exact=True).scroll_into_view_if_needed()
    await page.mouse.wheel(0, 170)
    await pause(page, 7)

async def scene7(page):
    await source(page, "kubernetes", 5)
    await page.mouse.wheel(0, 420)
    await pause(page, 3)
    await terminal(page, "kubectl apply -f k8s/", 1)
    await terminal(page, "kubectl rollout status deployment/heart-api", 2)
    await terminal(page, "kubectl get pods,svc", 5)
    await terminal(page, "curl -sS localhost:8002/predict -H 'Content-Type: application/json' -d @data/sample_input.json", 5)

async def scene8(page):
    await page.goto("http://127.0.0.1:9091/targets")
    await pause(page, 5)
    await point(page, page.get_by_text("UP", exact=True))
    await pause(page, 2)
    await page.goto("http://127.0.0.1:3001/d/heart-api-dashboard/heart-disease-api-monitoring?from=now-15m&to=now&refresh=5s")
    await pause(page, 6)
    await point(page, page.get_by_text("Total Predictions", exact=True))
    await pause(page, 3)
    await page.mouse.move(1000, 570, steps=25)
    await pause(page, 4)
    await page.mouse.wheel(0, 420)
    await pause(page, 4)
    await terminal(page, "docker compose logs --tail=3 heart-api", 4)

async def scene9(page):
    await page.goto(GITHUB + "/blob/main/README.md")
    await pause(page, 5)
    await page.get_by_role("heading", name="Kubernetes (Minikube)", exact=True).scroll_into_view_if_needed()
    await pause(page, 7)
    await page.mouse.wheel(0, 600)
    await pause(page, 7)

async def main():
    global SCENE_START
    scenes = json.loads((WORK / "scenes.json").read_text())
    state = json.loads((WORK / "browser_state.json").read_text())
    ml = json.loads((WORK / "mlflow_state.json").read_text())
    state["origins"].extend(ml["origins"])
    recordings = []
    selected = [int(x) for x in sys.argv[1:]] or range(1, 10)
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=BROWSER, headless=True)
        for i in selected:
            scene = scenes[i - 1]
            print(f"RECORDING scene {i}: {scene['title']}", flush=True)
            context = await browser.new_context(viewport={"width":1920,"height":1080}, record_video_dir=str(WORK / "raw"), record_video_size={"width":1920,"height":1080}, storage_state=state, device_scale_factor=1)
            await context.add_init_script(POINTER)
            page = await context.new_page()
            page.set_default_timeout(20000)
            started = time.monotonic()
            SCENE_START = started
            EVENTS.clear()
            try:
                await globals()[f"scene{i}"](page)
                elapsed = time.monotonic() - started
                await pause(page, max(2, scene["duration"] + 3 - elapsed))
                await page.screenshot(path=str(WORK / f"scene_{i:02}_end.png"))
                print(f"FINISHED scene {i}: {time.monotonic()-started:.1f}s", flush=True)
            except Exception:
                await page.screenshot(path=str(WORK / f"scene_{i:02}_ERROR.png"))
                print((await page.locator('body').inner_text())[-5000:],flush=True)
                raise
            finally:
                (WORK / f"scene_{i:02}_events.json").write_text(json.dumps(EVENTS, indent=2))
                video=page.video
                await context.close()
                await video.save_as(str(WORK / f"scene_{i:02}.webm"))
            recordings.append(i)
        await browser.close()
    print("Captured scenes", recordings, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
