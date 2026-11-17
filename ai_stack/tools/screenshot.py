"""Screenshot the running studio with headless Chrome (DevTools protocol), after it is really ready.

    .venv/bin/python tools/screenshot.py out/studio.png "http://localhost:8790/#compose&seek=1.2&view=front"

Waits until every mesh is loaded and the page's startup (including a #compose run) has finished, then captures.
Used to check UI changes without a display.
"""
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets


async def shoot(url, out, width=1500, height=920, timeout=90):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        raise SystemExit("needs google-chrome or chromium")
    port = 9333
    prof = tempfile.mkdtemp(prefix="zang-shot-")
    gpu_flags = os.environ.get("ZANG_SHOT_FLAGS", "--use-angle=swiftshader --enable-unsafe-swiftshader").split()
    proc = subprocess.Popen([chrome, "--headless=new", *gpu_flags,
                             f"--remote-debugging-port={port}", f"--window-size={width},{height}",
                             f"--user-data-dir={prof}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.1)
        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            n = 0

            errors = []

            async def call(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("method") == "Runtime.exceptionThrown":
                        d = msg["params"]["exceptionDetails"]
                        errors.append(f"{d.get('text')} {d.get('exception', {}).get('description', '')[:300]} @ {d.get('url', '')}:{d.get('lineNumber')}")
                    elif msg.get("method") == "Runtime.consoleAPICalled" and msg["params"]["type"] in ("error", "warning"):
                        errors.append("console." + msg["params"]["type"] + ": " + " ".join(str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"])[:300])
                    elif msg.get("method") == "Log.entryAdded" and msg["params"]["entry"]["level"] == "error":
                        e = msg["params"]["entry"]
                        errors.append(f"log: {e['text'][:300]} {e.get('url', '')}")
                    if msg.get("id") == n:
                        return msg.get("result", {})

            await call("Page.enable")
            await call("Runtime.enable")
            await call("Log.enable")
            await call("Page.navigate", url=url)
            t0 = time.time()
            ready = "window.__ready === true && window.__meshes && window.__meshes.loaded === window.__meshes.total"
            while time.time() - t0 < timeout:
                r = await call("Runtime.evaluate", expression=ready, returnByValue=True)
                if r.get("result", {}).get("value"):
                    break
                await asyncio.sleep(0.25)
            else:
                print("warning: page not ready before timeout, capturing anyway")
            await asyncio.sleep(0.6)  # a few render frames
            shot = await call("Page.captureScreenshot", format="png")
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "wb") as fh:
                fh.write(base64.b64decode(shot["data"]))
            print(f"{out} ({time.time() - t0:.1f}s)")
            for e in errors:
                print("  JS:", e)
    finally:
        proc.terminate()
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(shoot(sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8790/", sys.argv[1]))
