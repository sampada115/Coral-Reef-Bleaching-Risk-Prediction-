"""
Performance / load test of the real server (non-functional testing).

    .venv/bin/python testing/perf_test.py

Starts the server on port 8765 (offline LLM, no background refresh), then measures
response times (median / 95th percentile), throughput under concurrent load, forecast
time (cold = model load + inference, warm = cached) and server memory.
Writes testing/reports/performance.json and performance.md.
"""

import concurrent.futures as cf
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

ROOT = Path(__file__).resolve().parent.parent
REPORTS = Path(__file__).resolve().parent / "reports"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
CASE = "2024-03-03"


def timed(fn, n):
    out = []
    for _ in range(n):
        t0 = time.perf_counter(); r = fn(); out.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200, r.text[:200]
    return out


def summary(ms):
    ms = sorted(ms)
    return {"n": len(ms), "median_ms": round(statistics.median(ms), 1),
            "p95_ms": round(ms[max(0, int(len(ms) * 0.95) - 1)], 1), "max_ms": round(ms[-1], 1)}


def main():
    REPORTS.mkdir(exist_ok=True)
    env = dict(os.environ, CORALWATCH_NO_REFRESH="1", CORALWATCH_LLM="offline", CORALWATCH_PORT=str(PORT))
    server = subprocess.Popen([sys.executable, "-m", "backend.app"], cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc = psutil.Process(server.pid)
    try:
        t0 = time.time()
        while True:
            try:
                if httpx.get(f"{BASE}/api/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() - t0 > 120:
                raise SystemExit("server didn't start")
            time.sleep(0.5)
        startup_s = round(time.time() - t0, 1)
        rss_idle = proc.memory_info().rss / 2**20
        c = httpx.Client(base_url=BASE, timeout=600)
        snap = sorted(c.get("/api/health").json()["snapshot_dates"])[-1]
        res = {"startup_s": startup_s}
        print("health…", flush=True)
        res["health"] = summary(timed(lambda: c.get("/api/health"), 100))
        print("map grid…", flush=True)
        res["grid_snapshot"] = summary(timed(lambda: c.get(f"/api/grid/{snap}"), 50))
        rss_peak = proc.memory_info().rss / 2**20
        print("forecast (cold)…", flush=True)
        t = time.perf_counter(); r = c.get(f"/api/forecast?date={CASE}"); cold = (time.perf_counter() - t) * 1000
        assert r.status_code == 200, r.text
        res["forecast_cold"] = {"ms": round(cold, 1), "weather_used": r.json()["weather"], "model": r.json()["model"]}
        rss_peak = max(rss_peak, proc.memory_info().rss / 2**20)
        res["forecast_warm"] = summary(timed(lambda: c.get(f"/api/forecast?date={CASE}"), 20))
        print("AI briefing (offline mode)…", flush=True)
        res["explain_offline"] = summary(timed(lambda: c.post("/api/explain", json={"date": CASE, "zone": "N"}), 10))
        print("concurrent load…", flush=True)
        for workers in (10, 25, 50):
            n = workers * 4
            t = time.perf_counter()
            with cf.ThreadPoolExecutor(workers) as pool:
                lat = list(pool.map(lambda _: timed(lambda: httpx.get(f"{BASE}/api/grid/{snap}", timeout=120), 1)[0], range(n)))
            wall = time.perf_counter() - t
            res[f"load_{workers}_users"] = {**summary(lat), "requests": n, "throughput_rps": round(n / wall, 1), "errors": 0}
        res["memory_mb"] = {"idle": round(rss_idle, 0), "with_models_loaded": round(max(rss_peak, proc.memory_info().rss / 2**20), 0)}
        res["machine"] = {"cpu_cores": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
                          "python": sys.version.split()[0]}
    finally:
        server.terminate(); server.wait(10)

    (REPORTS / "performance.json").write_text(json.dumps(res, indent=1))
    rows = [("Server start-up", f"{res['startup_s']} s", "")]
    for key, label in [("health", "Health check"), ("grid_snapshot", "Map data (saved day)"),
                       ("forecast_warm", "Forecast (cached)"), ("explain_offline", "AI briefing, offline mode")]:
        s = res[key]; rows.append((label, f"{s['median_ms']} ms", f"{s['p95_ms']} ms"))
    rows.insert(3, ("Forecast (first call: load 3 models + run)", f"{res['forecast_cold']['ms'] / 1000:.1f} s", ""))
    md = ["# Performance test", "", f"Machine: {res['machine']['cpu_cores']} cores, {res['machine']['ram_gb']} GB RAM, Python {res['machine']['python']}", "",
          "| Operation | Median | 95th percentile |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    md += ["", "| Concurrent users | Requests | Median | 95th pct | Throughput | Errors |", "|---|---|---|---|---|---|"]
    for w in (10, 25, 50):
        s = res[f"load_{w}_users"]
        md.append(f"| {w} | {s['requests']} | {s['median_ms']} ms | {s['p95_ms']} ms | {s['throughput_rps']} req/s | {s['errors']} |")
    md += ["", f"Server memory: {res['memory_mb']['idle']:.0f} MB idle, {res['memory_mb']['with_models_loaded']:.0f} MB with the forecast models loaded."]
    (REPORTS / "performance.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
