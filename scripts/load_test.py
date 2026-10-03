"""
Load test for the Inference Serving Capstone API — sends concurrent requests
across all 3 backend modes at several concurrency levels, and generates a
latency/throughput comparison report.

Usage: uv run scripts/load_test.py --base-url http://localhost:8000
"""
import argparse
import asyncio
import json
import os
import time

import aiohttp
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_TEXTS = [
    "This capstone project is coming together nicely.",
    "The service is slow and keeps returning errors.",
    "I'm not sure how I feel about this update.",
    "Absolutely fantastic result, exceeded expectations.",
    "This was a frustrating and disappointing experience.",
]


async def send_one(session, url, mode, text, results):
    start = time.time()
    try:
        async with session.post(url, json={"text": text, "mode": mode}) as resp:
            await resp.json()
            ok = resp.status == 200
    except Exception:
        ok = False
    results.append({"latency_ms": (time.time() - start) * 1000, "ok": ok})


async def run_load_test(base_url, mode, n_concurrent):
    url = f"{base_url}/predict"
    results = []
    async with aiohttp.ClientSession() as session:
        start = time.time()
        tasks = [send_one(session, url, mode, SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)], results)
                 for i in range(n_concurrent)]
        await asyncio.gather(*tasks)
        total_elapsed = time.time() - start

    latencies = [r["latency_ms"] for r in results if r["ok"]]
    n_success = sum(1 for r in results if r["ok"])

    return {
        "mode": mode, "n_concurrent": n_concurrent,
        "requests_per_sec": n_concurrent / total_elapsed,
        "success_rate": n_success / n_concurrent,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else None,
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else None,
    }


def plot_results(all_results, modes, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mode in modes:
        mode_results = [r for r in all_results if r["mode"] == mode]
        xs = [r["n_concurrent"] for r in mode_results]
        axes[0].plot(xs, [r["requests_per_sec"] for r in mode_results], marker="o", label=mode)
        axes[1].plot(xs, [r["p50_latency_ms"] for r in mode_results], marker="o", label=mode)

    axes[0].set_xlabel("Concurrent requests"); axes[0].set_ylabel("Requests/sec")
    axes[0].set_title("Throughput vs Concurrency"); axes[0].legend(); axes[0].grid(True)
    axes[1].set_xlabel("Concurrent requests"); axes[1].set_ylabel("P50 Latency (ms)")
    axes[1].set_title("Latency vs Concurrency"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "load_test_chart.png")
    plt.savefig(path)
    print(f"Saved chart: {path}")


async def main_async(base_url, modes, concurrency_levels, output_dir):
    all_results = []
    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        for n in concurrency_levels:
            r = await run_load_test(base_url, mode, n)
            print(f"  concurrency={n:<4} req/s={r['requests_per_sec']:.1f}  "
                  f"P50={r['p50_latency_ms']:.1f}ms  P99={r['p99_latency_ms']:.1f}ms  "
                  f"success={r['success_rate']*100:.0f}%")
            all_results.append(r)

    with open(os.path.join(output_dir, "load_test_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    plot_results(all_results, modes, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--modes", nargs="+", default=["pytorch-fp32", "onnx-fp32", "onnx-int8"])
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument("--output-dir", default="dashboard")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    asyncio.run(main_async(args.base_url, args.modes, args.concurrency_levels, args.output_dir))


if __name__ == "__main__":
    main()