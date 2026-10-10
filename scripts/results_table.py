import json

with open("docs/load_test_results.json") as f:
    results = json.load(f)

modes = list(dict.fromkeys(r["mode"] for r in results))
levels = sorted({r["n_concurrent"] for r in results})
lookup = {(r["mode"], r["n_concurrent"]): r for r in results}


def table(title, key, fmt):
    print(f"**{title}**\n")
    print("| Concurrent requests | " + " | ".join(modes) + " |")
    print("|---:|" + "---:|" * len(modes))
    for n in levels:
        print(f"| {n} | " + " | ".join(fmt.format(lookup[(m, n)][key]) for m in modes) + " |")
    print()


table("Throughput (requests/sec)", "requests_per_sec", "{:.1f}")
table("P50 latency (ms)", "p50_latency_ms", "{:.1f}")
table("P99 latency (ms)", "p99_latency_ms", "{:.1f}")