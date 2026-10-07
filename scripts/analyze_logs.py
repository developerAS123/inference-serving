"""Computes per-mode latency statistics purely by parsing logs/requests.log —
proving the structured logs are genuinely queryable, not just noise on disk."""
import json
from collections import defaultdict

import numpy as np

by_mode = defaultdict(list)

with open("logs/requests.log") as f:
    for line in f:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("mode") and entry.get("latency_ms") is not None:
            by_mode[entry["mode"]].append(entry["latency_ms"])

print(f"{'Mode':<16}{'Count':<8}{'Mean (ms)':<12}{'P50 (ms)':<12}{'P99 (ms)':<12}")
for mode, latencies in by_mode.items():
    arr = np.array(latencies)
    print(f"{mode:<16}{len(arr):<8}{arr.mean():<12.2f}{np.percentile(arr,50):<12.2f}{np.percentile(arr,99):<12.2f}")