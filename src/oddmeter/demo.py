"""Generate deterministic sample telemetry with planted unusual events."""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

FIELDS = ["timestamp", "cpu_pct", "memory_pct", "requests_per_min", "error_pct", "known_event"]


def make_demo(rows: int = 600, anomaly_rate: float = 0.04, seed: int = 7) -> list[dict[str, object]]:
    """Create synthetic service telemetry with a small set of injected spikes."""
    if rows < 10:
        raise ValueError("The demo needs at least 10 rows.")
    if not 0 < anomaly_rate < 0.5:
        raise ValueError("Anomaly rate must be between 0 and 0.5.")
    rng = random.Random(seed)
    count = max(1, round(rows * anomaly_rate))
    anomaly_indices = set(rng.sample(range(rows), count))
    data = []
    for i in range(rows):
        hour_wave = math.sin(2 * math.pi * (i % 144) / 144)
        cpu = max(3, min(90, rng.gauss(38 + 12 * hour_wave, 7)))
        memory = max(20, min(90, rng.gauss(54 + i * 0.006, 3)))
        requests = max(5, rng.gauss(190 + 65 * hour_wave, 22))
        error = max(0, rng.gauss(0.7, 0.28))
        anomaly = i in anomaly_indices
        if anomaly:
            # Different injected patterns teach a detector more than one trick.
            if i % 3 == 0:
                cpu, error = rng.uniform(94, 100), rng.uniform(12, 28)
            elif i % 3 == 1:
                memory, requests = rng.uniform(96, 100), rng.uniform(620, 900)
            else:
                requests, error = rng.uniform(750, 1100), rng.uniform(18, 35)
        data.append({
            "timestamp": f"2026-01-{1 + (i // 144):02d}T{(i % 144) * 10 // 60:02d}:{(i % 6) * 10:02d}:00Z",
            "cpu_pct": round(cpu, 2),
            "memory_pct": round(memory, 2),
            "requests_per_min": round(requests, 2),
            "error_pct": round(error, 2),
            "known_event": "yes" if anomaly else "no",
        })
    return data


def write_demo(path: str | Path, rows: int = 600, anomaly_rate: float = 0.04, seed: int = 7) -> Path:
    """Write generated telemetry to CSV and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = make_demo(rows=rows, anomaly_rate=anomaly_rate, seed=seed)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(data)
    return target
