from __future__ import annotations

import threading


def record_metric(
    name: str,
    value_ms: float,
    *,
    metrics_lock: threading.Lock,
    metrics: dict[str, list[float]],
    metrics_window: int,
) -> None:
    if not isinstance(name, str) or not name:
        return
    value = float(value_ms)
    if not (value >= 0):
        return
    with metrics_lock:
        samples = metrics.get(name)
        if samples is None:
            samples = []
            metrics[name] = samples
        samples.append(value)
        if len(samples) > metrics_window:
            del samples[: len(samples) - metrics_window]


def metric_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, float(percentile))) * float(len(sorted_values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    fraction = position - float(lo)
    return float(sorted_values[lo] * (1.0 - fraction) + sorted_values[hi] * fraction)


def metrics_snapshot(
    *,
    metrics_lock: threading.Lock,
    metrics: dict[str, list[float]],
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    with metrics_lock:
        items = list(metrics.items())
    for name, samples in items:
        if not samples:
            continue
        sorted_samples = sorted(float(item) for item in samples)
        out[name] = {
            "count": len(sorted_samples),
            "last_ms": float(samples[-1]),
            "p50_ms": metric_percentile(sorted_samples, 0.50),
            "p95_ms": metric_percentile(sorted_samples, 0.95),
            "max_ms": float(sorted_samples[-1]),
        }
    return out
