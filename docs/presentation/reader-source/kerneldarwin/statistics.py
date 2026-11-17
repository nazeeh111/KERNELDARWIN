"""Paired latency analysis. All times are positive microseconds."""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def _samples(values: Sequence[float]) -> list[float]:
    samples = list(values)
    if not samples or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("timings must be a nonempty sequence of positive finite numbers")
    return samples


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(samples: Sequence[float]) -> dict:
    values = _samples(samples)
    mean = statistics.fmean(values)
    return {"median_us": statistics.median(values), "p05_us": _quantile(values, .05),
            "p95_us": _quantile(values, .95), "cv": statistics.pstdev(values) / mean,
            "n": len(values)}


def compare_paired(baseline: Sequence[float], candidate: Sequence[float], seed: int = 0,
                   min_speedup: float = 1.03, resamples: int = 2000) -> dict:
    """Bootstrap the ratio of paired medians; preserves pair-level covariance.

    CI is descriptive for this fixed candidate, not a multiple-search guarantee.
    Promotion requires fresh independent confirmation after candidate selection.
    """
    first, second = _samples(baseline), _samples(candidate)
    if len(first) != len(second):
        raise ValueError("paired timings must have equal lengths")
    if not math.isfinite(min_speedup) or min_speedup < 1:
        raise ValueError("min_speedup must be finite and >= 1")
    if type(resamples) is not int or resamples < 100:
        raise ValueError("resamples must be an integer >= 100")
    rng = random.Random(seed)
    count = len(first)
    ratios = []
    for _ in range(resamples):
        indexes = rng.choices(range(count), k=count)
        ratios.append(statistics.median([first[i] for i in indexes]) /
                      statistics.median([second[i] for i in indexes]))
    low, high = _quantile(ratios, .025), _quantile(ratios, .975)
    accepted = count >= 20 and low > min_speedup
    reason = ("repeatable improvement above threshold" if accepted else
              "fewer than 20 independent timing pairs" if count < 20 else
              "confidence interval does not clear minimum improvement")
    return {"speedup": statistics.median(first) / statistics.median(second),
            "ci_low": low, "ci_high": high, "accepted": accepted,
            "reason": reason, "n": count, "min_speedup": min_speedup,
            "confidence_level": .95, "bootstrap_resamples": resamples,
            "method": "paired bootstrap ratio of medians", "seed": seed}


def compare_regression(previous: Sequence[float], current: Sequence[float], seed: int = 0,
                       min_slowdown: float = 1.03) -> dict:
    result = compare_paired(current, previous, seed, min_slowdown)
    return {**result, "slowdown": result["speedup"], "regression": result["accepted"]}
