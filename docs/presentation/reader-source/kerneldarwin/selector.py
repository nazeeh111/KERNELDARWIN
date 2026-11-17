"""Small contextual ridge/UCB selector, trained only on supplied observations.

This module ships no pretrained weights. Its uncertainty is model leverage, not
an empirically calibrated probability of GPU improvement.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
from typing import Any
from collections.abc import Callable

from .records import WorkloadSpec

FEATURE_VERSION = 1


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with pivoting, suitable for this tiny ridge system."""
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("singular model matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    return [row[-1] for row in augmented]


def features(workload: dict, candidate: dict, use_context: bool = True) -> list[float]:
    """Versioned features, bounded log scales plus explicit configuration context."""
    spec_keys = ("family", "shape", "dtype", "stride", "eps", "seed")
    spec = WorkloadSpec.from_dict({key: value for key, value in workload.items() if key in spec_keys})
    block = candidate.get("block_size", 256)
    warps = candidate.get("num_warps", 4)
    if type(block) is not int or block <= 0 or type(warps) is not int or warps <= 0:
        raise ValueError("candidate block_size and num_warps must be positive integers")
    b = math.log2(block) / 16
    w = math.log2(warps) / 5
    if not use_context:
        return [1., b, w, b * w]
    rows, columns = spec.shape
    r, c = math.log2(rows) / 24, math.log2(columns) / 24
    family = [float(spec.family == name) for name in ("sum", "softmax", "layernorm", "fused")]
    return [1., r, c, math.log2(spec.stride) / 10, float(spec.dtype == "float32"),
            b, w, r * b, c * b, c * w, b * w, *family,
            *(value * b for value in family)]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _observation(observation: dict) -> dict:
    if not isinstance(observation, dict):
        raise ValueError("observation must be an object")
    result = copy.deepcopy(observation)
    features(result["workload"], result["candidate"])
    result["reward"] = _number(result["reward"], "reward")
    result["cost"] = _number(result["cost"], "cost")
    if result["cost"] <= 0:
        raise ValueError("cost must be positive")
    if type(result["valid"]) is not bool:
        raise ValueError("valid must be boolean")
    if result.get("device_kind") not in ("cpu", "cuda"):
        raise ValueError("device_kind must be cpu or cuda")
    if not isinstance(result.get("group"), str) or not result["group"]:
        raise ValueError("observation requires a nonempty source group")
    return result


class InterventionSelector:
    """Ridge regression of reward adjusted for invalidity and experiment cost.

    Utility = reward - failure_penalty * (not valid) - cost_weight * log1p(cost).
    rank() adds alpha * sqrt(x^T A^-1 x), encouraging uncertain interventions.
    Fit refuses mixing CPU and CUDA evidence. A CPU model cannot rank CUDA work
    when device_kind='cuda' is supplied. Callers must pass device_kind at use.
    """
    def __init__(self, regularization: float = 1., alpha: float = .25,
                 cost_weight: float = .02, failure_penalty: float = 1.,
                 use_context: bool = True):
        self.regularization = _number(regularization, "regularization")
        self.alpha = _number(alpha, "alpha")
        self.cost_weight = _number(cost_weight, "cost_weight")
        self.failure_penalty = _number(failure_penalty, "failure_penalty")
        if self.regularization <= 0 or min(self.alpha, self.cost_weight, self.failure_penalty) < 0:
            raise ValueError("regularization must be positive; other weights must be nonnegative")
        self.use_context = bool(use_context)
        self.count = 0
        self.device_kind: str | None = None
        self.weights: list[float] = []
        self.matrix: list[list[float]] = []
        self.groups: list[str] = []

    def fit(self, observations: list[dict]) -> InterventionSelector:
        if not observations:
            raise ValueError("fit requires measured observations; no pretrained model is bundled")
        measured = [_observation(item) for item in observations]
        kinds = {item["device_kind"] for item in measured}
        if len(kinds) != 1:
            raise ValueError("train CPU and CUDA selectors separately")
        vectors = [features(item["workload"], item["candidate"], self.use_context) for item in measured]
        size = len(vectors[0])
        matrix = [[self.regularization * float(i == j) for j in range(size)] for i in range(size)]
        rhs = [0.] * size
        for item, vector in zip(measured, vectors):
            target = (item["reward"] - self.failure_penalty * (not item["valid"]) -
                      self.cost_weight * math.log1p(item["cost"]))
            for i in range(size):
                rhs[i] += vector[i] * target
                for j in range(size):
                    matrix[i][j] += vector[i] * vector[j]
        weights = _solve(matrix, rhs)
        self.count, self.device_kind = len(measured), measured[0]["device_kind"]
        self.weights, self.matrix = weights, matrix
        self.groups = sorted({item["group"] for item in measured})
        return self

    def rank(self, workload: dict, candidates: list[dict]) -> list[dict]:
        if not self.count:
            raise ValueError("selector is untrained; fit actual observations first")
        requested_kind = workload.get("device_kind")
        if requested_kind is not None and requested_kind != self.device_kind:
            raise ValueError("selector training device kind does not match workload")
        ranked = []
        for candidate in candidates:
            vector = features(workload, candidate, self.use_context)
            uncertainty = math.sqrt(max(0., _dot(vector, _solve(self.matrix, vector))))
            predicted = _dot(vector, self.weights)
            ranked.append({"candidate": copy.deepcopy(candidate), "score": predicted + self.alpha * uncertainty,
                           "predicted_utility": predicted, "uncertainty": uncertainty,
                           "training_device_kind": self.device_kind})
        return sorted(ranked, key=lambda item: (-item["score"], json.dumps(item["candidate"], sort_keys=True)))

    def save(self, path: str | Path) -> None:
        if not self.count:
            raise ValueError("cannot save an untrained selector")
        data = {"schema_version": 1, "feature_version": FEATURE_VERSION,
                "parameters": {"regularization": self.regularization, "alpha": self.alpha,
                               "cost_weight": self.cost_weight, "failure_penalty": self.failure_penalty,
                               "use_context": self.use_context},
                "count": self.count, "device_kind": self.device_kind, "weights": self.weights,
                "matrix": self.matrix, "groups": self.groups,
                "limitation": "Leverage uncertainty is uncalibrated. CPU evidence does not validate GPU selection."}
        Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> InterventionSelector:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or data.get("feature_version") != FEATURE_VERSION:
            raise ValueError("unsupported selector model version")
        result = cls(**data["parameters"])
        result.count = data["count"]
        result.device_kind = data["device_kind"]
        size = 19 if result.use_context else 4
        if type(result.count) is not int or result.count < 1 or result.device_kind not in ("cpu", "cuda"):
            raise ValueError("invalid selector provenance")
        weights, matrix = data["weights"], data["matrix"]
        if len(weights) != size or len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError("invalid selector dimensions")
        result.weights = [_number(value, "weight") for value in weights]
        result.matrix = [[_number(value, "matrix") for value in row] for row in matrix]
        _solve(result.matrix, [1.] * size)
        result.groups = list(data["groups"])
        return result


def _workload_key(item: dict) -> str:
    workload = item["workload"]
    # Seed is deliberately excluded: repeated seeds for the same shape cannot leak.
    return json.dumps({key: workload.get(key) for key in ("family", "shape", "dtype", "stride", "eps")}, sort_keys=True)


def _candidate_key(candidate: dict) -> str:
    return json.dumps(candidate, sort_keys=True)


def _heuristic(workload: dict, candidates: list[dict], kind: str) -> list[dict]:
    # CPU batching favors amortizing Python calls; GPU heuristic favors warp count
    # appropriate to row width. These are declared heuristics, never a ground truth.
    if kind == "cpu":
        return sorted(candidates, key=lambda c: (-c.get("block_size", 256), c.get("num_warps", 4)))
    target = 4 if workload["shape"][-1] <= 4096 else 8
    return sorted(candidates, key=lambda c: (abs(c.get("num_warps", 4) - target), c.get("block_size", 256)))


def _config_vector(candidate: dict) -> list[float]:
    block, warps = candidate.get("block_size", 256), candidate.get("num_warps", 4)
    if type(block) is not int or block <= 0 or type(warps) is not int or warps <= 0:
        raise ValueError("Bayesian baseline requires positive integer block_size/num_warps")
    return [math.log2(block) / 16, math.log2(warps) / 5]


def _rbf(left: list[float], right: list[float], length_scale: float) -> float:
    return math.exp(-sum((a - b) ** 2 for a, b in zip(left, right)) / (2 * length_scale ** 2))


def _gp_posterior(seen: list[list[float]], rewards: list[float], point: list[float],
                  length_scale: float = .25, noise: float = 1e-4) -> tuple[float, float]:
    """Zero-mean, unit-variance RBF Gaussian process posterior (mean, stddev)."""
    if not seen:
        return 0., 1.
    kernel = [[_rbf(a, b, length_scale) + (noise if i == j else 0.)
               for j, b in enumerate(seen)] for i, a in enumerate(seen)]
    covariance = [_rbf(point, point_seen, length_scale) for point_seen in seen]
    mean = _dot(covariance, _solve(kernel, rewards))
    variance = max(0., 1. - _dot(covariance, _solve(kernel, covariance)))
    return mean, math.sqrt(variance)


def bayesian_order(candidates: list[dict], observe: Callable[[dict], float], seed: int = 0,
                   length_scale: float = .25, noise: float = 1e-4,
                   exploration: float = 2.) -> list[dict]:
    """Sequential finite-pool GP-UCB replay; observe is called only after selection.

    One seeded random initial configuration is followed by posterior mean plus
    exploration times posterior standard deviation. Hyperparameters are fixed,
    not selected on the held-out rewards. This is an original compact baseline,
    not an invocation or reproduction of BoTorch's model/acquisition defaults.
    """
    for value, label in ((length_scale, "length_scale"), (noise, "noise"), (exploration, "exploration")):
        if _number(value, label) <= 0:
            raise ValueError(f"{label} must be positive")
    pool = sorted(copy.deepcopy(candidates), key=_candidate_key)
    if len({_candidate_key(candidate) for candidate in pool}) != len(pool):
        raise ValueError("Bayesian baseline requires unique candidates")
    vectors = {_candidate_key(candidate): _config_vector(candidate) for candidate in pool}
    rng = random.Random(seed)
    selected, seen, rewards = [], [], []
    while pool:
        if not selected:
            index = rng.randrange(len(pool))
        else:
            scores = []
            for candidate in pool:
                mean, deviation = _gp_posterior(seen, rewards, vectors[_candidate_key(candidate)], length_scale, noise)
                scores.append(mean + exploration * deviation)
            # Stable tie breaking is independent of unobserved outcomes.
            index = max(range(len(pool)), key=lambda i: (scores[i], -i))
        candidate = pool.pop(index)
        selected.append(candidate)
        seen.append(vectors[_candidate_key(candidate)])
        rewards.append(_number(observe(copy.deepcopy(candidate)), "observed reward"))
    return selected


def evaluate_grouped(observations: list[dict], seed: int = 0) -> dict:
    """Offline leave-family/shape-out replay on completely observed candidate sets.

    This evaluation does not launch experiments or reproduce Triton autotuning
    or agent systems. A compact sequential GP-UCB baseline reveals only selected
    outcomes. Measured rewards/costs are replayed, not imputed; invalid rewards=-1.
    """
    measured = [_observation(item) for item in observations]
    if not measured:
        return {"status": "insufficient_data", "reason": "No observations", "splits": {}}
    kinds = {item["device_kind"] for item in measured}
    if len(kinds) != 1:
        raise ValueError("evaluate CPU and CUDA observations separately")
    kind = next(iter(kinds))
    rng = random.Random(seed)
    splits = {}
    for split_name in ("family", "shape"):
        def split_key(item: dict) -> str:
            return str(item["workload"]["family"]) if split_name == "family" else json.dumps(item["workload"]["shape"])
        labels = sorted({split_key(item) for item in measured})
        if len(labels) < 2:
            splits[split_name] = {"status": "insufficient_data", "reason": f"Need at least two distinct {split_name} groups"}
            continue
        rows = []
        for heldout in labels:
            train = [item for item in measured if split_key(item) != heldout]
            test = [item for item in measured if split_key(item) == heldout]
            models = {"learned_ucb": InterventionSelector().fit(train),
                      "no_uncertainty": InterventionSelector(alpha=0).fit(train),
                      "no_context": InterventionSelector(use_context=False).fit(train)}
            workloads: dict[str, list[dict]] = defaultdict(list)
            for item in test:
                workloads[_workload_key(item)].append(item)
            for key, items in sorted(workloads.items()):
                # Aggregate repeat measurements of an intervention inside holdout.
                outcomes: dict[str, list[dict]] = defaultdict(list)
                for item in items:
                    outcomes[_candidate_key(item["candidate"])].append(item)
                if len(outcomes) < 2:
                    continue
                candidates = [group[0]["candidate"] for group in outcomes.values()]
                workload = {**items[0]["workload"], "device_kind": kind}
                rewards = {identifier: sum(item["reward"] if item["valid"] else -1. for item in group) / len(group)
                           for identifier, group in outcomes.items()}
                costs = {identifier: sum(item["cost"] for item in group) / len(group)
                         for identifier, group in outcomes.items()}
                orders = {name: [rank["candidate"] for rank in model.rank(workload, candidates)] for name, model in models.items()}
                orders["heuristic"] = _heuristic(workload, candidates, kind)
                # Random repeated permutations reduce dependence on one lucky seed.
                random_orders = [rng.sample(candidates, len(candidates)) for _ in range(32)]
                # Each run starts cold on this workload. The callback reveals only
                # the chosen candidate's recorded reward after its selection.
                bayesian_orders = [bayesian_order(candidates, lambda candidate: rewards[_candidate_key(candidate)],
                                                  seed=rng.randrange(2**32)) for _ in range(32)]
                replay_orders = {"random": random_orders, "bayesian_ucb": bayesian_orders}
                oracle = max(rewards.values())
                for budget in range(1, len(candidates) + 1):
                    for name, order in orders.items():
                        chosen = [_candidate_key(c) for c in order[:budget]]
                        rows.append({"heldout": heldout, "workload": key, "policy": name, "budget": budget,
                                     "best_reward": max(rewards[c] for c in chosen),
                                     "cost": sum(costs[c] for c in chosen), "oracle_reward": oracle})
                    for policy, policy_orders in replay_orders.items():
                        rows.append({"heldout": heldout, "workload": key, "policy": policy, "budget": budget,
                                     "best_reward": sum(max(rewards[_candidate_key(c)] for c in order[:budget]) for order in policy_orders) / len(policy_orders),
                                     "cost": sum(sum(costs[_candidate_key(c)] for c in order[:budget]) for order in policy_orders) / len(policy_orders),
                                     "oracle_reward": oracle})
        aggregates = []
        for policy, budget in sorted({(row["policy"], row["budget"]) for row in rows}):
            selected = [row for row in rows if row["policy"] == policy and row["budget"] == budget]
            aggregates.append({"policy": policy, "budget": budget, "workloads": len(selected),
                               "mean_best_reward": sum(row["best_reward"] for row in selected) / len(selected),
                               "mean_cost": sum(row["cost"] for row in selected) / len(selected),
                               "mean_oracle_gap": sum(row["oracle_reward"] - row["best_reward"] for row in selected) / len(selected)})
        splits[split_name] = {"status": "evaluated" if rows else "insufficient_data", "groups": labels,
                              "folds": len(labels), "aggregates": aggregates, "per_workload": rows}
    return {"status": "evaluated", "schema_version": 1, "device_kind": kind,
            "observations": len(measured), "seed": seed, "splits": splits,
            "method": "leave-one-family/shape-out offline candidate replay; repeated seeds kept together",
            "bayesian_baseline": {"method": "sequential finite-pool GP-UCB", "kernel": "RBF, unit variance, zero mean",
                                  "length_scale": .25, "observation_noise_variance": 1e-4, "exploration": 2.,
                                  "configuration_features": "log2(block_size)/16, log2(num_warps)/5",
                                  "initialization": "one seeded random configuration", "restarts_per_workload": 32,
                                  "implementation": "stdlib baseline; not BoTorch"},
            "limitations": ["No GPU benefit can be inferred from CPU controls.",
                            "Offline replay uses only observed candidates. GP-UCB updates after each selected outcome; other policies use fixed initial rankings.",
                            "Leverage uncertainty is uncalibrated; no hardware holdout or causal-diagnosis ablation is claimed.",
                            "Triton autotuning and profile-guided agent comparisons are not reproduced. GP-UCB is a fixed-hyperparameter finite-pool baseline, not BoTorch.",
                            "Costs use the unit supplied by the observation recorder; policies compared by equal candidate budget."]}
