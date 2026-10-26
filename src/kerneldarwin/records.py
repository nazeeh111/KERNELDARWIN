"""Versioned, JSON-only experiment contracts and atomic local persistence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

FAMILIES = ("sum", "softmax", "layernorm", "fused")
DTYPES = ("float32", "float16", "bfloat16")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class WorkloadSpec:
    family: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    stride: int = 1
    eps: float = 1e-5
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        self.validate()

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if len(self.shape) != 2:
            raise ValueError("shape must contain exactly two dimensions")
        for dim in self.shape:
            _positive_int(dim, "shape dimension")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        _positive_int(self.stride, "stride")
        if isinstance(self.eps, bool) or not isinstance(self.eps, (int, float)) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result

    @classmethod
    def from_dict(cls, value: dict) -> WorkloadSpec:
        value = dict(value)
        actual_strides = value.pop("actual_strides", None)
        if actual_strides is not None:
            if not isinstance(actual_strides, (list, tuple)) or len(actual_strides) != 2:
                raise ValueError("actual_strides must contain two positive element strides")
            for stride in actual_strides:
                _positive_int(stride, "actual stride")
        spec = cls(**value)
        if actual_strides is not None and actual_strides[1] != spec.stride:
            raise ValueError("actual last-dimension stride differs from declared stride")
        return spec


@dataclass(frozen=True)
class CandidateConfig:
    block_size: int = 256
    num_warps: int = 4

    def __post_init__(self) -> None:
        _positive_int(self.block_size, "block_size")
        if self.block_size & (self.block_size - 1) or self.block_size > 65536:
            raise ValueError("block_size must be a power of two <= 65536")
        if type(self.num_warps) is not int or self.num_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError("num_warps must be 1, 2, 4, 8, 16 or 32")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def candidate_id(self) -> str:
        return f"b{self.block_size}-w{self.num_warps}"


def _finite_json(value: Any, path: str = "record") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a nonfinite number")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} has a non-string key")
            _finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_json(item, path)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError(f"{path} contains a non-JSON value")


def _validate_promotion_phase(phase: dict, label: str) -> None:
    """Recompute the claimed statistical gate; never trust an accepted flag."""
    from .statistics import compare_paired
    if not isinstance(phase, dict):
        raise ValueError(f"{label} requires paired timing evidence")
    baseline = phase.get("baseline_us", [])
    candidate = phase.get("candidate_us", [])
    if not isinstance(baseline, list) or not isinstance(candidate, list) or len(baseline) != len(candidate) or not 20 <= len(baseline) <= 10000:
        raise ValueError(f"{label} requires 20..10000 matched timing pairs")
    for duration in baseline + candidate:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{label} timing samples must be positive finite numbers")
    comparison = phase.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError(f"{label} requires comparison metadata")
    threshold = comparison.get("min_speedup")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 1.03:
        raise ValueError(f"{label} minimum speedup must be finite and >=1.03")
    seed, resamples = comparison.get("seed"), comparison.get("bootstrap_resamples")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError(f"{label} requires an integer bootstrap seed in [0, 2**64)")
    if type(resamples) is not int or not 100 <= resamples <= 10000 or resamples * len(baseline) > 2_000_000:
        raise ValueError(f"{label} bootstrap validation exceeds bounded resampling budget")
    if type(comparison.get("n")) is not int or comparison["n"] != len(baseline):
        raise ValueError(f"{label} sample count differs from raw samples")
    expected = compare_paired(baseline, candidate, seed=seed, min_speedup=threshold, resamples=resamples)
    if not expected["accepted"] or comparison.get("accepted") is not True:
        raise ValueError(f"{label} raw samples do not clear the declared improvement gate")
    for key in ("speedup", "ci_low", "ci_high", "confidence_level"):
        claimed = comparison.get(key)
        if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed) or not math.isclose(claimed, expected[key], rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError(f"{label} {key} differs from recomputed raw-sample statistics")
    if comparison.get("method") != expected["method"]:
        raise ValueError(f"{label} statistical method is unsupported or inconsistent")


def validate_record(record: dict) -> None:
    """Validate consistency, not authenticity; JSON claims cannot attest execution."""
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    _finite_json(record)
    if type(record.get("schema_version")) is not int or record.get("schema_version") != 1:
        raise ValueError("unsupported schema_version (expected 1)")
    identifier = record.get("id", "")
    if not isinstance(identifier, str) or not _SAFE_ID.fullmatch(identifier):
        raise ValueError("id must be a safe filename identifier")
    if record.get("status") not in ("accepted", "no_improvement", "blocked", "failed"):
        raise ValueError("invalid run status")
    kind = record.get("evidence_kind")
    if kind not in ("cuda_run", "cpu_control", "reference"):
        raise ValueError("invalid evidence_kind")
    WorkloadSpec.from_dict(record.get("workload", {}))
    device_kind = record.get("device", {}).get("kind")
    if device_kind not in ("cpu", "cuda"):
        raise ValueError("device.kind must be cpu or cuda")
    if kind == "cuda_run" and (device_kind != "cuda" or record.get("provenance", {}).get("gpu_execution") is not True):
        raise ValueError("cuda_run requires CUDA device and explicit GPU execution provenance")
    if kind == "cpu_control" and device_kind != "cpu":
        raise ValueError("CPU control must use a CPU device")
    if kind == "reference" and record.get("status") == "accepted":
        raise ValueError("reference records cannot claim an accepted optimization")
    experiments = record.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("experiments must be an array")
    ids: set[str] = set()
    winner = None
    for experiment in experiments:
        if not isinstance(experiment, dict) or not isinstance(experiment.get("id"), str):
            raise ValueError("each experiment requires a string id")
        if experiment["id"] in ids:
            raise ValueError("duplicate experiment id")
        ids.add(experiment["id"])
        if experiment["id"] == record.get("winner_id"):
            winner = experiment
        if experiment.get("status") == "accepted":
            if record["status"] != "accepted" or experiment["id"] != record.get("winner_id"):
                raise ValueError("Only the promoted run winner may have accepted status")
            if experiment.get("correctness", {}).get("passed") is not True:
                raise ValueError("accepted experiment requires passing correctness")
            if experiment.get("timing", {}).get("comparison", {}).get("accepted") is not True:
                raise ValueError("accepted experiment requires passing performance gate")
        timing = experiment.get("timing", {})
        for key in ("baseline_us", "candidate_us"):
            for duration in timing.get(key, []):
                if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
                    raise ValueError("timing samples must be positive finite numbers")
    winner_id = record.get("winner_id")
    if winner_id is not None and (winner is None or winner.get("status") != "accepted"):
        raise ValueError("winner_id must identify an accepted experiment")
    if record["status"] == "accepted" and winner_id is None:
        raise ValueError("accepted run requires winner_id")
    if winner is not None:
        _validate_promotion_phase(winner.get("timing"), "Discovery")
        confirmation = winner.get("confirmation")
        _validate_promotion_phase(confirmation, "Independent confirmation")
        if confirmation.get("correctness_passed") is not True:
            raise ValueError("Independent confirmation requires passing correctness")
        confirmation_seed = confirmation.get("seed")
        if type(confirmation_seed) is not int or not 0 <= confirmation_seed < 2**64:
            raise ValueError("Independent confirmation requires an explicit fresh input seed")
        discovery_seeds = {record["workload"].get("seed", 0)}
        for experiment in experiments:
            discovery = experiment.get("timing", {})
            for seed in (discovery.get("seed"), discovery.get("comparison", {}).get("seed")):
                if type(seed) is int:
                    discovery_seeds.add(seed)
            for test in experiment.get("correctness", {}).get("tests", []):
                if isinstance(test, dict) and type(test.get("seed")) is int:
                    discovery_seeds.add(test["seed"])
        if confirmation_seed in discovery_seeds or confirmation["comparison"]["seed"] in discovery_seeds:
            raise ValueError("Independent confirmation seeds must differ from discovery seeds")
        declared_strides = record["workload"].get("actual_strides")
        confirmed_strides = confirmation.get("actual_strides")
        if declared_strides is not None and confirmed_strides != declared_strides:
            raise ValueError("Independent confirmation must preserve actual workload strides")
        if confirmed_strides is not None:
            WorkloadSpec.from_dict({**record["workload"], "actual_strides": confirmed_strides})


class ExperimentStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, record: dict) -> Path:
        validate_record(record)
        destination = self.root / f"{record['id']}.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite experiment {record['id']}")
        payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.root, prefix=".record-", delete=False) as handle:
                temporary = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            # Hard-link creation is atomic and refuses concurrent overwrite.
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        return destination

    def load(self, identifier: str) -> dict:
        if not isinstance(identifier, str) or not _SAFE_ID.fullmatch(identifier):
            raise ValueError("invalid experiment id")
        with (self.root / f"{identifier}.json").open(encoding="utf-8") as handle:
            record = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid number {value}")))
        validate_record(record)
        if record["id"] != identifier:
            raise ValueError("stored record id does not match filename")
        return record

    def list(self) -> list[dict]:
        return [self.load(path.stem) for path in sorted(self.root.glob("*.json"))]
