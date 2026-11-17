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
        from .promotion import require_candidate
        require_candidate(record,winner)


def read_record_file(path, *, validator=None, with_bytes=False):
    """Capture one bounded regular descriptor before JSON admission."""
    import stat
    if not hasattr(os,'O_NONBLOCK'):raise ValueError('Nonblocking record admission unavailable')
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK)
    try:
        info=os.fstat(fd);maximum=50_000_000
        if not stat.S_ISREG(info.st_mode):raise ValueError('Record must be a regular file')
        if info.st_size>maximum:raise ValueError('Record too large (>50 MB)')
        parts=[];total=0
        while total<=maximum:
            part=os.read(fd,min(65536,maximum+1-total))
            if not part:break
            parts.append(part);total+=len(part)
        if total>maximum:raise ValueError('Record too large (>50 MB)')
    finally:os.close(fd)
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate record key')
            result[key]=value
        return result
    payload=b''.join(parts)
    record=json.loads(payload,object_pairs_hook=unique,parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'invalid number {value}')))
    (validator or validate_record)(record)
    return (record, payload) if with_bytes else record


def snapshot_record(record):
    """Detach JSON once so validation and persistence consume the same bytes."""
    return json.loads(json.dumps(record,allow_nan=False))


def publish_exclusive(destination,payload):
    """Publish complete bytes without replacing any existing destination."""
    import warnings
    destination=Path(destination);temporary=None;published=False
    try:
        with tempfile.NamedTemporaryFile('wb',dir=destination.parent,prefix='.record-',delete=False) as handle:
            temporary=Path(handle.name);handle.write(payload);handle.flush();os.fsync(handle.fileno())
        os.link(temporary,destination);published=True
    finally:
        if temporary is not None:
            try:temporary.unlink(missing_ok=True)
            except OSError as exc:
                warnings.warn(f'{"Published complete output;" if published else "Publication failed;"} owned staging cleanup failed: {temporary}: {exc}',RuntimeWarning)


class ExperimentStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, record: dict) -> Path:
        record=snapshot_record(record)
        validate_record(record)
        destination = self.root / f"{record['id']}.json"
        payload = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
        publish_exclusive(destination,payload)
        return destination

    def load(self, identifier: str) -> dict:
        if not isinstance(identifier, str) or not _SAFE_ID.fullmatch(identifier):
            raise ValueError("invalid experiment id")
        record=read_record_file(self.root / f"{identifier}.json")
        if record["id"] != identifier:
            raise ValueError("stored record id does not match filename")
        return record

    def list(self) -> list[dict]:
        return [self.load(path.stem) for path in sorted(self.root.glob("*.json"))]
