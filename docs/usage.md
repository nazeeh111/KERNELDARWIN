# Installation and use

Basic: Python >=3.11, `python -m pip install .`. CPU controls add `.[control]`; CUDA adds `.[cuda]`. Supported CUDA scope is Linux and NVIDIA compute capability >=8.0 with compatible drivers. `kd doctor` reports availability, optional profiler tools and exact requirements. CPU CI does not install CUDA.

## CLI

```bash
kd --help
kd doctor
kd control --family layernorm --shape 256 513 --stride 2 --seed 42 --store .runs
kd investigate --family softmax --shape 128 1024 --dtype float32 --max-candidates 6 --repeats 40 --wall-seconds 180 --trace --store .runs
kd show .runs/ID.json
kd validate .runs/ID.json
kd train --store .runs --device-kind cuda --output .runs-models/cuda-policy.json
kd investigate --family softmax --shape 128 1024 --model .runs-models/cuda-policy.json
kd regress before.json after.json
kd export .runs/ID.json --output optimized.py
kd lab
```

`investigate` returns exit 2 and saves a blocked reference record if CUDA or Triton is unavailable. Invalid configuration/failed execution returns 1. A completed investigation with no improvement returns 0 and preserves all attempts. Each record is a new file. Existing record IDs are never overwritten.

`train` requires at least 12 observations across two families. Keep models/evaluations outside the run store: it contains investigation records only. CPU and CUDA observations cannot be mixed, and a CPU-trained policy cannot rank a CUDA investigation. GPU hardware transfer is not validated. The current model uses shape/family/layout/configuration features, not measured profiler features. Invalid attempted candidates are retained with a failure penalty and no invented speedup.

`regress` compares two compatible recorded baselines using an independent bootstrap interval. It checks workload, stable device identity and software environment; differences in observed device telemetry are reported. The comparison detects a repeatable difference in those records, not its cause. To investigate a revision slowdown, collect controlled runs of each implementation, inspect source/trace/layout changes, then test the targeted repair. Automatic Git bisection is not implemented. The prepared engine invokes authored templates, not arbitrary revisions.

## Python API

```python
from pathlib import Path
from kerneldarwin.records import WorkloadSpec, ExperimentStore
from kerneldarwin.engine import investigate_cuda, observation_rows
from kerneldarwin.selector import InterventionSelector

spec = WorkloadSpec(family="softmax", shape=(128, 1024), dtype="float32")
# CUDA required; direct API uses the calling process. Prefer CLI for timeout.
record = investigate_cuda(spec, max_candidates=6, repeats=40)
path = ExperimentStore(Path(".runs")).save(record)

observations = observation_rows(ExperimentStore(".runs").list())
model = InterventionSelector().fit(observations)
ranking = model.rank({**spec.to_dict(), "device_kind": "cuda"}, [
    {"block_size": 1024, "num_warps": 4},
    {"block_size": 1024, "num_warps": 8},
])
```

`gpu.reference(family,x,eps)` is the declared PyTorch reference; `gpu.candidate(family,config,x,eps)` is the authored Triton adapter. They are inference functions with finite-input semantics; no autograd contract. CPU-only imports work without PyTorch or Triton.

## Integrate an accepted export

`kd export` accepts only a CUDA record whose winner passed discovery and fresh numerical/timing confirmation. It checks current kernel source identity. The exported callable rechecks source identity and input/device/stride contracts when invoked. It requires the installed matching KERNELDARWIN CUDA package; it is not a vendored independent Triton distribution.

```python
from optimized import run
output = run(cuda_input)
```

Run `python examples/attention.py optimized.py --output application.json` after exporting a contiguous softmax record. It measures attention's `QKᵀ / sqrt(d) → softmax → V` path with eager, optimized and compiled baselines. No application result is bundled until actual CUDA execution. Shape-specific microkernel gains do not imply end-to-end gains.

## Memory safety and profiling

Run numerical checks separately from timing:

```bash
python scripts/sanitizer_check.py
compute-sanitizer --tool memcheck --error-exitcode 86 python scripts/sanitizer_check.py
compute-sanitizer --tool racecheck --error-exitcode 86 python scripts/sanitizer_check.py
compute-sanitizer --tool synccheck --error-exitcode 86 python scripts/sanitizer_check.py
```

Inspect exit codes and logs. Racecheck targets shared-memory hazards; it does not prove all memory accesses safe. Sanitizer results are not automatically attached or marked passed in run records. The script checks the authored configurations at a bounded shape; it does not prove safety for every possible input.

`--trace` captures a separate PyTorch profiler pass and extracts real CUDA duration events. The raw Chrome trace is stored under `STORE/traces/`; traces do not train the selector. Trace overhead is excluded from benchmark samples. A missing profiler/CUPTI path records unavailable evidence. Optional Nsight Compute counters should be collected in a separate profiler pass and interpreted with their units and replay settings. Automatic Nsight/Proton integration is not implemented in this preview. The LAB deliberately omits rooflines/utilization when required measurements are absent.

The manual GitHub GPU workflow requires a trusted self-hosted Linux CUDA runner that you explicitly provision. It never triggers on incoming pull requests. No paid runner or cloud service is provisioned.

## Reproduce CPU evidence

```bash
python -m pip install '.[control]'
python scripts/control_suite.py --output evidence/my-new-run
kd train --store evidence/my-new-run/records --device-kind cpu --output evidence/my-new-run/cpu-model.json
python scripts/check_evidence.py
```

The suite uses 40 paired measurements after warmup, 4 families × 6 shapes × 3 row-batch interventions. NumPy uses float32 candidates against a higher-precision reference and declared tolerances. These results test the software, not GPU search quality. `control_suite.py` refuses to reuse a destination; preserve earlier measurements.

## Troubleshooting

- **No CUDA / Apple Silicon:** use the LAB and CPU controls. Metal does not validate NVIDIA code. Do not relabel CPU runs as CUDA.
- **Triton compilation fails:** preserve the error and software/device versions. The path has not been compiler-tested on the release host. No speedup is accepted from a failed candidate.
- **No candidate accepted:** this is a valid result. Inspect correctness, noise and confirmation before increasing the budget.
- **Existing export path:** choose a new filename; exports never silently overwrite an existing file.
- **Source mismatch:** rerun the investigation with current code; never transplant old acceptance metadata onto changed source.
- **LAB has no bundled records:** serve over HTTP with `kd lab`; local-file fetch restrictions vary. Local import still reads JSON as data. Run `kd validate` to independently re-evaluate record consistency and timing gates.
- **Missing model/service API key:** none is required. No billed model provider is used.


## Inspect and share evidence in the LAB

Use the candidate interval comparison to select an attempt. The finding, matched timing samples, verification ledger and implementation view follow that selection. Discovery and independent confirmation remain distinct; the 1× reference axis does not replace the recorded promotion threshold. Open the complete intervention ledger when you need its additional columns.

Press Command K on macOS or Control K elsewhere to search views, investigations and candidate IDs. Arrow keys move through results, Enter opens one, and Escape closes search and restores focus.

Share selection creates a URL identifying the record, candidate and view. Bundled records open directly; a recipient of a local-record link must import the same JSON. Record contents are never uploaded or persisted by sharing. Raw samples downloads matched timing pairs as an unrounded CSV.

Implementation comparison marks actual line insertions and deletions, retaining both original line numbers. Changes only filters the full comparison; Previous/Next changed line reaches changes outside the current preview window. This is a textual comparison, not semantic equivalence or correctness verification. Oversized source remains downloadable for external review.
