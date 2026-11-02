# Installed batch-call resource accounting

This fixed eight-call diagnostic measures Metal API activity in its own process. It ran the unchanged installed `TransactionPlan.search` library default on the three declared development profiles and a negative-stride fallback, with two changed-input seeds. It does not resume the closed calibration, time authored candidates or pass a performance gate. No accepted artifact exists.

The run completed in 0.893 seconds, with 7,835,052 bytes peak MLX memory, Apple API/GPU validation enabled, no dropped records or pending callbacks, and successful hook restoration. Every output passed float64 score and ID-association checks; all inputs were preserved. All 76 installed runtime files still equal the previously validated package and current source. No product runtime or study pin changed.

| Profile | Compute dispatches | Generic copy dispatches | Logical host inputs | MLX constructor copy span | Additional host adapter copy | Numeric host output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Notes, contiguous | 27 | 3 | 102,912 B | 102,912 B | 0 B | 120 B |
| Manuals, column stride 2 | 28 | 3 | 568,320 B | 1,062,908 B | 0 B | 960 B |
| Archive, transposed | 29 | 3 | 1,302,528 B | 1,302,528 B | 0 B | 2,040 B |
| Notes, negative strides | 27 | 3 | 102,912 B | 102,912 B | 102,912 B | 120 B |

Both seeds produced these same counts. Compute dispatches and names are directly observed at the Metal API. The three `gg1_copyfloat32float32` launches each cover exactly one query group's element count. Their correspondence to query concatenation is a source-supported inference: buffer identities and offsets were not captured. No other named copy kernel or explicit blit-copy call appeared in these scopes. That does not prove the absence of every possible fused memory movement.

The host-copy columns are source-derived logical quantities, checked against the actual shapes/strides and guard path. The pinned [MLX 0.32 CPU-array conversion](https://github.com/ml-explore/mlx/blob/v0.32.0/python/src/convert.cpp#L141-L163) copies the full positive-stride storage span, preserving layout. For the stride-2 document matrix, this includes padding between logical elements. The adapter first copies negative-stride inputs to dense storage; the MLX constructor then copies those dense arrays. These are four constructor calls per request. Output payload counts describe five IDs and scores per query plus two separate finite-check scalar bytes; they exclude Python object overhead and hardware cache-line traffic. They are not measured physical transfer or memory-bus bytes.

Each observed scope submitted two command buffers and received two successful completion callbacks. The scope includes the diagnostic's final `mx.synchronize`; no explicit API-return marker was recorded. Therefore two must not be presented as an exact API-only submission count. Dispatch geometry, names and raw event order are retained in `result.json`.

No new Metal buffer allocation API was observed in these warmed scopes. The active/cached memory readings remained unchanged across each call. The [MLX allocator](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/allocator.cpp#L116-L165) can reuse buffers from its cache. Zero new buffers does not mean zero logical arrays, host copies or allocation requests. CPU heap allocations, cache-coherence traffic, exclusive GPU duration and the split between compilation and driver-cache work remain unmeasured.

The positive control performs a verified 16-byte blit, device/heap allocations, copied and no-copy buffer creation, and one named dispatch through each compute entry point. Those records verify that zero observed application allocations/blits are not merely untested hooks. The helper records no pointers or buffer contents and touches only methods in its own process. It restores them before exit. No system trace, foreign process record or native binary is distributed.

- `preregistered.json`, `started.json`, `process.json`: fixed source/binary hashes, limits and one bounded attempt.
- `probe.py`, `call_counters.mm`: exact executed task-local producer and observer source. The historical producer expects its original `.work/batch-call-accounting-20261102` location and temporary helper build; it is not a new installed tracing service.
- `result.json`, `native.log`: owned raw events, numerical checks, actual installed import, runtime identity and backend-validation activation.
- `source-audit.json`: unchanged installed runtime/environment and post-run helper identity.
- `replay.py`, `replayed.json`: separately implemented CPU reconstruction of all eight scopes and positive controls, including dimensions, tags, counts, copy spans and resource limits.

Run the portable CPU replay with `python evidence/workload-2026-11-02/batch-call-accounting/replay.py`. Optional `--output /tmp/new-call-audit.json` writes exclusively and preserves existing reports. CPU replay cannot establish GPU execution or performance by itself; the actual native producer supplies the execution evidence above.
