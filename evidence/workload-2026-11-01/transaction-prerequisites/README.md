# Compiled batch-transaction prerequisites

This is a new application composition, not a reinterpretation of V2. A pure transaction normalizes raw document vectors once, joins three changed-query batches, performs vendor matrix multiplication and top-five ranking, checks document/query finiteness and materializes the original three host-result groups. Both authored configurations and all four optimized library paths receive the same composition. No accepted custom transaction artifact or frozen performance result exists.

The numerical-only prototype passed 132 checks with native Metal API/GPU validation: float64 scores/ID association, agreement with the original sequential pipeline, zero ties, nonfinite rejection, irregular dimensions, layouts and large/tiny magnitudes. Batched vendor GEMM need not reproduce separate-query scores bitwise. Private authored trials separately require exact IDs and selected value bits relative to the same batched stable library.

A separate library-only owned-buffer diagnostic ran forty balanced groups per library with changed inputs. All 640 sample orders and 5,920 owned buffer records replayed exactly; 48 independent float64/ID checks passed. There were no dropped records or pending callbacks, and the observer was restored. The 3.37-second run peaked at 8,349,016 MLX bytes.

| Library path | Sequential buffers per three-task sequence | Batched buffers | Instrumented wall total-cost primary ratio interval |
| --- | ---: | ---: | --- |
| Stable sorting | 13 | 4 | [2.2254, 2.3424] |
| Fast sorting | 13 | 4 | [2.0034, 2.2982] |
| Stable repeated reduction | 13 | 7 | [2.0285, 2.4052] |
| Fast repeated reduction | 13 | 7 | [2.2160, 2.4346] |

These intervals compare sequential library computation with the same library in a batch transaction. They include observer overhead and are a development pilot, not an accepted application speedup, custom-kernel advantage or supported release. Buffer counts are command-buffer submissions, not kernel-dispatch, copy, allocation or transfer counts. Device-duration intervals are not pure kernel time. Any future authored study needs equally composed library baselines, fresh untouched data and separate frozen criteria.

The guarded production `search_batches` API then passed **12 methods from a fresh source-free installed wheel: 2 CPU contract methods and 10 actual Metal methods**, zero failures/errors/skips. Native API/GPU validation was verified in the original log. All six runtime imports were inside the fresh environment; its root had no source tree or development corpus. A separate installed consumer reversed document contents and passed eight top-match checks across two calls. Its dispatch is explicitly compiled-library only. Lower-precision cases are smoke checks, not full float32 adversarial coverage.

Executed wheel SHA-256: `b9f5b7a03a64a3c90c6128e34a0754dcf088fdc64d23125e8f452cbecc55d14f`. All 72 runtime files matched both the wheel and actual installed/current source bytes. Production source identity: `14eb0dbd65076f1830785a1b220e908231da3343b4d21cde1d2538a3d67664ca`. `runtime-prerequisite-source.zip` and its 49-file manifest preserve this exact source version and base public commit. No report was relabeled after a code change.

The prototype `.py` files preserve their original `.work/rank-transaction-premise-20261101` execution locations and are historical sources. `replay_owned_buffers.py` is a portable CPU audit of the raw analysis from a checkout containing the pinned rank measurement helper:

```sh
python evidence/workload-2026-11-01/transaction-prerequisites/replay_owned_buffers.py \
  evidence/workload-2026-11-01/transaction-prerequisites
```

The public API and [changed-input example](../../../examples/transaction_search.py) need no research corpus, tuning cache, UI, network, model weights or billed API. The [API contract](../../../docs/workload-transactions.md) describes mixed dtype and empty-input behavior. Read the original JSON/logs for evidence levels and bounds. All earlier frozen failures remain immutable.
