# Workload results and limits

**The first frozen study failed its application performance gate.** Its numerical and resource checks passed, all three confirmation sessions and 24 fresh-process startup runs completed, and no optimized artifact was accepted. The second target passed seven actual Metal numerical/application tests, then failed its prospective baseline-stability gate before tuning. These results do not establish acceleration.

## Executed numerical evidence

| Scope | Executed result | Evidence |
| --- | --- | --- |
| Stable row normalization, application and installation guards | 16 tests, zero failures, errors or skips; nine GPU test methods and seven CPU contract methods | [Current-source row validation](../evidence/workload-2026-10-29/discovery/metal-validation-study.json) |
| Fused query normalization and similarity scoring | Seven GPU test methods, zero failures, errors or skips | [Fused validation](../evidence/workload-2026-10-29/discovery/metal-validation-fused-v1.json) |

These are separate executed suites, not a claim that every future source revision has passed them. Both used Metal API and GPU shader validation. Recorded hardware was a 10-core Apple M4 GPU with 32 GiB memory, MLX and MLX Metal `0.32.0`, Python `3.12.13`, and arm64 macOS `27.2`, build `26B123`. CUDA sanitizers and other-device validation are not implied.

The checks use independent float64 references, optimized libraries, changed input contents, extreme magnitudes, cancellation, irregular widths, offsets, transposes, negative strides and broadcast views. Application checks verify top-k scores and their associated valid, unique IDs. Failed index refresh preserves the previous index. Strict tiny-normal tests use zero absolute tolerance so an incorrectly zeroed result cannot pass. Float32 subnormals follow native Metal flush-to-zero behavior. [Semantic details](workload-semantics.md).

Tests manufacture plan fixtures to check actual static dispatch, stale-source rejection and fallback. Those fixtures are not accepted performance artifacts. Ordinary CPU CI explicitly skips Metal tests; a skip is not GPU execution evidence.

## Reference application and frozen protocol

The application uses 161 chunks of original Apache-2.0 project documentation, converted to deterministic token-count vectors. It downloads no model or weights. This is a declared reference trace, not production telemetry or a semantic-search quality evaluation.

| Configuration | Document matrix | Query rows | Raw document layout |
| --- | --- | --- | --- |
| Notes | 64 × 384 | 1 | Contiguous |
| Manuals | 161 × 768 | 8 | Column stride 2 |
| Archive | 161 × 1,536 | 17 | Transposed |

Each task performs one index refresh and three changed-query searches. Tasks receive equal frequency; total costs are summed. Full application timing includes input MLX allocation, refresh, normalization, matrix multiplication, ranking and host ID/score materialization. File I/O and token encoding are separate. Confirmation adds an untouched 97 × 513 index, three query rows, column stride 2 and three complete untouched sequence orders.

The first study initially rejected several baseline attempts under extreme host contention. A later fully serialized run passed the unchanged prospective identical-baseline gate: all three 95% total-cost ratio intervals fit inside `[0.9, 1.1]`. They were `[0.9797, 1.0469]`, `[0.9730, 1.0153]` and `[0.9778, 1.0428]`. Before tuning, the study froze normalization lower-bound improvement above **1.10×** and complete-application lower-bound improvement above **1.03×** in every independent session against every required optimized library.

The [frozen protocol](../evidence/workload-2026-10-29/target1-normalization/protocol.json), corpus, numerical report, development bytes, policy and all sessions are source/hash bound. Earlier rejected pairs remain under [discovery](../evidence/workload-2026-10-29/discovery/); the original rejected protocol is historical, not the accepted freeze. Its large noise-derived threshold must not be reused.

## First target: normalization did not confirm an application benefit

Four authored configurations fed the same observations to fixed, per-shape and joint selectors. Development selected `vector4` as best fixed and `vector4` plus `group128` for six exact float32 shapes. The two-kernel policy's weighted normalization cost was **363.62 µs**, versus **385.94 µs** for fixed `vector4`, about 6% lower on development data. Per-shape selection used three configurations. This is a development result, not independent confirmation.

The following costs are mean milliseconds per complete four-task application sequence, from 40 paired samples per session. Each sample averaged four full sequence repetitions. Ratios divide the named library's total cost by dispatcher total cost; values below one favor the library.

| Session | Whole compiled stable | Whole compiled fast | Dispatcher | Dispatcher vs stable, ratio [95% interval] | Dispatcher vs fast, ratio [95% interval] |
| --- | ---: | ---: | ---: | --- | --- |
| 0 | 13.70 ms | 16.39 ms | 14.12 ms | 0.970 [0.908, 1.038] | 1.161 [0.962, 1.497] |
| 1 | 29.00 ms | 27.80 ms | 29.47 ms | 0.984 [0.729, 1.289] | 0.943 [0.776, 1.154] |
| 2 | 56.32 ms | 48.47 ms | 44.30 ms | 1.271 [0.966, 1.650] | 1.094 [0.867, 1.373] |

Every session fails the frozen **1.03** lower-bound application criterion. Session 0's isolated normalization comparison against the separate compiled normalizer was 1.528× [1.355, 1.749], yet its full application failed. This is the practical distinction between a faster operation and a faster application.

For completeness, pooled descriptive means across all 120 application samples were:

| Method | Mean full-sequence cost |
| --- | ---: |
| Eager stable MLX | 37.29 ms |
| Separate compiled stable normalization plus ranking | 33.46 ms |
| Separate optimized RMS-based normalization plus ranking | 33.25 ms |
| Whole compiled stable application | 33.01 ms |
| Whole compiled fast application | 30.89 ms |
| Best fixed `vector4` | 30.55 ms |
| Bounded per-shape selection | 29.22 ms |
| Two-kernel dispatcher | 29.30 ms |

These pooled costs cannot hide an independent-session loss or establish the required benefit. The first target's dispatcher also failed to confirm a normalization advantage over the fixed candidate on the full sequence distribution. Its exact-shape rules sent the untouched task through fallback; the fixed candidate remained semantically eligible there. That cost is retained rather than removed from the comparison.

[Raw confirmations and replayed verification](../evidence/workload-2026-10-29/target1-normalization/verification-null.json) report `numerical_passed: true`, `resource_passed: true`, `normalization_passed: false`, `application_passed: false` and `confirmation_passed: false`. The frozen policy remains unconfirmed. The exact runtime and measurement source is preserved in [the source snapshot](../evidence/workload-2026-10-29/target1-normalization/runtime-measurement-source.zip).

## Search, startup and resource costs

The bounded development search took **7.15 seconds** and reached **163,264,740 bytes** of MLX peak allocation. Confirmation peaks were at most **217,404,396 bytes**, below the 2 GiB limit. The four authored normalization Metal sources total **4,453 bytes**. There is no accepted artifact whose installed size can be reported.

The [cold study](../evidence/workload-2026-10-29/target1-normalization/cold-study.json) ran all eight methods three times in randomized order, each in a fresh Python/MLX process. It took 95.33 seconds. Shared OS and driver caches were not deleted. Reported process wall includes harness identity probes and input preparation; pure compiler time was not isolated.

| Method | Median process wall | Median first full application sequence |
| --- | ---: | ---: |
| Whole compiled stable | 3,158.88 ms | 377.66 ms |
| Whole compiled fast | 3,144.88 ms | 1,062.51 ms |
| Best fixed | 3,355.39 ms | 263.51 ms |
| Two-kernel dispatcher | 6,029.17 ms | 279.35 ms |

Three observations per method under changing contention are descriptive startup costs, not a confirmed cold-start advantage. The verifier's point-estimate amortization calculation is 6,311 sequences, but the performance gate failed. A reliable break-even count is therefore **not established**.

Each confirmation dispatcher recorded 12,832 calls and 3,208 fallbacks, including warm/numerical and operation-burst work as well as application requests. These are dispatch counters, not a count of user searches.

Host load was about 6.6 at the successful baseline freeze and increased materially during later work. Timings and uncertainty are retained as observed; no independent session was rerun for a favorable outcome. Root-owned heavy work was serialized. These records do not identify the cause of other host load. The absence of a `pmset` thermal warning is not a GPU temperature measurement or proof that throttling was absent.

## Profiler evidence and competing explanations

An installed application completed 56 changed-input refresh/search iterations across seven controlled paths with independent score/ID checks. A successful Instruments Metal System Trace contains **157 owned-process active Compute intervals**. [Sanitized execution](../evidence/workload-2026-10-29/discovery/profile-installed.json) and [GPU-interval summary](../evidence/workload-2026-10-29/discovery/profile-gpu-intervals.json) preserve that evidence. Unrelated system context stays local.

The stable and fast complete-query compiler graphs contain 16 and 17 primitive nodes. They reveal graph structure and guide target selection; node counts do not measure time. The interval summary's method alignment is approximate and some phase intervals are omitted. It is not precise device-only benchmark evidence. Hardware counters were unavailable and the shader timeline was disabled, so there is no measured bandwidth, spill, roofline or causal bottleneck claim.

| Competing explanation | Observation and remaining uncertainty |
| --- | --- |
| Reduction/dispatch overhead leaves normalization headroom | Controlled normalization variants produced a first-session operation gain, but the application criterion still failed |
| Whole-query compilation removes useful application headroom | The complete compiled stable graph beat the dispatcher at session 0's point estimate; repeatable relative benefit remains unproven |
| Scheduling, allocation or host results dominate | Wide later intervals and high observed load are consistent with interference; no process-specific causal attribution is made |
| Layout or vector loading matters | Real layouts and scalar/packed variants were checked; no hardware-counter access-cost diagnosis was available |
| Query normalization and scoring can share work | The second target implements this hypothesis; numerical success alone cannot answer its performance question |

The equivalent optimized RMS-based baseline preserves the same clamped, max-scaled L2 semantics. Plain RMS with additive epsilon or direct squaring is not substituted to create an easier baseline. An intentionally unstable direct-square regression fails the independent numerical comparison.

## Second target: numerical success, unstable baseline

`score_simd` uses one SIMD group per query/document pair. `score_tile4` normalizes a query in shared memory and reuses it across four documents. Both execute on the actual GPU and passed the seven-method fused validation suite. Index refresh remains the compiled stable library; unfamiliar tuples use the complete compiled stable query, including ranking and finite checks.

The second study was prospectively bounded to two authored variants, 15 minutes of development, 2 GiB MLX memory and 128 static rules. It reserved fresh input contents, an untouched 113 × 641 index with five query rows and complete sequences. Its identical-baseline intervals failed the unchanged `[0.9, 1.1]` gate:

| Task | Identical-library ratio, 95% interval |
| --- | --- |
| Notes | [0.466, 1.452] |
| Manuals | [0.430, 1.251] |
| Archive | [0.420, 1.318] |

[The raw rejection](../evidence/workload-2026-10-29/target2-fused-scores/rejected-baseline.json) records load averages about 474, 403 and 262. The system was on AC power, charging at 24%; no thermal warning was recorded, which is not a temperature measurement. No candidate timing, tuning, policy selection, application confirmation or cold optimization study followed this rejection. The study stopped rather than retrying for a favorable result.

A separate [independent installed-wheel consumer](../evidence/workload-2026-10-29/discovery/installed-fused-consumer.json) changed data in three repetitions and passed 15 method checks, covering the installed plan, both optimized libraries and both direct authored candidates. The plan reported `library_only`, zero custom dispatches and four fallback calls, including an unknown tuple. Direct candidate execution proves those authored computations ran; it does not establish an accepted installed custom plan or a performance result.

An accepted artifact still requires repeatable end-to-end improvement against the strongest required libraries under the frozen criterion, installed changed-input execution without private development files, and verified release delivery. There is no learned-selector, cross-device, production-adoption or CUDA validation claim. The broader workload-acceleration objective remains incomplete.
