# Workload results and limits

**The first frozen study failed its application performance gate.** Its numerical and resource checks passed, all three confirmation sessions and 24 fresh-process startup runs completed, and no optimized artifact was accepted. The compiled fused-score target also completed three frozen sessions and 18 fresh-process starts, but failed both performance gates. These results do not establish acceleration.

## Executed numerical evidence

| Scope | Executed result | Evidence |
| --- | --- | --- |
| Current stable row normalization, application and installation guards | 18 tests, zero failures, errors or skips; nine GPU methods and nine CPU contract methods | [Current row validation](../evidence/workload-2026-10-29/discovery/metal-validation-row-guarded-current.json) |
| Current fused query normalization and similarity scoring | Eight GPU test methods, zero failures, errors or skips | [Current compiled-path validation](../evidence/workload-2026-10-29/discovery/metal-validation-fused-block128.json) |

The original frozen first study retains its earlier 16-test report and exact source snapshot. Later source-guard changes were separately revalidated in the reports above; historical records were not relabeled. These are separate executed suites, not a claim that every future source revision has passed them. Both used Metal API and GPU shader validation. Recorded hardware was a 10-core Apple M4 GPU with 32 GiB memory, MLX and MLX Metal `0.32.0`, Python `3.12.13`, and arm64 macOS `27.2`, build `26B123`. CUDA sanitizers and other-device validation are not implied.

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

After the frozen study, independent review tightened three row-path contracts: readable artifact bytes must equal the loaded authored source; the verifier reconstructs prospective minimum criteria and budgets; and future confirmation must independently associate returned IDs with float64 document scores. The old confirmation records checked returned top-k values but did not record ID-association flags, although separate GPU tests and installed consumers did check associations. Current verification therefore also rejects that historical study's older source/contract. It remains a preserved null result, not a current export candidate. Synthetic regression fixtures for these guards are not hardware evidence.

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

The stable and fast complete-query compiler graphs contain 16 and 17 primitive nodes. They reveal graph structure and guide target selection; node counts do not measure time. The interval summary's method alignment is approximate and some phase intervals are omitted. It is not precise device-only benchmark evidence. The hardware counter set was null and the shader timeline was disabled, so there is no measured bandwidth, spill, roofline or causal bottleneck claim.

| Competing explanation | Observation and remaining uncertainty |
| --- | --- |
| Reduction/dispatch overhead leaves normalization headroom | Controlled normalization variants produced a first-session operation gain, but the application criterion still failed |
| Whole-query compilation removes useful application headroom | The complete compiled stable graph beat the dispatcher at session 0's point estimate; repeatable relative benefit remains unproven |
| Scheduling, allocation or host results dominate | Wide later intervals and high observed load are consistent with interference; no process-specific causal attribution is made |
| Layout or vector loading matters | Real layouts and scalar/packed variants were checked; no hardware-counter access-cost diagnosis was available |
| Query normalization and scoring can share work | The second target implements this hypothesis; numerical success alone cannot answer its performance question |

The equivalent optimized RMS-based baseline preserves the same clamped, max-scaled L2 semantics. Plain RMS with additive epsilon or direct squaring is not substituted to create an easier baseline. An intentionally unstable direct-square regression fails the independent numerical comparison.

## Second target: initial rejected baseline

`score_simd` uses one SIMD group per query/document pair. `score_tile4` normalizes a query in shared memory and reuses it across four documents. Both execute on the actual GPU. The current eight-method suite also covers whole-query compilation and changed inputs through both configured paths. Index refresh remains the compiled stable library; unfamiliar tuples use the complete compiled stable query, including ranking and finite checks.

The second study was prospectively bounded to two authored variants, 15 minutes of development, 2 GiB MLX memory and 128 static rules. It reserved fresh input contents, an untouched 113 × 641 index with five query rows and complete sequences. Its identical-baseline intervals failed the unchanged `[0.9, 1.1]` gate:

| Task | Identical-library ratio, 95% interval |
| --- | --- |
| Notes | [0.466, 1.452] |
| Manuals | [0.430, 1.251] |
| Archive | [0.420, 1.318] |

[The raw rejection](../evidence/workload-2026-10-29/target2-fused-scores/rejected-baseline.json) records load averages about 474, 403 and 262. The system was on AC power, charging at 24%; no thermal warning was recorded, which is not a temperature measurement. That version stopped before candidate timing, tuning, policy selection or confirmation. The later compiled implementation and prospective measurement-method change are separately described below; this raw rejection was never promoted into an accepted protocol.

A separate [independent installed-wheel consumer](../evidence/workload-2026-10-29/discovery/installed-fused-consumer.json) changed data in three repetitions and passed 15 method checks, covering the installed plan, both optimized libraries and both direct authored candidates. The plan reported `library_only`, zero custom dispatches and four fallback calls, including an unknown tuple. Direct candidate execution proves those authored computations ran; it does not establish an accepted installed custom plan or a performance result.

An accepted artifact still requires repeatable end-to-end improvement against the strongest required libraries under the frozen criterion, installed changed-input execution without private development files, and verified release delivery. There is no learned-selector, cross-device, production-adoption or CUDA validation claim. The broader workload-acceleration objective remains incomplete.

## Second target: compiled-query frozen confirmation also failed

After whole-query compilation was numerically validated, host load materially fell from hundreds to single digits. A fresh current-source 32-input noise check still failed narrowly; its exact source and raw rejection remain [preserved](../evidence/workload-2026-10-29/target2-compiled-quiet-rejected/). A bounded [library-only assay](../evidence/workload-2026-10-29/discovery/block128-assay/) then tested 128 distinct staged inputs and passed. No candidate had been timed or selected. Before candidate timing, the measurement protocol was prospectively changed to 128 score inputs per block and 16 complete application sequence repetitions per confirmation sample. Development application samples remain individual complete invocations. All thresholds, workload weights, session counts, numerical tolerances and resource limits remained unchanged.

A separate fresh [canonical baseline](../evidence/workload-2026-10-29/target2-compiled-block128/protocol.json) passed with identical-library intervals [0.9772, 1.0270], [0.9254, 1.0146] and [0.9915, 1.0100]. The noise-derived score criterion froze at a **1.149259** lower speedup bound; the application criterion remained **1.03** against both complete compiled libraries in every independent session. Baseline load was about 21.5. Larger blocks enabled a valid prospective experiment; they did not establish performance.

Development took **28.76 seconds**, within the unchanged 900-second budget. Both authored configurations shared the same observations for fixed, per-shape and joint selection, including separately measured routing and fallback overhead. The best global fixed implementation was the stable library. The joint selector chose one exact custom rule, `score_simd` for `[1, 64, 384]`; larger shapes used the complete compiled stable query. Development cost was 5,946.34 µs for this policy versus 6,117.64 µs for best fixed and 5,862.98 µs for per-shape selection. These are development observations, not a confirmed portfolio advantage.

Each confirmation used fresh contents, an unseen 113 × 641 document index with five queries, a complete untouched sequence order and 40 randomized paired samples. Every application repetition includes refresh, transfers, allocation, three changed-query searches per task and host results. Mean milliseconds below are per complete four-task sequence, after dividing by the 16 actual repetitions:

| Session | Whole compiled stable | Whole compiled fast | Dispatcher | Dispatcher vs stable, ratio [95% interval] | Dispatcher vs fast, ratio [95% interval] |
| --- | ---: | ---: | ---: | --- | --- |
| 0 | 7.78 ms | 7.88 ms | 7.48 ms | 1.039 [0.991, 1.092] | 1.052 [1.007, 1.097] |
| 1 | 14.21 ms | 14.13 ms | 13.83 ms | 1.028 [0.923, 1.131] | 1.022 [0.928, 1.127] |
| 2 | 17.31 ms | 18.40 ms | 16.25 ms | 1.065 [0.961, 1.165] | 1.132 [0.988, 1.318] |

All three sessions failed both the frozen score and application gates. None was rerun, and the policy was not retuned. The pooled dispatcher versus fixed application ratio was 1.066 [1.000, 1.139], and versus per-shape was 0.958 [0.893, 1.030]. These descriptive intervals do not establish a portfolio advantage or override any session failure. A reliable amortization break-even is not established.

The 18 serialized independent process starts completed in **15.34 seconds**, below the 180-second limit. Median process wall was 846.42 ms for stable library and 1,356.97 ms for dispatcher; median first application sequences were 28.56 ms and 37.60 ms. Shared OS/driver caches remained. Pure compiler time was not isolated. Development peak MLX allocation was 290,637,544 bytes; confirmation peak was 535,749,744 bytes, below 2 GiB. The two authored score shaders total 4,217 bytes. There is no accepted installed artifact size or reliable payback claim.

The current installed verifier independently replayed raw selection, protocol, source/environment bindings, numerical reports, all sessions and startup/resource checks: [verification](../evidence/workload-2026-10-29/target2-compiled-block128/verification-null.json) reports numerical/resource pass and score/application/confirmation failure. A real [export attempt was refused](../evidence/workload-2026-10-29/target2-compiled-block128/export-refusal.json) and created no output. The exact [runtime and measurement source](../evidence/workload-2026-10-29/target2-compiled-block128/runtime-measurement-source.zip) remains unchanged.

Counter clarification for the preserved measurement metadata: global library **score bursts** increment route counters, while their application calls bypass those counters. Its sentence claiming zero global-library counters is inaccurate. Each session records 20,480 score calls/fallbacks for each global library and 28,184 dispatcher calls with 21,138 fallbacks across validation, score bursts and application work. These are instrumentation totals, not user-search counts. This reporting correction does not change samples, policy, source or acceptance.

The larger-block method resolved the prospective baseline gate. It did not resolve the lack of independently established application benefit. No third target or favorable retry was added. Both failed investigations, losses and initial rejections remain available. No optimized artifact or successful Workload release is claimed.

## Source delivery checks before the larger-block study

The public compiled-query source at `b3f1ae5844f44e3662829fa33a858d54c1e8d6af` passed remote Python 3.11/3.12/3.13 CI: 80 methods, 17 explicit Metal skips, and 63 CPU passes. Its 22 frontend and 20 Chromium cases passed. The later prospective block-size change is separately versioned; earlier CI and downloads are not relabeled as proof of a later revision.

The current independent installed wheel passed [28 row checks](../evidence/workload-2026-10-29/discovery/installed-row-consumer-guarded-current.json), [15 fused checks](../evidence/workload-2026-10-29/discovery/installed-fused-consumer-guarded-current.json), and [two changed NPZ quickstarts plus three explicit CLI failures](../evidence/workload-2026-10-29/discovery/installed-cli-smoke-guarded-current.json). The source tree, corpus and tuning files were absent. Plans remained library-only, while direct authored candidates executed and were independently checked. No accepted custom dispatch or successful optimized release is implied.

## Whole-query compilation compatibility

After the first public source preview, a bounded [compatibility probe](../evidence/workload-2026-10-29/discovery/compiled-score-compatibility/) executed both existing score kernels inside one MLX graph with finite validation, sorting and gathering. All 88 checks passed with native Metal validation. No timing or selection occurred. The exported ordinary-case graph has seven primitive nodes; that is graph-structure evidence, not a device-time, copy-count or causal-performance result.

The current implementation and fixed/per-shape comparators now use that same compiled query scope. Python guards and counters run outside the graph once per request. Cached callables capture only variant and ranking width, with current arrays supplied on every call. The new eight-method GPU report validates both variants across five sizes, changing document/query contents, irregular tails and negative query strides. The local Python suite now runs 80 methods with 17 explicit Metal skips and 63 CPU passes. Earlier CI/download evidence at commit0c312bd remains its own recorded source version; the new revision needs separate delivery checks. That compatibility revision preceded the frozen second-target study above; compatibility alone made no performance claim.

The [independent installed compiled-path suite](../evidence/workload-2026-10-29/discovery/installed-fused-compiled-suite.json) repeated all eight current GPU methods from a fresh wheel environment with no source tree or tuning files. Its synthetic plans exercise dispatch and source guards only; they do not constitute accepted performance artifacts.
