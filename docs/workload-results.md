# Workload results and limits

**The first frozen study failed its application performance gate.** Its numerical and resource checks passed, all three confirmation sessions and 24 fresh-process startup runs completed, and no optimized artifact was accepted. The compiled fused-score target also completed three frozen sessions and 18 fresh-process starts, but failed both performance gates. These results do not establish acceleration.

The third target changes stable top-five ranking rather than normalization or scoring. Twelve production Metal methods passed from an independent installed wheel. V1 controls failed before candidate timing. A separately preregistered V2 study then completed development, three untouched confirmations and 24 cold processes, with an empty selected policy and failed performance gates. Its independent installed exporter refused the result. No optimized artifact is accepted.

A later host-input batch API combines document refresh and three query groups in one compiled transaction. Its installed numerical suite passed 28 methods, including 14 native Metal methods. Its single [prospective calibration](workload-transaction-study.md) reached the fixed 180-second cap after 120 completed rank groups and 14 of 40 application groups. It stopped before candidate tuning; the application equality gate was not reached. Independent installed replay matched all completed rank comparisons and verified the stop. Export was refused with no artifact created. This is an incomplete resource-bounded calibration, not a confirmed speedup or a measured custom-kernel loss.

## Executed numerical evidence

| Scope | Executed result | Evidence |
| --- | --- | --- |
| Current stable row normalization, application and installation guards | 18 tests, zero failures, errors or skips; nine GPU methods and nine CPU contract methods | [Current row validation](../evidence/workload-2026-10-29/discovery/metal-validation-row-guarded-current.json) |
| Current fused query normalization and similarity scoring | Eight GPU test methods, zero failures, errors or skips | [Current compiled-path validation](../evidence/workload-2026-10-29/discovery/metal-validation-fused-block128.json) |
| Stable top-five ranking, original V1 measurement identity | 12 production GPU methods, zero failures, errors or skips; separate installed wheel; native API/GPU validation enabled | [V1 ranking validation](../evidence/workload-2026-11-01/rank-v1/numerical-validation.json), [native log](../evidence/workload-2026-11-01/rank-v1/native-validation.log) |
| Stable top-five ranking, current V2 measurement identity | 12 fresh production GPU methods, zero failures, errors or skips; installed module paths recorded directly; native API/GPU validation enabled | [V2 ranking validation](../evidence/workload-2026-11-01/rank-v2/numerical-validation.json), [native log](../evidence/workload-2026-11-01/rank-v2/native-validation.log) |
| Host-input batch transaction and static planner | 28 methods including 14 native Metal methods; zero failures, errors or skips; source-free installation and API/GPU validation; 16 changed-input API/CLI top-match checks | [Batch validation and consumers](../evidence/workload-2026-11-02/transaction-v1/), [retained negative-stride regression](../evidence/workload-2026-11-02/transaction-v1/regression/) |

The original frozen first study retains its earlier 16-test report and exact source snapshot. Later source-guard and measurement-identity changes were separately revalidated in the reports above; historical records were not relabeled. These are separate executed suites, not a claim that every future source revision has passed them. All recorded native suites used Metal API and GPU shader validation. Recorded hardware was a 10-core Apple M4 GPU with 32 GiB memory, MLX and MLX Metal `0.32.0`, Python `3.12.13`, and arm64 macOS `27.2`, build `26B123`. CUDA sanitizers and other-device validation are not implied.

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

## Third target V1: ranking validated, controls rejected

The new operation selects five stable `(score, document index)` pairs without sorting the entire score row. It retains stable normalization, vendor matrix multiplication and query validation. Two authored configurations use one SIMD group or four cooperating groups per row. Their 5,572 source bytes passed the production suite from an independently installed wheel: **12 methods, zero failures/errors/skips**, native Metal API and GPU validation enabled, **15,166,683 bytes** peak MLX memory. The earlier private prototype passed 318 composite checks; those prototype cases are distinct from the production report.

The prospective study includes compiled stable/fast normalization with both full sorting and a library-only repeated-reduction ranker, as well as eager, fixed, per-shape and actual static-dispatch adapters. Development would compare separately measured global, routed and actual-plan costs. Acceptance would require both canonical ranking baselines and all four complete-application libraries in every independent session. V1 produced no development observations, frozen policy, confirmation sessions or cold study.

The first prerequisite measured identical compiled library ranking in randomized pairs, each block forcing both IDs and values for 128 separately evaluated inputs. Every 95% total-cost ratio interval had to fit inside `[0.9,1.1]`. All three tasks failed that gate. Load averages were approximately 273/206/192; power was AC and `pmset` reported no recorded thermal warning. That observation does not identify the cause of the variation.

| Identical-library diagnostic | Notes interval | Manuals interval | Archive interval | Outcome |
| --- | --- | --- | --- | --- |
| Prospective 128-input baseline | [0.8473, 1.0094] | [0.9468, 1.1399] | [0.9450, 1.1804] | Rejected before candidate timing |
| One 1,024-input library-only diagnostic | [0.9519, 1.1287] | [0.9487, 1.0338] | [0.9489, 1.0722] | Rejected; no 1,024-input canonical study |
| One fresh original 128-input check after substantial external load reduction | [0.8604, 1.0000] | [0.9815, 1.1283] | [0.9096, 1.0380] | Rejected before candidate timing |

The longer-block diagnostic used one persistent normalized document matrix per task, distinct queries across the original perturbation range and separately evaluated scores. This avoided retaining 1,024 copies of every document matrix. It took **11.81 seconds** and peaked at **142,830,552 bytes**, within its 180-second/2-GiB caps. Its unchanged noise gate still failed. These are comparisons of the library with itself, not kernel speedups. No custom candidate timing, selection, accepted artifact, portfolio advantage or amortization estimate follows from them.

The [rejected baseline](../evidence/workload-2026-11-01/rank-v1/rejected-baseline.json), [42-file exact source snapshot](../evidence/workload-2026-11-01/rank-v1/source-frozen.zip) and [longer-block raw diagnostic](../evidence/workload-2026-11-01/rank-longblock-assay/result.json) remain preserved. Performance attempts stop under these conditions; repeated trials or weaker gates would not establish a repeatable benefit.

A later audit observed a substantial external change: load had fallen to 12.79/27.87/53.71, compared with 272.56/206.28/191.54 at the original rejection and 121.31/196.96/194.02 before the longer-block diagnostic. Independent review supported exactly one fresh original 128-input check, preserving all source, gates, seeds and prior failures. That [separate prospective attempt](../evidence/workload-2026-11-01/rank-v1-lower-load-20261101/) failed for notes and manuals even as recorded load fell to 8.25/24.04/50.66. No candidate was timed. CPU load is not a measurement of GPU contention or temperature; its reduction did not establish usable timing stability. No further V1 timing attempt followed this rejection.

After explicit goal resumption, a separate [prospective V2 amendment](workload-ranking-v2.md) was independently reviewed before any candidate observations. It measures complete-application controls and changes the rank-only entry rule, with conservative symmetric noise margins and IID/block sensitivity envelopes. This does not reinterpret any V1 rejection. The 147-method CPU suite passed with 29 explicit Metal skips, and a fresh installed wheel passed twelve native numerical methods in 31.07 seconds, peaking at 12,357,492 MLX bytes. The numerical report records actual installed module paths. The single V2 calibration and complete frozen study subsequently ran; the null result follows.

## Third target V2: completed with an empty policy

The separately preregistered [V2 methods](workload-ranking-v2.md) materially relaxed rank-only entry in favor of an application control. The original V1 failures remain unchanged. The exact preregistered 44-file [source packet](../evidence/workload-2026-11-01/rank-v2/source-frozen.zip) was frozen from public commit `dfc5b01` before calibration or candidate observations; its manifest records every hash. Runtime/math, two configurations, tasks, seeds and budgets stayed fixed throughout.

The single calibration passed in **9.45 seconds**, with application negative-control primary interval **[0.9929, 1.0277]**. Rank controls remained wide. The fixed symmetric-noise formula therefore required ranking lower bounds above **2.0886013** against both canonical algorithms and complete-application lower bounds above **1.0561013** against all four optimized libraries. Every primary interval is the envelope of paired IID and circular-block-4/8 sensitivities. The bounds are conservative decision rules, not guaranteed coverage or a proven bound on future noise.

Development took **22.32 seconds**, peaked at **292,998,232 MLX bytes** and selected **zero custom rules**. The best fixed arm was `group128`; per-shape point-cost selection chose `group128`, `group128`, `simd32`. Neither authored kernel's actual-plan rank lower bound cleared the frozen requirement against both canonical baselines on any development task. The resulting empty dispatcher used the complete compiled library fallback. A candidate's point cost or operation improvement does not qualify it for export.

All three untouched confirmations and all 24 fresh-process starts completed. Every confirmation failed both ranking and complete-application gates. Comparisons below are baseline total cost divided by dispatcher total cost, not averages of selected speedup ratios. The dispatcher contains no accepted custom computation.

| Session | Rank primary interval vs stable sorting | Application primary interval vs stable sorting | Rank gate | All-library application gate |
| --- | --- | --- | --- | --- |
| 0 | [0.7980, 0.9477] | [0.9360, 1.1519] | Failed | Failed |
| 1 | [0.9161, 1.2670] | [0.9774, 1.2380] | Failed | Failed |
| 2 | [0.8439, 1.0600] | [0.8589, 1.1297] | Failed | Failed |

All optimized application comparators remain visible:

| Session | Stable sorting | Fast sorting | Stable repeated reduction | Fast repeated reduction |
| --- | --- | --- | --- | --- |
| 0 | [0.9360, 1.1519] | [0.9542, 1.1965] | [0.9992, 1.2465] | [1.0710, 1.3830] |
| 1 | [0.9774, 1.2380] | [0.9621, 1.2056] | [1.1014, 1.4051] | [1.1203, 1.3505] |
| 2 | [0.8589, 1.1297] | [0.9106, 1.1648] | [1.0886, 1.3472] | [1.1072, 1.4742] |

The three processes had distinct IDs. Measured gaps after the prior process's end were 120.061/120.063 seconds and 120.076/120.075 seconds on Unix/monotonic clocks. These are operational separation checks, not proof of independent environmental draws. Execution used fixed gaps without favorable-load scheduling or foreign-job intervention. Calibration recorded battery power and load near 5.05/8.74/21.98; power/load observations are retained for all sessions. No recorded thermal warning is not a GPU temperature measurement or proof against throttling.

Every session recorded **28,192 dispatcher calls and 28,192 fallbacks**, including numerical checks, rank bursts and application searches. These are not user-request totals. The maximum observed MLX peak was **584,782,748 bytes**, below 2 GiB. The two kernels plus shared header remain **5,572 bytes**. The study installed no artifact, so there is no installed optimized artifact size or supported amortization break-even. Every separate portfolio claim flag also failed, including advantages over best fixed/per-shape and a two-authored-kernel portfolio.

The 24-process cold study took **12.69 seconds**, within its 180-second cap. Each method ran three times; OS/driver caches remained. Process time includes setup, identity checks, input preparation and first execution. Pure compilation time was not isolated. These startup medians are descriptive, not a confirmed advantage:

| Method | Median process wall | Median first full application sequence |
| --- | ---: | ---: |
| `stable_library` | 457.18 ms | 37.95 ms |
| `best_fixed` | 452.00 ms | 36.04 ms |
| `per_shape` | 450.06 ms | 38.57 ms |
| `dispatcher` | 676.30 ms | 37.84 ms |

The separately installed numerical wheel replayed the original bounded raw records and returned `numerical_passed: true`, `resource_passed: true`, and `rank_passed`, `application_passed`, `confirmation_passed: false`. Its actual export call raised `Frozen confirmation failed; preserve the null result`; the requested output path remained absent. [Independent replay](../evidence/workload-2026-11-01/rank-v2/verification.json), [actual installed refusal](../evidence/workload-2026-11-01/rank-v2/export-refusal.json), [raw evidence](../evidence/workload-2026-11-01/rank-v2/).

No source, criteria or policy were retuned after candidate observations, and no confirmation was retried. This ranking investigation is closed. A repeatable complete-application benefit, a nonempty accepted exported artifact, independent installed accepted-custom integration and a successful optimized Workload release remain unfulfilled. Exact per-call kernel dispatch, copy/materialization/transfer accounting and pure device elapsed time also remain partial; the earlier whole-process profile does not supply those missing counts.

## Subsequent timing mechanisms and batch-transaction premise

After V2 closed, the user's request for a different timing mechanism led to two bounded library-only diagnostics. Native completion timestamps separated injected host delay from GPU-buffer duration but did not stabilize the controls. A separate 71.70-second comparison found that long balanced ABBA/BAAB windows substantially tightened rank controls while the application interval remained [0.9157,1.0526]. No outliers were removed, candidate results were reinterpreted or V2 sessions retried. [Exact controls, source and limits](../evidence/workload-2026-11-01/rank-noise-mechanisms/).

That remaining application variability motivated a real composition change: raw refresh and three query batches execute in one compiled transaction. A 132-check numerical prototype preserved float64/ID association, old-pipeline score agreement, zero ties and explicit nonfinite rejection across all four library paths and both authored rankers. An instrumented library-only pilot reduced sorting-path command-buffer submissions from thirteen to four per three-task sequence; reduction paths used seven. Its observed wall intervals favored batching. These are development observations with instrumentation overhead, not accepted custom-kernel acceleration or pure device time.

The guarded `search_batches` API then passed two CPU contract methods and ten native Metal methods from a fresh source-free installed wheel, with API/GPU validation enabled and zero failures/errors/skips. A separate installed consumer passed eight changed-input top-match checks. All 72 runtime files matched the executed wheel and actual installed/source bytes. [Source-bound prerequisite evidence and replay](../evidence/workload-2026-11-01/transaction-prerequisites/), [API and limits](workload-transactions.md).

No custom transaction policy or frozen confirmation has run. A future authored comparison must use the equally batched optimized libraries, fresh held-out shapes/sequences and a separate fixed protocol. Batching alone cannot establish a custom-kernel or portfolio advantage. V2 remains an immutable null result, and the accepted-artifact/optimized-release gates remain unfulfilled.

## Later owned-process shader diagnostic

A separate baseline-only Instruments capture on the installed public source produced **1,304 exact-process shader records** with 61 registered shader names. Sort/gather, reductions and matrix multiplication are present. The captured application passed 72 independent float64 score/ID checks. [Sanitized rows, summary and provenance](../evidence/workload-2026-11-01/owned-shader-diagnostic/) retain the source/runtime bindings.

Overlapping and nested intervals are unioned. Per-shader unions are not additive percentages; records are not assumed to equal launch counts. Host markers include input and numerical work, have approximate alignment and omit boundary records through a 2-ms guard. These partial instrumented observations do not establish causal application headroom. Shared accelerator identifiers could not attribute global numeric counters to this process, and the available counter definition described raytracing activity rather than bandwidth or register spills. Raw trace/XML files include unrelated system context and remain private. No application numeric-counter claim is made.

A later extraction from that same recording provides [owned resource and submission accounting](../evidence/workload-2026-11-01/owned-shader-diagnostic/resource-accounting.json). No additional capture or benchmark was run. Independently checked process filtering and cross-table identities establish the following whole-process observations:

| Recorded quantity | Owned-process observation |
| --- | ---: |
| Unique command-buffer submissions | 181 |
| Submissions reporting zero encoders | 13 |
| Unique encoder records, matching submission counts | 168 |
| Buffer-allocation events | 86 |
| Cumulative recorded allocation resource sizes | 19,038,208 bytes |
| Metal allocation-level records | 122 |
| Highest observed `MTLDevice.currentAllocatedSize` | 14,385,152 bytes |

These counts include setup, compilation, validation and input work; dividing them by searches would misstate their scope. Encoder records are not kernel launches. Cumulative allocation sizes are not transfer volume or live memory, and the Metal allocation-level metric differs from MLX's allocator peak. Exact dispatches, implicit contiguous copies, tensor materializations, host/GPU transfer bytes and pure GPU elapsed time per application call remain unmeasured. The available accounting is partial and does not establish acceleration.
