# Prospective ranking protocol V2

This protocol was defined before any ranking candidate timing or selection. It changes the entry rule because the intended result is a faster complete application. It does not turn earlier failed controls into passing results. The original rank-only prerequisite, longer-block diagnostic and lower-load attempt remain preserved with their original source, criteria and outcomes.

The V2 implementation has received independent source/replay review. The CPU suite passed 147 methods with 29 explicit Metal skips. A fresh independent installed wheel passed all twelve actual numerical methods with native validation. One calibration attempt and any candidate study are still pending. No optimized artifact or acceleration is accepted.

## Why amend the measurement design?

V1 required every identical-library rank-only interval to fit inside `[0.9,1.1]`. It stopped before measuring application stability. Its rank timer included Python graph construction, backend evaluation and synchronization for small operations. A separate library-only probe observed no garbage collection in 64 measured cycles, but did not identify a cause of variability or isolate device time. [Diagnostic evidence](../evidence/workload-2026-11-01/rank-measurement-diagnostic/).

V2 explicitly measures complete-application negative controls before tuning. This materially relaxes rank-only entry. Final numerical checks remain mandatory, and final improvement requirements use conservative noise margins and temporal-dependence sensitivity. No candidate result informed this amendment.

| Rule | V1 | V2 |
| --- | --- | --- |
| Entry prerequisite | Each rank-only control interval inside `[0.9,1.1]` | Complete development-application control interval inside `[0.9,1.1]` |
| Rank-only uncertainty | IID paired bootstrap | Envelope of IID and circular blocks of four/eight rounds |
| Rank lower-bound improvement | At least 1.10, raised by observed noise | At least 1.10, raised by squared symmetric noise envelope |
| Application lower-bound improvement | 1.03 | At least 1.03, raised by squared symmetric noise envelope |
| Session execution | Separate process invocations and fixed seeds | Same, with at least 120 seconds between sessions on both clocks |
| Portfolio evidence | Development selection and descriptive confirmation | Separate untouched-session claim flags |

## Controls, timing boundaries and uncertainty

Rank controls retain the original three development tasks, 128 distinct evaluated inputs, five warmup bursts and forty randomized identical-library pairs. The original rank-only hard interval cap is absent in V2; rank uncertainty raises the required improvement instead.

Application controls use the same `stable_library` method instance for A and B. Each timed block executes sixteen complete three-task development sequences and divides wall time by sixteen. Nested host outputs remain alive through that timed call, matching confirmation's output lifetime. Every application transfers arrays, refreshes the index, normalizes queries, computes similarity, ranks, checks finiteness and materializes IDs/scores. Inputs change between paired rounds; each task includes three changed queries. The first call, independent float64 checks and five warmup blocks use order `[0,1,2]`. Later control rounds cycle three fixed task orders and use a separate fixed A/B-order seed.

Calibration uses only development tasks and seeds. It does not execute the held-out shape or confirmation-only seeds. Its three-task control cannot establish stability of the untouched four-task confirmation sequence. The latter must still pass its actual comparison intervals.

Each comparison uses 4,000 paired resamples for each of three interval families. IID resampling draws individual paired rounds. Circular block resampling draws consecutive groups of four or eight paired rounds, wrapping within the recorded sequence; each method receives identical sampled indices. The primary interval spans the smallest lower bound and largest upper bound from all three families. A block sensitivity can widen this interval but cannot rescue a failed IID lower bound.

For each control, let `L` and `U` be its primary lower and upper ratio bounds. Define `E = max(1, U, 1/L)`, taking the maximum across rank tasks for the rank margin. Freeze:

- Rank improvement: `max(1.10, E_rank²)`.
- Application improvement: `max(1.03, E_application²)`.

The symmetric envelope is unchanged by swapping identical A/B labels. For the same controls, its squared rank margin is at least the old `1 + 2*worst_endpoint_distance` margin. These formulas are conservative decision rules, not a proven bound on future noise or a guarantee of nominal 95% coverage. Short blocks do not rule out longer dependence or changing environmental conditions.

## Frozen study and acceptance

The computation, corpus, workload weights, two authored configurations, layouts, dtypes, search budget, candidate observations and held-out inputs remain unchanged. Fixed selection uses global application costs, per-shape selection uses routed costs and joint selection uses actual `RankPlan` costs. Rank eligibility uses actual-plan rank costs and the new primary interval. A second authored configuration requires the existing 2% development cost advantage over the best subset using at most one.

Three confirmation sessions use separate Python/MLX invocations, fixed seeds and the original untouched sequence orders. Each starts at least 120 seconds after the previous session ends, checked with both Unix and monotonic clocks before reservation or GPU setup. Execution is not scheduled around favorable load readings. Power/load/thermal observations and unavailable telemetry are recorded. Shared OS/driver caches remain. This operational separation does not prove independent environmental draws.

Every session must exceed the frozen rank threshold against both canonical rank algorithms and the frozen application threshold against all four optimized-library query paths. All primary intervals include the IID and both block sensitivities. Pooled results are descriptive and cannot rescue an individual session. Actual resource, numerical, source, fallback and cold-process checks remain required. An empty policy cannot export.

Portfolio claims are separate. Untouched application comparisons must have a lower bound above 1.02 versus best fixed in every session to support that advantage, and above 1.00 versus per-shape in every session to support that advantage. A two-authored-kernel claim additionally requires two selected custom configurations. Passing library acceptance alone does not establish either portfolio claim.

## Bounds and reproducibility

Calibration has one exclusive attempt and a 120-second wall cap. Development retains its 900-second cap; the cold study retains 24 fresh processes within 180 seconds. MLX memory/cache limits remain 2 GiB/256 MiB. The authored sources remain below 12 KiB, with at most two configurations and 128 rules. Compilation, staging, validation, warmup and analysis count toward their applicable stage budget.

The finalized measurement source receives a new checked-in pin. The old rank pin and source remain in immutable V1 archives and Git history; row/fused study pins are unchanged. A fresh executed twelve-method numerical report must match the new full study identity. Earlier reports are not relabeled. The independent exporter reconstructs controls, all interval families, thresholds, selection, individual gates, clock gaps, resources and claim flags from bounded raw records before atomic exclusive installation.

If the single V2 application prerequisite fails, stop this ranking investigation. If development or confirmation loses, preserve the result. Do not adjust criteria, add variants, retry tuning or repeat confirmation after observing candidate results. No purchases, foreign-job stopping or online runtime search is part of this protocol.

See the [measurement source](../scripts/rank_study.py), [independent replay](../src/kerneldarwin/rank_export.py) and [CPU measurement regressions](../tests/test_rank_measurement.py). The current public result remains numerical validity with unavailable performance acceptance until the new evidence is actually executed and verified.
