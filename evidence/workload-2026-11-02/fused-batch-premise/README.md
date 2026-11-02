# Fused document/query normalization premise

This is a bounded numerical and resource experiment for a new computation, not a latency study or accepted artifact. The new kernel normalizes documents and three query groups in one dispatch, writing the normalized queries directly into a joined output. It checks every input row for nonfinite values. The compiled chain then uses vendor matrix multiplication, one of the two existing stable rankers and a finite-status reduction. No answer, index or input contents persist across calls.

The single registered attempt completed in 15.018 seconds with 139,353,840 bytes peak MLX memory, within 120 seconds and 2 GiB. Apple API and GPU validation were enabled. There were zero dropped events, pending callbacks or restoration failures. Actual installed callees were unchanged from the preceding 28-method validation. The new prototype itself is retained here with exact source hashes; it has not yet been delivered as a guarded installed artifact API.

All 42 recorded numerical check groups passed. Coverage includes seven shapes from width 1 through 2,048, three document layouts, both authored paths and all four libraries; stable zero ties; extreme magnitudes, cancellation, native subnormal behavior and tiny-normal comparisons with zero absolute tolerance; every document/query input position with NaN/Inf; aliasing and recovery; and six unsupported-request fallback cases. The float64 reference computes norms directly, independently of the rescaled GPU formula. These are check groups, not 42 independent test methods or performance sessions.

| Method | Notes dispatches | Manuals dispatches | Archive dispatches | API command buffers per call |
| --- | ---: | ---: | ---: | ---: |
| Compiled stable sort | 27 | 28 | 29 | 1 |
| Compiled fast sort | 29 | 30 | 31 | 1 |
| Compiled stable reduction | 51 | 52 | 53 | 2 |
| Compiled fast reduction | 53 | 54 | 55 | 2 |
| New packed normalization + SIMD32 rank | 6 | 6 | 6 | 1 |
| New packed normalization + group128 rank | 6 | 6 | 6 | 1 |

The 18 resource calls use the original three development profiles and one changed-input seed. The custom chain contains one normalize/pack dispatch, two vendor GEMM dispatches, one rank dispatch and two finite-status dispatches. It has no generic copy-kernel dispatch. The library paths retain three query-join copies; reduction paths also assemble five score and five ID outputs and cast an index array. Input staging and host results still occur. Counts are observed API events, not measured physical memory traffic or application speedups.

An explicit API-return marker now separates submissions made by the callable from the additional command buffer submitted by the diagnostic's final synchronization. Every scope received successful completion callbacks. A positive control checked allocation, a blit of 16 bytes and both named compute-dispatch entry points before application use. The helper observes only its own process and restores its hooks; no system trace, foreign record or binary is distributed. It extends the previous fixed helper only with bounded longer names, ABI-size checking and the API-return marker.

The source portfolio contains one new normalization configuration shared by two reused rank configurations, three authored entry points total. Readable source, including the existing rank key helper, totals 7,294 bytes within 12 KiB. Inference, float32 and exactly three supported nonempty query groups are the custom domain. Unknown layouts, direct MLX requests, lower precision and other group boundaries use the unchanged whole-library fallback. This is a numerical trial interface, not permission to load an unconfirmed artifact into the production dispatcher.

`preregistered.json` binds the producer, prototype, MSL, helper and original helper-binary hash before execution. `result.json` retains numerical outcomes and raw resource events; `native.log` records backend-validation activation; `source-audit.json` checks the 76 unchanged installed runtime files and matching environment after execution. The helper binary remains private. The historical producer expects its original `.work/fused-batch-premise-20261102` location; it is retained for source inspection, not advertised as an installed tracing service.

Run `python evidence/workload-2026-11-02/fused-batch-premise/replay.py --output /tmp/new-fused-premise-audit.json` for the portable independent CPU replay. It verifies the source packet, all 42 check group domains and all 18 resource samples, including exact event counts and API boundaries. It refuses to overwrite an existing output. The first reader draft incorrectly treated reduction-output copies as query-join copies; classification was corrected using raw names and geometry. No native result, producer or acceptance rule changed.

The evidence supports building a complete guarded implementation of this new full chain. It does not establish latency headroom, freeze a performance criterion, tune a policy, expose the reserved holdout, export an accepted artifact or complete the goal. All earlier studies remain closed and immutable. A future application claim needs prospective end-to-end comparison against all four equally batched optimized libraries, matched fixed/per-shape/static selection and untouched independent confirmations.
