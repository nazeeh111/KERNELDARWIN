# Owned-process baseline shader diagnostic

This is an instrumented diagnostic of public source `65c090f994af8e37ad53ff6e14eb9d2bc4c90043`, before the ranking implementation. It contains no candidate timing or accepted performance claim.

`owned-shader-rows.json` contains 1,304 records filtered to the launched Python process only. `summary.json` records interval unions, source/runtime hashes, approximate phase boundaries and explicit limits. `application.json` records 72 independent float64 score/ID checks. `profile_baselines.py`, `sanitize.py` and `provenance.json` describe the actual capture and extraction. The scripts retain historical local paths; raw trace/XML are intentionally private because they include unrelated system context. Their hashes bind provenance but are not cryptographic hardware attestation.

Overlap/nesting is unioned, shader record counts are not assumed to be launch counts, and per-shader unions cannot be added into application percentages. Host markers include validation/input work; 2-ms interior guards omit boundary records. Sort/gather presence motivates an algorithmic investigation but does not prove a causal bottleneck. Global numeric counters could not be attributed by the shared accelerator identifier. No application bandwidth, register-spill or numeric-counter conclusion is supported.

## Resource and submission accounting from the same capture

`account_resources.py` extracts four additional recorded tables without starting another capture or benchmark. `owned-resource-accounting-rows.json` retains only exact-PID rows and excludes allocation backtraces. `resource-accounting.json` binds the parsed raw bytes, application and extractor, reconciles command-buffer encoder counts and records remaining unavailable measurements.

The capture contains 181 owned command-buffer submissions, including 13 reporting zero encoders. Its 168 unique encoder records match the submission counts exactly; their recorded label is `Compute Command 0`. There are 86 unique buffer-allocation events totaling 19,038,208 cumulative resource bytes. The 122 `MTLDevice.currentAllocatedSize` records reach an observed maximum of 14,385,152 bytes. These are whole instrumented-process observations, including setup, compilation, input preparation and numerical checks. They are not per-search rates.

Cumulative resource creation sizes are neither live memory nor transferred bytes. Metal allocation levels and MLX peak allocation have different scopes; neither measures physical GPU residency. Recorded encoder labels do not prove that all copy activity would appear in a separate blit encoder. Exact kernel launches, implicit contiguous copies, tensor materializations, transfer bytes and pure GPU time per application call remain unavailable. Missing event durations remain null. This improves the available accounting but does not complete the goal's accounting or performance requirements.
