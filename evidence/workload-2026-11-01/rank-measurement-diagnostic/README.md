# Baseline measurement diagnostic before V2

This preserves a single instrumented library-only probe executed against public source a57fe7e and measurement script 841223, before the V2 amendment. It is not a candidate benchmark, stability test or acceptance result. The original numerical/performance outcomes are unchanged.

`probe.py`, `started.json`, `result.json` and `native.log` are the exact executed bytes and outputs. The probe took 1.3567 seconds and peaked at 148,940,920 MLX bytes. All 64 measured cycles forced 128 distinct rank inputs and 256 output arrays. No garbage collection was observed inside them; collections did occur during staging/warmup and explicit between-task cleanup. This does not establish what happened in earlier uninstrumented attempts.

Current-thread CPU time accounted for approximately 62.0% and 71.6% of aggregate instrumented wall across notes/archive. Execution-wait spans included backend coordination and occasional long waits. They are not pure GPU elapsed time. Wall minus thread time also includes scheduling and other threads. The probe did not establish a cause of instability.

The script retains historical local paths and its original source-commit label. It imported the original study source, whose exact841223 bytes remain in the adjacent rank-v1/source-frozen.zip. Do not rerun this historical script against a changed checkout and treat it as the same experiment. In started.json, source_sha256 identifies the probe script; result.json separately records script_sha256 and the rank runtime source_sha256.

`order_analysis.py` and `order-analysis.json` provide reproducible CPU-only descriptive summaries of the retained A/B pairs. They show no consistent first-versus-second explanation across the tasks/attempts; they do not prove an order effect is absent. This analysis does not reinterpret the original failed gates.

The subsequent application-primary V2 protocol is a prospective amendment justified by the end-to-end application objective, not a claimed causal fix for GC or ordering. No candidate performance had been observed when it was defined.
