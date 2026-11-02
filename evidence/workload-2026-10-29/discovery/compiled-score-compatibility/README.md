# Whole-query compilation compatibility

The exact probe executed both existing authored score kernels inside an MLX compiled graph containing finite validation, sorting and gathering. All 88 bounded numerical/layout/application checks passed with Metal API and GPU shader validation enabled. No timings, tuning or candidate selection were performed. The baseline source is public commit 0c312bd66c0e30d015c4f9eda8e5b93dee5ca3ea, fused adapter/kernel SHA e10fbfb5fb613d8d271c87bcc832bd0aca30d4ac12b1f954253f8237f8bd4e10. The script's SHA is in result.json.

Synthetic in-memory rules exercise dispatch only; they are not confirmed artifacts. The DOT files show the exported graph structure for one ordinary case. Seven primitive nodes are not a device-time, copy-count or causal-performance measurement. This establishes compatibility and numerical behavior, not an application speedup. A later source integration needs matching validation and a new frozen performance study.
