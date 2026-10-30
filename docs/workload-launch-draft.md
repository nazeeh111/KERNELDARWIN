# Workload numerical preview: launch draft

Draft copy only. This file has not been posted or sent. v0.3 publication and release verification remain separate delivery work.

KERNELDARWIN Workload adds a local cosine-search consumer, readable Apple Silicon Metal kernels and source/device-bound static dispatch with an MLX fallback.

The row-normalization suite completed 18 current numerical/guard tests on an Apple M4. A separate eight-method current suite exercised fused query normalization and similarity scoring with Metal API and GPU validation enabled. An independent installed consumer changed inputs and passed 15 method checks across the library paths and two direct authored score candidates. Its installed plan remained library-only; direct candidate execution is not an accepted optimized artifact.

The performance result is deliberately negative. The first frozen study completed three confirmation sessions but failed its complete-application improvement gate. The second target passed numerical checks, then stopped before tuning because identical-baseline timings were unstable. No accepted optimized dispatch artifact or speedup is claimed. Raw losses, startup costs, memory and source identities remain with the code.

Start with `kd workload doctor` and `kd workload demo`, or bring document/query vectors in a local NPZ archive. Without an accepted artifact, search uses the complete compiled stable MLX query. No model weights, runtime network access or online tuning are required. The original CPU investigation workbench and earlier releases remain preserved.

Read the [quickstart](workload.md), [executed results and limits](workload-results.md), and [editable architecture](workload-interface.md). Repository and earlier release files: [KERNELDARWIN on GitHub](https://github.com/nazeeh111/KERNELDARWIN).
