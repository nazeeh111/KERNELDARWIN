# Run a local workload

**The application runs; optimized export remains unconfirmed.** Row-normalization kernels and fused query-score kernels have executed on an Apple M4 with independent numerical checks. The first frozen performance study failed its application gate. No accepted optimized artifact is bundled. The default consumer uses the complete compiled stable MLX query. [Executed results](workload-results.md).

The application performs local cosine search on caller vectors or deterministic token-count features. It needs no LLM, downloaded model weights, runtime network access or answer cache. Token-count features are a small lexical demonstration, not a pretrained semantic model.

## Install from this checkout

Use Apple Silicon macOS and Python 3.11 or later. The executed environment used Python `3.12.13`. From the current source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[metal]'
kd workload doctor
kd workload demo --query "cosine search vectors"
```

The Metal extra pins `mlx==0.32.0`, `mlx-metal==0.32.0` and `numpy==2.3.5`. `doctor` executes a tiny GPU sum-of-squares check and reports its runtime identity. Inspect `metal_executed`; a correctness probe is not performance confirmation. Other devices or environments have not received the recorded M4 validation.

The planned versioned wheel installation, after v0.3.0 is published and its download verified, is:

```bash
python -m pip install './kerneldarwin-0.3.0-py3-none-any.whl[metal]'
```

[Release files](https://github.com/nazeeh111/KERNELDARWIN/releases) currently retain the earlier releases. This documentation does not claim the v0.3.0 release is already available.

The demo returns document matches, scores, artifact status, dispatch scope, fallback counters and the last decision. Its `compute_ms` is one invocation including cold work, not a benchmark or acceleration claim. Input preparation is reported separately. `artifact_status: library_only` is the expected state without an accepted plan.

Run `kd lab`, then open `http://127.0.0.1:8765/workload.html` for installation, run and guard examples. The [original LAB](../lab/index.html) retains the earlier CPU evidence. The browser displays records and examples; local Python performs the actual computation.

## Use your own vectors

`kd workload search` reads a NumPy `.npz` archive with arrays named `documents` and `queries`. Both must be rank two, share a feature width, and share explicit `float32` or `float16` dtype. The index must be nonempty. Set `k` between one and the document count. Pickled object arrays are disabled; compressed and uncompressed archives are each limited to 256 MiB.

```bash
python - <<'PY'
import numpy as np

rng = np.random.default_rng(27)
documents = rng.standard_normal((64, 384), dtype=np.float32)
queries = documents[[3, 27]] + np.float32(0.01) * rng.standard_normal(
    (2, 384), dtype=np.float32
)
np.savez("vectors.npz", documents=documents, queries=queries)
PY
kd workload search vectors.npz --k 5
```

File loading and input preparation are outside the reported compute stage. Compute includes index refresh, query normalization, scoring, ranking and materializing result IDs/scores on the host. This CLI call performs one search; the research trace separately specifies three changed-query searches per refresh.

## Call it from Python

```python
import mlx.core as mx
from kerneldarwin.fused_search import FusedPlan
from kerneldarwin.search import text_vectors

mx.set_default_device(mx.gpu)
plan = FusedPlan()  # complete compiled stable library query
documents = ["Metal kernel", "cosine vector search", "CPU control"]
index = plan.index(text_vectors(documents))
ids, scores = index.search(text_vectors(["vector search"]), k=2)
print([(documents[i], score) for i, score in zip(ids[0], scores[0])])
print(plan.status, plan.last_decision, plan.calls, plan.fallback_calls)

# Replace the index with changed values; previous answers are not cached.
index.refresh(text_vectors(["Metal source", "local vector index", "test record"]))
```

`index` also accepts MLX arrays directly. `search` returns two nested Python lists: document row IDs and cosine scores. `refresh` normalizes and replaces the stored document vectors. A failed refresh preserves the previous index. Empty query batches return `([], [])`; ties follow MLX sorting rather than a guaranteed ID order.

For a future accepted score export, use `FusedPlan("artifact")` or `--artifact PATH`. `Normalizer("artifact")` with `CosineIndex(vectors, normalize=...)` remains the separate row-normalization API. The CLI selects the authored operation from manifest data; invalid plans keep a recorded library fallback. These APIs do not imply that an accepted artifact exists today.

## Guards and numerical scope

Normalization is row-wise `x / max(norm(x), 1e-12)`. Maximum scaling avoids overflow from directly squaring large finite float32 values. Normalization accumulates in float32 and restores the input dtype. Fused custom scoring accumulates float32 dot products and returns float32 `[query_count, document_count]` scores.

| Condition | Runtime behavior |
| --- | --- |
| No installed score plan | Complete compiled stable MLX query, including normalization, finite check, scoring and ranking |
| Confirmed exact float32 tuple `[queries, documents, width]`, matching source/device/runtime | Named authored fused score candidate |
| Unknown tuple, float16/bfloat16 or non-GPU default device | Complete compiled stable query fallback |
| Missing, malformed, oversized, stale or changed-source plan | Clear custom rules, record rejection reason and use fallback |
| Noncontiguous/broadcast view within a valid custom rule | Read actual device strides; no requested implicit contiguous copy |
| Non-MLX input, non-floating dtype, wrong rank, width mismatch or invalid `k` | Explicit input error |
| Nonfinite index or query in the application | Explicit error; failed refresh retains the previous index |
| Autodiff request | Outside the inference contract; `score`/normalization mode accepts only `inference` |

The lower-level `candidate_scores` investigation helper requires already finite normalized document rows with component magnitudes at most one. It does not rescan them. `FusedCosineIndex.refresh` establishes this precondition through compiled stable normalization and a finite check. Use the application index when working with raw document vectors.

A fused rule permits nonempty float32 inputs with query count at most 128, document count at most 2,048, feature width at most 2,048 and `queries * documents * width <= 16,777,216`. Plans contain at most 128 exact tuple rules and two distinct authored score configurations: `score_simd` and `score_tile4`. Row-normalization candidates have their own matrix bounds and four authored configurations. There is no arbitrary source input or learned runtime policy.

Zero rows remain zero. A NaN/Inf input row produces all-NaN lower-level normalization or fused-score output; the application rejects nonfinite results. Float32 input subnormals follow native Metal flush-to-zero behavior. Tiny normal values below epsilon remain proportional and receive strict zero-absolute-tolerance tests. [Detailed contract and tolerances](workload-semantics.md).

## Offline study and export

[The first preserved study](../evidence/workload-2026-10-29/target1-normalization/) is complete and failed. Do not retune or rerun its confirmation sessions. The second target passed numerical checks but failed its prospective identical-baseline stability gate before any candidate timing or tuning. Its rejected raw pairs are preserved; no retuning or favorable retry is implied.

The following is the implemented source workflow, not a claim that a new study has passed:

1. After an intentional implementation change, update measurement pins with `python scripts/pin_study_source.py`, rebuild the package and use matching source/script versions. Run current-source Metal validation into a fresh report path.
2. Choose a fresh study directory. Run `python scripts/fused_study.py freeze --directory STUDY`. Every identical-baseline 95% total-cost interval must lie within `[0.9, 1.1]` before tuning. Rejection stops the study.
3. Copy the matching executed GPU report into `STUDY/numerical-validation.json`; run `python scripts/fused_study.py tune --directory STUDY`. The search has a 15-minute development budget and at most two authored variants. The same measurements feed fixed, per-shape and joint selection.
4. Run `confirm --session 0`, `1` and `2` against the same directory without retuning. Each independent session must meet the frozen score criterion and an application 95% lower speedup bound above `1.03` against stable and fast complete compiled libraries. Untouched shapes use fallback; every loss remains included.
5. Run `cold-study` for the 18 serialized fresh-process starts. Only after all checks pass, attempt `kd workload export STUDY --output NEW_ARTIFACT`. Export rejects failed confirmation and refuses overwrite.

Use `scripts/workload_study.py` for the preserved first-target protocol, which has eight comparison methods and 24 cold starts. Both scripts support `freeze`, `tune`, `confirm`, `cold` and `cold-study`; stages write immutable evidence. Source, application, baseline, measurement, corpus, numerical report, policy and environment identities are checked. Independent process starts do not delete shared OS/driver caches.

The readable kernels live in [`src/kerneldarwin/metal`](../src/kerneldarwin/metal/). Sources are snapshotted at import so changed files cannot relabel cached pipelines; restart after changing installed code. Manifests are trusted local reproducibility records, not signed third-party attestations. They never provide executable source for runtime compilation outside the authored templates.

See the [custom Metal kernel API](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), [optimized RMS normalization](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rms_norm.html), [results and costs](workload-results.md), and [editable architecture](workload-interface.md).
