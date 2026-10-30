# Workload numerical and dispatch contract

Both operations are inference-only. Inputs are rank-two MLX arrays; host conversion is explicit. Kernels allocate new outputs and never write inputs or their aliases. A package cache stores compiled graphs/templates, never input contents, normalized answers or search results. Numerical validity and measured performance coverage are separate.

## Stable row normalization

The operation is `y[row] = x[row] / max(L2_norm(x[row]), 1e-12)`. Maximum-magnitude scaling before the square-sum prevents finite float32 overflow from directly squaring large values. Custom candidates accumulate float32. The library computes normalization in float32 and restores float16, bfloat16 or float32 input dtype. No precision is selected from benchmark values. Custom compilation uses Metal `math_mode=safe`.

The floor applies to the norm, not additively to its square. Zero rows return zeros. Tiny normal values below the floor remain proportional to the original values. Their tests use zero absolute tolerance, so returning zeros cannot pass. Float32 input subnormals follow observed native Metal flush-to-zero behavior; their output signs are not promised. A NaN or either infinity anywhere in a row makes the entire lower-level output row NaN.

Zero rows or columns produce the corresponding empty matrix through the library. Wrong rank and unsupported dtypes raise. Authored row candidates require nonempty float32, at most 2,048 columns and at most 16,777,216 elements. Unknown shapes, float16/bfloat16, wider rows or changed environments use the compiled MLX fallback. `Normalizer` supports only `mode="inference"`; callers requiring autodiff must use library operations directly.

Four bounded configurations are available for offline investigation: `simd`, `simd4`, `vector4` and `group128`. A production normalization plan may use at most two of them.

## Fused query scoring

The second operation is `scores = stable_l2_rows(raw_queries) @ normalized_documents.T`. Queries have shape `[Q, D]`, documents `[N, D]`, and custom output is a new contiguous float32 `[Q, N]` matrix. Both custom inputs are float32; normalization and dot accumulation use float32 safe math. The same `1e-12` clamped denominator, max-scaling and native subnormal policy apply.

The lower-level `candidate_scores` helper and `FusedPlan.score` require document vectors that are already finite and normalized with component magnitudes at most one. They do not synchronize to rescan that precondition. The application establishes it on every `FusedCosineIndex.refresh` using compiled stable normalization and an executed finite check. Its stored index is replaced only after validation succeeds. Callers supplying raw document vectors should use `FusedPlan.index` rather than the lower-level helper.

A query NaN/Inf yields an all-NaN lower-level score row. A zero query yields zero scores. The application rejects nonfinite query results before returning IDs or values. Empty query batches return `([], [])` after width/dtype/`k` checks; an empty document index is invalid. Direct authored score calls reject empty inputs. The library score helper supports empty matrices, returning the corresponding zero-shaped score matrix.

Candidate bounds are `1 <= Q <= 128`, `1 <= N <= 2048`, `1 <= D <= 2048`, and `Q*N*D <= 16,777,216`. Exact static rules key `[Q, N, D]` and float32. `score_simd` assigns one SIMD group to each query/document pair. `score_tile4` reuses a shared normalized query across four documents, with all threadgroup barriers preceding document-tail branches. Neither changes behavior based on benchmark values.

A `FusedPlan` application fallback runs the entire compiled stable query, including normalization, finite validation, matrix multiplication, ranking and gathering. It does not force a separated normalization/scoring path on an unknown tuple. Refresh remains the compiled stable library. Float16/bfloat16 and mixed float inputs use MLX library operations and their normal dtype behavior; the custom adversarial guarantees are for float32. Lower-precision application cases are separate smoke checks, not the full float32 adversarial study.

## Layouts, tolerance and application results

Both custom operations read actual Metal-side signed strides with `ensure_row_contiguous=False`. Covered layouts include contiguous inputs, row/column slices, transposes, offset views, reversed axes and broadcast views. Python does not infer strides from shape. The original vectorized row candidate uses `packed_float4` to retain float alignment for offset views; scalar paths handle non-unit column strides and irregular tails. The fused candidates use scalar device-stride loads. Broad layout semantics do not imply measured performance coverage for every possible view.

Normal-magnitude float32 numerical checks use independent float64 maximum-scaled references with relative tolerance `2e-5` and absolute tolerance `2e-6`, allowing different parallel reduction orders. Tiny-normal row checks use relative tolerance `2e-5` and zero absolute tolerance; the fused tiny-score test uses relative tolerance `3e-5` and zero absolute tolerance. These tolerances are operation-specific, not a promise of identical bits. Native subnormal handling is applied explicitly to the reference policy.

The application requires finite index/query vectors and validates outputs on the GPU. Such validation and synchronization belong in application cost. `search` returns nested host lists of unique, in-range document IDs and scores. Tests associate each returned score with its independently calculated document score and compare the top-k score set. Equal-score ID order follows MLX and is not guaranteed stable across implementations. Zero vectors are valid and have zero similarity.

Refresh creates a new normalized index. Mutating a source array cannot be treated as a cached-answer key; changed queries and index contents are exercised in repeated calls. No backward/gradient contract is supplied for either authored operation. Lower-level `score` accepts only inference mode.

## Installation, invalidation and resource limits

A confirmed artifact contains a JSON manifest, readable authored Metal sources, exact rules and verification identities. It runs through the pinned installed package and MLX, without the research UI, private tuning data, an LLM or network. There are at most 128 rules and two distinct candidate configurations per plan. Missing or unknown inputs use library fallback, never hidden offline search. MLX may compile an unfamiliar fallback graph as it normally does.

Decisions bind semantic operation, loaded adapter and Metal bytes, application/baseline source, measurement code, compiler options, MLX/MLX Metal, Python version, OS version/build, device name/architecture, GPU core count and memory. The Metal compiler is the bound MLX runtime JIT and OS, not falsely identified as the Xcode command-line compiler. Missing files, malformed/oversized plans, unconfirmed evidence, changed runtime/source or invalid rule metadata clear custom rules and record a fallback reason. Invalid semantic inputs still raise.

Each process snapshots authored source at import. Newly compiled shapes and cached pipelines use those same bytes; editing files cannot attach a new identity to cached old code. Restart to install a change. Artifact source bytes must agree with the installed authored implementation. Independent installation tests use changed inputs without development data or Python/MLX process caches; shared OS/driver caches are not deleted.

These are trusted local measurement records, not signed attestations. A plan supplies no arbitrary executable or source path. The public interface is a static viewer and local consumer, not an arbitrary-code or GPU execution service. A subprocess or worktree is not a hostile-code sandbox.

The current static scoring plan compiles the authored score primitive and ranking/finite validation as one query graph. Score-only plan calls also use a compiled graph. Cached functions capture variant and `k`, never input arrays or results. Outer guards and dispatch counters still execute per request. Compilation does not extend numerical or performance coverage to an unmeasured environment.
