# Current validation status

## Workload source preview, v0.3.0 unreleased

The current local Python suite passes 63 CPU-compatible tests, with 16 Metal methods explicitly skipped in that CPU invocation. The actual M4 Metal runs separately passed 18 row/application/guard methods (nine GPU and nine CPU guards) and seven fused-score GPU methods, with native API and GPU shader validation enabled. [Executed reports and limits](workload-results.md). The merged frontend suite passes 22 tests. Final merged browser/package/remote verification is recorded separately when complete.

Actual Metal computation does not establish acceleration. The original frozen row study failed its application criterion; the second target failed its prospective baseline-stability gate before tuning. No accepted optimized artifact or successful Workload release exists. The default application uses the complete compiled stable MLX query. CUDA execution, hardware counters, cross-device results and learned GPU superiority remain unverified.

The remaining sections preserve the earlier v0.2.1 validation snapshot and its CUDA research scope. Their historical statements do not override the current Metal reports above.

## Archived v0.2.1 research preview snapshot

## Established on this host

- Apple M4 arm64 macOS development host; Python 3.12.14 for local verification.
- 22 CPU tests pass, including actual NumPy correctness checks across all four families, float32/float16 and strided inputs; skipped-computation rejection, atomic persistence, tampered promotion rejection, learned selection and grouped replay.
- 24 actual CPU-control records and 72 measured interventions. Promotion gates are recomputed from raw samples; source hashes, fitted weights, deterministic evaluation and LAB copies are checked by `scripts/check_evidence.py`.
- A genuinely fitted CPU selector and family/shape-held-out replay against random, fixed rules, sequential GP-UCB, context ablation and uncertainty ablation. Learned ranking matched the heuristic; no added learning benefit was established.
- Independent source review found promotion, source-pinning, malformed-import, trace-storage and packaging issues; those issues were corrected and rechecked. No unresolved consequential defect was found in that reviewed snapshot. This is a bounded review, not a guarantee of correctness.
- Static LAB native Chrome checks cover eight views at desktop/mobile sizes, explicit missing-evidence states, invalid-import preservation, keyboard focus, export, atlas filtering and actual evaluation controls. Current-source screenshots and a recorded CPU demo accompany the release.

## Not established

This host has no NVIDIA GPU, PyTorch CUDA, Triton, Nsight Compute or CUDA Compute Sanitizer. No CUDA compilation/execution, GPU numerical validation, GPU timing, sanitizer pass, GPU-trained policy, measured hardware counters, causal GPU diagnosis, cross-device result or application-level GPU improvement is claimed. The attention example is an integration path requiring a genuine accepted CUDA export; it has not run here.

The static LAB reads self-reported records. Its consistency checks and Python bootstrap recomputation do not authenticate the hardware origin of arbitrary imported JSON. Inspect trusted collection provenance and source identities. CPU controls cannot be relabeled as CUDA evidence.

## Exact remaining GPU requirement

A trusted Linux machine with NVIDIA compute capability >=8.0 and compatible driver/PyTorch/Triton is needed. Run the CUDA investigation, independent correctness and sanitizer procedures in [usage](usage.md), collect real traces/counters separately, reproduce eager/compiled/autotuned comparisons, train/evaluate on whole held-out workloads and hardware when available, and run the end-to-end attention benchmark. Preserve failures and null results. Use only authorized available compute; this repository does not provision paid resources.

The research hypothesis requires a substantially richer measured CUDA corpus and matched comparisons. Profiler-feature and diagnosis-conditioned ablations are not implemented or evaluated by the CPU preview. The original full GPU research objective remains unmet without this evidence.

## Presentation limit

An original social preview image is included at `docs/assets/social-preview.png`. GitHub's UI exposes the upload control, but its file picker could not be completed because Codex's native-app safety policy rejects control of `com.openai.codex`. No alternate route was used to bypass that control; the repository's custom social image remains unset.


## v0.2 workbench verification

The redesign adds 16 pure frontend tests and 7 reproducible Playwright browser regressions alongside the 22 Python tests. Native Chrome checks cover all eight views across five desktop/tablet/phone widths, plus the independent review's four-record/route matrix including 320 px phones. Selected text contrast was checked programmatically across overview, hypotheses, evolution, source and research views. This is scoped evidence, not a universal accessibility guarantee.

Concrete checks cover matched timing-phase display, candidate and select focus preservation, malformed dormant-view imports, reduced-motion resolution, rapid navigation, atlas/evaluation controls, a 150,000-event synthetic trace, and non-flat parent histories. Synthetic records are test fixtures only, never published benchmark evidence. The 24 recorded CPU experiments and four bundled records remain byte-identical. `assets/v0.2/qa.json` records the source identities and check scope.

The recorded UI clip is silent, labeled CPU-control playback, and derived from actual browser interactions. Full media decode and representative paired/evolution frames were inspected. It demonstrates functional motion; sampled images and automated state checks are not full normal-speed perceptual approval on every platform.


## v0.2.1 evidence comparison verification

The current hosted suites pass 27 Python tests (Python 3.11, 3.12 and 3.13), 22 pure frontend tests and 16 Chromium browser regressions. These are 65 logical tests; repeating the Python suite across versions does not add unique cases. The byte-for-byte CPU evidence checker also passes. Browser checks include all eight routes at 1440, 1024, 768, 390 and 320 px, plus current-source desktop/mobile/source/command previews.

New regressions cover candidate URL/back/reload continuity, shared local-record imports, exact raw CSV values, source text injection, bounded oversized sources, edits beyond the first 1,600 preview rows, command search focus/Escape, disclosure focus, reduced motion and interrupted navigation. Independent source review found two continuity defects; both were fixed and rechecked. The first hosted run also caught an Escape-key focus defect; that correction passes. These are scoped checks, not a guarantee of perfection or universal accessibility.

Current visual artifacts and source identities are in `assets/v0.2.1/qa.json`. Hosted browser capture avoided interfering with a separate local frozen Metal timing study. The v0.2 video is retained and explicitly labeled as historical footage; this refinement adds no new GPU measurement. Public deployment is separately verified against source identities with `scripts/check_live_lab.mjs`.
