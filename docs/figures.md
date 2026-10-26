# Figure provenance

`assets/workflow.svg` and `assets/cpu-selection.svg` are original editable vector figures. Regenerate with `python scripts/render_figures.py`.

Workflow elements map to `gpu.py` profiling/reference adapters, `hypotheses.py` diagnostic leads, `engine.py` candidate scheduling/confirmation, `statistics.py` bootstrap checks and `selector.py` offline replay. The explicit research-open line prevents an implemented stage diagram from implying validated causal or GPU learning outcomes. The loop denotes retained experiment feedback; online causal-confidence updating is not implemented.

Numerical bars come from `evidence/cpu-2026-10-26/cpu-selector.evaluation.json`, family split, candidate budget 1. They show mean best discovery log-speedup, not aggregate final GPU speedup. The title states the negative comparison against the fixed heuristic. SVG IDs preserve panel/bar editability.

PaperVizAgent was investigated and credited in prior art. Its model-service configuration is unnecessary for these deterministic diagrams and would conflict with the no-billed-API constraint. No PaperVizAgent-generated output or code is included.
