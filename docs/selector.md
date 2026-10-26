# Intervention selection

KERNELDARWIN uses an explicit small ridge regression model with an upper-confidence-style acquisition score. No pretrained model is shipped. `fit()` learns real-valued weights from supplied, recorded observations; JSON inputs are evidence claims and cannot independently attest hardware execution.

Each observation contains `workload`, `candidate`, `reward`, `cost`, `valid`, `group` (source run identifier), and `device_kind` (`cpu` or `cuda`). Recommended reward is natural log of measured baseline/candidate median latency. Cost is measured experiment wall time in seconds. Invalid candidates have an explicit penalty. Utility is `reward - 1*(not valid) - .02*log1p(cost)`. Changing the units changes this penalty, so datasets must use one cost unit consistently.

The versioned feature vector includes log-scaled row count, row width, stride, dtype, four operation-family indicators, tile/batch size, warp count, and configuration interactions. CPU controls use block size as row batch size and one warp as a schema placeholder; this is not GPU tuning. The model fits `A = XᵀX + λI`, `θ = A⁻¹Xᵀy`, with pivoted elimination in the standard library. Ranking uses `θᵀx + α√(xᵀA⁻¹x)`; defaults are λ=1, α=.25. The latter term measures feature leverage, not a calibrated confidence interval. No automatic claim of Bayesian posterior validity is made.

```python
from kerneldarwin.selector import InterventionSelector, evaluate_grouped
model = InterventionSelector().fit(observations)
ranked = model.rank({**spec.to_dict(), "device_kind": "cpu"}, candidates)
model.save("selector.json")
report = evaluate_grouped(observations, seed=0)
```

A model refuses mixed CPU/CUDA training and rejects explicit device-kind mismatch at ranking. Pass `device_kind` when ranking. Persisted models include feature version, weights, regularized design matrix, observation count, device domain, and source group identifiers.

## What evaluation establishes

`evaluate_grouped` replays a finite, observed candidate table using leave-one-family-out and leave-one-shape-out folds. Seeds for the same workload stay together. Each fold trains only on other families/shapes, then ranks that fold's observed interventions. It reports best measured reward and total measured cost within each candidate-count budget, and oracle gap. Repeated observations of the same intervention are averaged inside a holdout workload. There is no imputation of missing candidate outcomes.

Comparisons include 32 seeded random permutations per workload, a declared batching/warp heuristic, the learned selector, a no-context ablation (configuration features only), a no-uncertainty ablation, and a sequential finite-pool Gaussian-process upper-confidence-bound (GP-UCB) Bayesian-optimization baseline. The CPU heuristic tries the largest row batch first. The GPU heuristic tries warp counts nearest four for rows up to 4096 columns and eight above that threshold. These simple heuristics are declared baselines, not asserted best practice.

The learned policies use offline initial rankings. The GP-UCB baseline updates its posterior after every selected candidate outcome. This is not an online bandit regret proof, a calibrated uncertainty study, a causal bottleneck detector, standard Triton autotuning, a BoTorch execution, or a replication of KernelAgent/Kernel Forge. It neither trains nor validates cross-device transfer. Genuine CUDA observations and hardware holdouts remain necessary for the GPU research question. CPU controls validate learning, serialization and evaluation plumbing only. Negative results must remain in the report.


## Bayesian baseline and information boundaries

`bayesian_order(candidates, observe, seed=0)` implements a compact, sequential GP-UCB baseline using only the standard library. It starts from one seeded random configuration. After selecting a candidate, it calls `observe(candidate)` to reveal that candidate's recorded reward. It fits a zero-mean, unit-variance Gaussian process with an RBF covariance kernel over `log2(block_size)/16` and `log2(num_warps)/5`, then chooses the remaining candidate maximizing posterior mean plus twice posterior standard deviation. The kernel length scale is fixed at .25; observation noise variance is fixed at 1e-4. The posterior is solved directly with the same pivoted small-system solver used by the selector. Hyperparameters are declared in advance and are not fitted on held-out outcomes.

Grouped replay runs 32 seeded initializations per workload and averages best reward and measured cost at each candidate-count budget, matching the number of seeded random permutations. Unselected rewards are hidden from each sequential GP policy until selection; the complete recorded table is used by the evaluator afterward to score each prefix and the oracle gap. The GP learns within a workload, while the contextual selector learns across training workloads. Their different information paths are part of the comparison and are retained in the report. This baseline does not invoke BoTorch, reproduce all practical Bayesian-optimization choices, estimate heteroscedastic timing noise, or launch additional CPU/GPU work. CPU replay remains CPU-only evidence.
