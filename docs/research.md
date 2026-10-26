# Research report: diagnosis before intervention

**Research preview. No CUDA performance claim is established by this release's development host.** The Apple M4 host cannot execute this project's NVIDIA CUDA path. CPU controls and tests establish software behavior in their stated scope. They do not establish a GPU speedup, a GPU-trained policy, a causal GPU diagnosis, or cross-device generalization.

## Question and intended contribution

Can an explicit performance-hypothesis ledger help select useful interventions using fewer expensive GPU experiments than random search, fixed heuristics, standard Triton autotuning or Bayesian optimization?

The proposed contribution is the connection between three artifacts: a falsifiable explanation, an intervention chosen to test it, and a reproducible correctness/performance decision. Each experiment should make a diagnosis more useful, not merely find another fast parameter value. A successful research result would show that this connection improves decisions under a fixed budget on held-out tasks. This is an empirical question, not a novelty or superiority claim.

Profiling, kernel generation, learned cost models, counter-guided search and graphical search histories all have strong precedents. [The prior-art review](prior-art.md) identifies these overlaps. The preview uses bounded kernel templates so that algorithm changes, precision changes and search strategy can be assessed separately. It does not promise arbitrary-program optimization.

## Competing mechanisms

Three practical routes were considered:

| Route | Mechanism | Decisive limitation or test |
|---|---|---|
| Profile-guided free-form agent search | Use model reasoning and compiler/runtime feedback to generate new programs. | Larger semantic and isolation burden; already demonstrated by prior systems. A model-provider dependency is unnecessary for the first evidence loop. |
| Exhaustive or Bayesian template search | Search a known configuration space and verify each candidate. | Strong conventional route. It may match a learned selector on this small space; compare equal budgets and include fitting overhead. |
| Evidence-conditioned intervention selection | Learn which controlled tests are informative/useful for a workload, preserving rejected explanations and failures. | Chosen research direction. It succeeds only if evidence improves held-out selection beyond ordinary workload/configuration features. |

The simplest useful first model is regularized linear regression with an uncertainty bonus for unexplored contexts. Regularization discourages unstable coefficients when data are sparse. An upper-confidence score balances predicted value and exploration; it is a ranking heuristic, not a calibrated probability that a bottleneck is causal. This is easier to audit and ablate than a large reinforcement-learning system. The exact implemented features and serialization are described in [selector documentation](selector.md).

## Experimental unit and evidence

A task is a complete workload specification, hardware/software environment and reference semantic contract. A candidate is a distinct kernel configuration or named implementation intervention. Observations must preserve task identity, source/configuration identity, valid/invalid outcome, paired timing samples, numerical checks and experiment cost. Invalid candidates consume budget and remain in the history.

A diagnosis entry includes supporting observations, unavailable measurements, an experiment and a falsifier. For example, a launch-overhead explanation should be tested using a size sweep or an explicit launch/graph intervention while holding semantics and timing mode fixed. A memory-bandwidth explanation needs measured traffic/bandwidth evidence or a documented analytical bound plus a controlled access-pattern intervention. A faster configuration alone cannot establish register pressure or occupancy as the cause.

The preview's heuristic explanations are candidate explanations. Unless a controlled experiment distinguishes alternatives, preserve them as unconfirmed. Confidence labels are evidence summaries, not trained causal probabilities. Roofline plots require defensible FLOP/byte counts and appropriate device ceilings; unavailable quantities remain absent rather than being guessed.

## Evaluation plan and current limits

The table separates planned comparisons from the software's offline evaluation. An evaluation function existing in source does not mean that a scientifically meaningful GPU dataset has been evaluated.

| Comparison | Controlled requirement | CUDA evidence in this preview |
|---|---|---|
| Random selection | Same candidate pool, many fixed seeds, equal measured budget | Not collected on the development host |
| Rule-based heuristic | Frozen rules before test collection; equal pool and budget | Not collected on the development host |
| Learned ranking and uncertainty ablation | Train only on designated training groups; freeze model before test | GPU-trained evaluation pending |
| Profiling-evidence ablation | Same model/features except measured evidence; no post-outcome leakage | Pending measured counters and intervention data |
| Hypothesis-testing ablation | Same candidates/budget with diagnosis-conditioned scheduling disabled | Pending a matched controlled study |
| Standard Triton autotuning | Invoke its real tuning path using equivalent configs and timing semantics | Pending CUDA execution |
| Bayesian optimization | Same domain/budget, count fit cost | Implemented sequential RBF GP-UCB CPU replay; actual CUDA/BoTorch comparison pending |
| Existing agent optimization | Pin system, model/provider, tokens, hardware and spend; preserve failures | Not reproduced; no paid model/compute is assumed |

Split by whole task groups. Hold out operation families to assess transfer, unseen shape regimes within known families to assess shape generalization, and whole hardware configurations when multiple GPUs exist. All observations of a task, including repeats and failed variants, stay in the same fold. Hardware holdout requires more than one hardware configuration; otherwise report it unavailable. Choose hyperparameters using training/validation groups only.

For each test task, report best independently verified latency after each experiment budget, experiments and wall time to first accepted improvement, invalid fraction, regret relative to the measured candidate pool, and diagnostic evidence coverage. Aggregate tasks with task-level uncertainty; repeated timing samples are not independent workloads. Include both no-improvement cases and losses. Count data-collection, compilation, fitting and profiling costs separately. Do not train on the test-task outcome and call the resulting replay prospective performance.

The shipped grouped replay is a software-level evaluator on recorded observations. It does not reconstruct temporal hardware noise, guarantee causal effects, or substitute for a prospective optimizer run. CPU controls can reveal bugs and exercise training/evaluation. External historical CUDA data, such as the CC BY 4.0 KTT dataset, could support a separately attributed replay study, but its operations, hardware and candidate semantics differ from the initial Triton scope. That route has not been adopted as current-device validation. The official dataset page and anonymous folder metadata were accessible during review, but a subsequent raw-file metadata request returned HTTP 403. No dataset bytes, model or historical training results are bundled from that route.

## Conditions for a publishable result

Freeze the workload suite and budgets, collect a training corpus on real NVIDIA hardware, then measure held-out results independently. Verify accepted candidates on new seeds and relevant boundary/stride/dtype cases, rerun paired performance confirmation, and run supported sanitizer tools. Measure one integrated application path, including dispatch costs, rather than only isolated kernels. Publish raw records and exact environment/source identities.

If diagnosis-conditioned selection does not beat simpler baselines, report the null result. The investigation record, failure archive and reproducible verification workflow remain useful without a positive ML result. A single fast kernel, a passing CPU suite, or an attractive interface is insufficient evidence for the research hypothesis.
