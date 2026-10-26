# Architecture

The Python core has no required runtime dependencies. Optional backends import only when selected.

| Component | Responsibility |
|---|---|
| `records.py` | Typed workload/configuration contracts, validation and atomic immutable JSON storage |
| `statistics.py` | Latency summaries and paired-bootstrap promotion gate |
| `hypotheses.py` | Analytical diagnostic leads, missing evidence and falsifiers |
| `selector.py` | Small ridge/UCB model, saved weights, grouped replay and baselines |
| `control.py` | Actual NumPy CPU experiments for validating software behavior |
| `kernels.py`, `gpu.py` | Bounded authored Triton templates, PyTorch references, correctness checks, CUDA timing and trace extraction |
| `engine.py` | Candidate budget, failed attempts, fresh-input confirmation, worker subprocess |
| `cli.py` | Explicit backend selection, record inspection, training, export and read-only LAB |
| `lab/` | Static, dependency-free evidence viewer, also included in the wheel |

The public viewer has no compute backend. A user imports JSON locally; source strings are rendered as text. Local `kd investigate` invokes only predefined templates through a worker with an environment allowlist and a hard timeout. This limits accidental resource use; it is not a security sandbox for arbitrary generated programs. No LLM API, service account, database or message bus is needed.

Raw records are the stable handoff between CLI, learning and UI. A record's self-reported provenance and structural validation are not cryptographic proof of execution. Promotion validation recomputes gates from raw timings, while execution authenticity still requires inspecting collection logs and the source/environment. Changes in kernel source invalidate export of an earlier candidate.

Kernel source/semantic correctness, numerical tolerances and measured performance are different checks. The current hypothesis engine offers test plans without claiming causal inference. A future controlled experiment layer can attach counter evidence and interventions to these IDs without replacing the record contract.
