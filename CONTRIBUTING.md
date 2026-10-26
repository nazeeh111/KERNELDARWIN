# Contributing

Start with a small reproducible issue or change. The useful unit of evidence is a workload specification and complete run record, including failures and environment. Never attach credentials, private tensors or confidential traces.

```bash
python -m pip install '.[control]'
python -m unittest discover -s tests -v
python scripts/check_evidence.py
```

Changes to semantics require reference checks across seeds, boundary shapes, strides and dtypes. Changes to timing or promotion need tests that fail on noise, regressions and invalid candidates. Keep CPU controls, synthetic unit fixtures and genuine CUDA runs visibly separate. Preserve raw samples and no-improvement cases. Do not claim a GPU pass from CPU CI.

The static LAB uses HTML, CSS and JavaScript with no build step. Inspect desktop/mobile, keyboard navigation, empty states and malformed-import preservation after UI changes. Numerical charts must derive from real data.

A GPU contribution should include compatible NVIDIA hardware/software details, eager and compiled baselines, fresh correctness/performance confirmation, relevant sanitizer logs and one practical invocation. GPU CI is manually triggered on a trusted self-hosted runner; never execute contributor code automatically on a privileged runner.

Please cite relevant prior systems and keep required licenses/notices. Performance search, profilers and learned ranking have substantial prior art. Negative experimental results are welcome.
