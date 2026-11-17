# Keep the gate, and keep the negative result

This reader binds two actual recorded CPU investigations to current source `e4d490c87f3e07f40bdee0bd19c58a346be87dac`, version 0.2.1. **No new timing sample, GPU operation or selector training was performed.** The original 24-workload/72-intervention dataset, four no-improvement cases, saved CPU selector and its negative learned-versus-heuristic evaluation remain included and unchanged.

Extract `kerneldarwin-cpu-reader.zip` into an empty directory and run:

```sh
python3 -B reopen.py
```

The standard-library route verifies every declared member, validates all 24 retained records against the exact original source snapshot, checks the CPU-function source identity and replays all seven selected paired-bootstrap comparisons from 280 original pairs. It neither reruns a timer nor fits a selector, and it does not attest who collected the original measurements. The source snapshot is the original KernelDarwin implementation, not an independent validator.

## Read the two phases correctly

The softmax float32 workload is 128×1024, seed 2028, stride 1. Discovery tests row batches 32,128,512. All clear their recorded fixed-candidate timing gate, but only batch 128 is selected and measured again on fresh inputs with confirmation seed 9029. Its confirmed ratio of medians is 1.7702413383, paired 95% bootstrap interval 1.6577075664–1.8448007119. An unselected candidate is still labeled not promoted, even when its discovery interval exceeds 1.03. The fixed-candidate interval is not a multiple-search guarantee.

The sum float32 workload is 32×31, seed 2026. All three candidates pass their numerical tests; median ratios are 0.8720,0.8831 and0.8846. Their intervals do not clear 1.03, no candidate is promoted, and no confirmation is invented. Reference float64 intermediates are compared with float32-intermediate candidates within declared tolerances. This is an implementation/precision-contract CPU comparison, not GPU optimization or an arbitrary-algorithm claim.

The [wide figure](gates.html), [narrow figure](gates-mobile.html), [figure data](figure-data.json) and editable [figure.py](figure.py) show all 280 per-pair ratios, with no excluded outlier. Small jitter is categorical only. The large mark is the ratio of medians; the small dots are individual baseline/candidate ratios, which are a different statistic. All seven intervals share one linear 0.5–4.0 axis, and 40 observations remain in every row. Discovery and confirmation use separate rows and marker shapes. PNGs are actual Chrome renders of the SVGs, with portable Arial text.

## Use the actual LAB

Follow the repository installation, then `kd lab`. In a source checkout, `kd lab --directory /absolute/path/to/lab --port 19002` is also supported. The public LAB reads recorded JSON; importing a record never compiles or executes its embedded source.

Select softmax 128×1024. Click batch 32 in the shared comparison to see **Measured, but not promoted** and discovery samples. Select batch 128 to inspect independent confirmation. Use **Paired samples**, inspect pair 13, and download **Raw samples**. The actual unrounded [discovery.csv](discovery.csv) and [confirmation.csv](confirmation.csv) exports exactly match their respective 40 pairs. Neither CSV is a new measurement.

The actual [desktop capture](workbench-desktop.png), [phone capture](workbench-mobile.png) and [sum negative capture](no-improvement-desktop.png) come from current source. Command search supports keyboard selection; Escape cancels the share/search dialog and restores focus. A malformed import preserves the current record. [ui-record.json](ui-record.json) is the actual LAB JSON export, byte-content-equivalent to the retained softmax record after JSON parsing; import it in a fresh browser to reopen the same investigation. A shared link identifies a record/selection, not uploaded record contents.

## Source, numerical checks and refusal

[checks.json](checks.json) records exact replay of all selected timing fields, all 24 CPU source identities and a bounded current-engine **numerical-only** softmax check using NumPy 2.3.5. Its three row-batch implementations agreed with the higher-precision reference within 2e−4, maximum absolute discrepancy3.7252902985e−9, while leaving inputs unchanged. No timing code ran for that check.

The actual CLI refuses to export an optimized CUDA callable from this accepted CPU record, exits 1 and creates no file. A successful CPU-control gate is never a CUDA admission. Existing mocked export-preservation tests do not establish live CUDA execution. No sanitizer, GPU counters, roofline, CUDA speedup or GPU-trained policy is present.

All raw historical data is under `retained/`; the exact package source is under `reader-source/`; the complete Apache-2.0 [LICENSE](LICENSE) remains with it. The stored selector evaluation reports no added learned-policy benefit over its heuristic in this small three-choice pool. Those results were not retrained, replaced or reframed as positive findings.

## Reproduce versus replay

`reopen.py` is a data/source/statistics replay. Running `kd control` yourself collects new machine-specific timings, which need their own software/hardware/load context and cannot be compared as an automatic regression. The original CPU records identify the device as arm64 (processor arm), Python 3.12.14, NumPy 2.3.5 and 26 October 2026. The repository README separately describes its development host as Apple M4; that model is not recorded in these JSON records or independently attested by this replay. This presentation was checked using Python 3.12.13. Timing values and recorded environment remain original. No general performance conclusion follows from a software-control rerun.

The source's CI contract includes CPU tests, Node browser-model tests, a CPU control, historical evidence/model recomputation, a byte-checked LAB wheel, source distribution and separate browser regressions. This pass ran selected non-timing/non-training tests and package checks, not the full timing/model-training CI route. The current code has no separate current-delivery manifest; `docs/demo/source-manifest.json` is historical film provenance and was preserved. The new presentation namespace is explicitly included in MANIFEST.in so source archives retain the complete reader, raw files and generator. No package/release or public deployment is implied.
