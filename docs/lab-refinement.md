# The workbench as an investigation instrument

## Design question

Can a reader determine what changed, which evidence supports it, and what remains unproven without visiting eight separate screens?

The previous LAB already showed raw samples and a verification ledger. Its main weaknesses were fragmented comparison, an oversized speedup without an immediate interpretation, transient candidate selections, and source highlighting based on line membership rather than sequence.

## Primary research and transfer

Inspected 2026-10-29:

- [NVIDIA Nsight Compute, Baselines and Source Comparison](https://docs.nvidia.com/nsight-compute/NsightCompute/): retains a reference while inspecting metric and source differences. Transfer: put every candidate's reported ratio and uncertainty on the same reference axis. Keep its matched reference samples and discovery/confirmation phase explicit. This is a comparison of reported records, not independent verification by the browser.
- [Perfetto UI, Track Event Selections and Commands](https://perfetto.dev/docs/visualization/perfetto-ui): event selection reveals a linked detail pane; searchable commands reduce navigation cost. Transfer: make a candidate selection update the chart, numerical verdict and source consistently; preserve it in the URL, and offer searchable native controls.
- [W3C H102, native dialog](https://www.w3.org/WAI/WCAG22/Techniques/html/H102): browser-managed modality, focus containment, Escape and focus return. Transfer: use an HTML dialog for command search, rather than implementing a custom focus trap. Test actual keyboard behavior. This does not constitute universal assistive-technology certification.

## Decisions and alternatives

Keep the existing restrained scientific identity. More animated panels or simulated GPU telemetry would add attention cost without evidence. A forest-style interval comparison earns its space because a single large speedup hides unsuccessful attempts and measurement uncertainty. The 1× axis is a visual reference, not the application's promotion threshold; the actual recorded gate remains separately visible.

Display discovery and independent confirmation as distinct phases. Do not pool them or claim overlapping intervals prove candidates equivalent. Preserve record order rather than visually implying a statistically established ranking. A failed numerical check remains visible even if its timing is fast.

Use a bounded longest-common-subsequence line diff, with explicit insertion/deletion markers and original line numbers. For oversized input, use a valid replacement diff and disclose that the minimal alignment was skipped. A line diff establishes textual change, not semantic equivalence, numerical correctness or causality.

Deep links identify bundled records and candidates. Imported records stay local; a shared link does not transmit or persist their contents. State that limitation when copying a link. Raw timing CSVs contain numerical observations and phase labels, without executing imported source.

## Verification

Pure tests cover statistical phase selection, interval domains, exact diff reconstruction, repeated/blank lines, bounded fallback and route parsing. Browser checks cover linked candidate state, browser back/reload, command dialog keyboard operation, source navigation, CSV content and responsive containment. Hosted Linux Chromium captures fresh wide/narrow previews to avoid interfering with an unrelated frozen local Metal timing study. Existing CPU evidence remains unchanged.
