# Investigation workbench v0.2

The interface starts with the measured workload, not a marketing headline. A charcoal instrument field, chalk type, steel reference trace, restrained orange candidate signal, and pale verification green connect the brand to its scientific task. The original branching K mark remains. Self-hosted IBM Plex Sans and Mono make typography consistent without remote font calls.

## Reading order

A compact record bar selects the investigation and states the evidence type. The overview pairs the latency fingerprint with its verification ledger; both follow the selected candidate. The distribution and paired-sample modes use actual recorded observations. An accessible range input inspects a specific pair. Reference/candidate medians, quantiles and the reported confidence interval use the same measurement phase. Tiny nonzero errors use scientific notation.

The intervention table preserves failed attempts. Flat search histories use a continuous baseline-to-attempt bus with a single horizontally scrollable row for larger pools. Branched, cyclic or unresolved histories are grouped by their actual parent IDs, with no fabricated graph edges. Implementation panels show original text and unique-line highlighting, explicitly distinguished from a semantic diff. Atlas and research controls retain actual record filtering and held-out budgets.

## Code and motion

`lab/app.js` owns state and navigation. `lab/js/records.js` validates incoming records, `model.js` contains pure numerical/data transforms, `charts.js` draws actual measurements, `lineage.js` preserves parent identity, `overview.js`/`inspection.js` render views, `dom.js` constructs text-safe DOM, and `motion.js` owns interruptible Web Animations. No runtime framework or animation dependency is required.

Navigation changes use a short opacity/position transition while the shell stays stable. Candidate changes reveal the affected evidence. Repeated input cancels prior motion; reduced motion resolves immediately, including preference changes. Values never count up from invented intermediate measurements. No idle pulse implies a live GPU.

## Trust and accessibility

CPU controls, references and CUDA records remain separate. The viewer checks consistency; `kd validate` independently recomputes statistical gates. A bad import cannot replace a valid record or poison another view. Missing profiler evidence stays absent. Large traces use bounded extrema computation and disclose their 300-visible-event cap. No imported code executes.

All icon-only navigation retains accessible names. Native selects, buttons, disclosures and range controls preserve keyboard behavior. Focus survives candidate/record changes. Responsive layouts retain the same content across phone, tablet and desktop. CSS and Web Animations honor reduced motion. Current-source checks and the exact tested scope are recorded in `assets/v0.2/qa.json`; passing checks do not claim universal accessibility conformance.

## Reproduce

The Python package includes every module, font and license. `kd lab` serves it read-only. For source work, `python3 -m http.server 8766 --directory lab` is sufficient. `npm test` runs dependency-free frontend unit tests. Optional browser tests: `npm ci`, `npx playwright install chromium`, `npm run test:browser`. A native Chrome path can be supplied with `PW_EXECUTABLE_PATH` for a local isolated test profile. Browser test dependencies are development-only.
