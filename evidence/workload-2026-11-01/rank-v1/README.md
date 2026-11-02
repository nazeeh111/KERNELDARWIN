# Stable top-five target: baseline rejected

Production GPU correctness passed. Performance study did not start.

- `numerical-validation.json`: 12 actual production Metal methods, independent installed wheel, no failures/errors/skips. `native-validation.log` records API and GPU validation enabled.
- `installed-provenance.json`: separate post-run audit of the source-free consumer root, installed module paths and equality to the original executed wheel. It preserves the original numerical report and states the later exporter-only installation change.
- `source-frozen.zip` / `source-frozen.json`: exact 42-file source and hash manifest before the attempted prospective freeze. Adapter, application, libraries, MSL, study, tests and package metadata are included. Archive SHA-256: `8dece911a72ded29f1fec53b1fe5db64e5b1b8617775ffa0a12a8a50b20518d1`.
- `freeze-started.json` and `rejected-baseline.json`: exclusive attempt and all raw randomized identical-library pairs. Every task's confidence interval violated the fixed `[0.9,1.1]` gate.

No protocol, development timings, candidate selection, policy, confirmation or cold study was produced. Source pin `841223eda2da3dfa7376221033723e780fcc87bd70950b32d4051049fe618031` is unchanged. This folder must not be retried or overwritten. The separate longer-block library-only diagnostic also failed and is preserved beside this folder. Neither diagnostic establishes candidate speed or an accepted artifact.
