# Host-input batch transaction study prerequisites

Status: CLOSED at the fixed calibration wall cap, before authored candidate performance timing. No accepted artifact or acceleration claim. The [prospective methods](../../../docs/workload-transaction-study.md) bind a new operation, source, workload sequence, held-out task and conservative all-session acceptance rule. Earlier V2 remains closed.

The single calibration recorded 181.377 seconds, 19,008,024 bytes peak MLX memory, all 120 rank groups and only 14 of 40 application groups. The 180-second check runs between complete groups; this one finished its last group after the cap and rejected immediately. An external 195-second watchdog did not fire. No protocol/criterion or candidate policy was frozen, and no tuning, confirmation, cold study or held-out execution followed. The application equality gate was not reached. No retry or budget/window adjustment is permitted for this study.

`rejected-baseline.json` retains all raw completed windows, orders, source identity and before/after telemetry. Load averages changed from about 8.04 to 12.59 during this run; battery power was observed and the system reported no recorded thermal warning. These observations do not establish a temperature or explain causality. The independent CPU `calibration-audit.json` reproduced all 21 complete rank comparisons exactly and validated partial application windows. A source-free installed reader produced the same report. `export-refusal.json` records the actual installed exporter refusing the incomplete study with its output absent. `post-calibration-audit.json` checks all 72 preregistered files and four runtime binaries before outcome documentation changed. These checks preserve provenance; they do not convert an incomplete study into accepted GPU performance evidence.

- numerical-validation.json / native-validation.log: actual fresh installed28-method run,14nativeMetal,API/GPUvalidation activated,0errors/failures/skips.
- consumer.json: installed public host-input API,8changed-input top matches,library default.
- cli-first.json / cli-changed-fallback.json: installed CLI,8changed-input top matches and explicit missing-artifact fallback. Single-call timings include cold work and are not a comparative claim.
- installed-prerequisites.json: executed wheel,76runtime byte comparisons,7actual installed module paths,4runtime binary comparisons. No source tree in validation root. No development corpus or tuning cache needed by these consumers.
- regression/: exact failed native run and old adapter source, preserved before the correction.
- preregistered-source.zip/json: exact prospective core, tests, methods and numerical evidence before calibration. CPU manufactured fixture timings are guard tests only, never GPU evidence.

Replay the retained failure with `python scripts/replay_transaction_calibration.py evidence/workload-2026-11-02/transaction-v1 --output /tmp/new-batch-audit.json`. This runs no GPU timing and refuses to overwrite an existing output.

No system trace, foreign process record, rawXML or private native helper binary is distributed. Fresh-agent review was unavailable due subscription limits; separately implemented replay and independent high-precision numerical checks are available, but no fresh-agent source review is claimed.
