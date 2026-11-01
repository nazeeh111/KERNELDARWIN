# Prospective ranking study V2

V2 was defined and independently reviewed before any ranking candidate timing or selection. It materially changes the rank-only entry prerequisite to a complete-application prerequisite, with conservative squared symmetric noise margins and IID/block4/block8 envelopes. This does not reinterpret any V1 failure. The full fixed design is in docs/workload-ranking-v2.md and scripts/rank_study.py.

Current prerequisites: 147 CPU-suite methods passed with 29 explicit Metal skips; a fresh independent installed wheel passed all 12 actual production Metal methods with API and GPU validation enabled. numerical-validation.json records actual installed module paths and the absence of a source tree in its validation root. It binds full study identity 99810abba1515ae17004937f84c99edd523cfb6ecc125b8ba179b232b0136a4f and current measurement pin 4470b506443feea5cf6a4f22b5fea01338e8a199621b02dd0f45f20f114f5b3a. No old numerical report was relabeled.

The single calibration and any candidate study are pending. No accepted artifact or acceleration exists. Do not start candidate timing unless the finalized source-bound application prerequisite passes; preserve any failure, with no retry or later criterion changes.
