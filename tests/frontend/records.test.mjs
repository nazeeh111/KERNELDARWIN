import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { validateRecord } from "../../lab/js/records.js";
const fixture = () =>
  JSON.parse(
    fs.readFileSync(
      new URL("../../lab/data/control-softmax.json", import.meta.url),
    ),
  );

test("all authentic shipped records retain their original contract", () => {
  for (const name of [
    "softmax",
    "sum-no-improvement",
    "layernorm-strided",
    "fused",
  ]) {
    const record = JSON.parse(
      fs.readFileSync(
        new URL(`../../lab/data/control-${name}.json`, import.meta.url),
      ),
    );
    assert.equal(validateRecord(record), record);
  }
});
test("malformed nested data cannot reach a view renderer", () => {
  for (const mutate of [
    (r) => (r.experiments = [null]),
    (r) => (r.hypotheses = [null]),
    (r) => (r.experiments[0].config = null),
    (r) => (r.workload.actual_strides = [1, 0]),
  ]) {
    const record = fixture();
    mutate(record);
    assert.throws(() => validateRecord(record));
  }
});
test("CUDA claims need explicit execution provenance and a CUDA device", () => {
  const record = fixture();
  record.evidence_kind = "cuda_run";
  assert.throws(() => validateRecord(record), /CUDA/);
});
test("promotion cannot survive failed numerical confirmation, stale seed or different layout", () => {
  for (const mutate of [
    (e) => (e.confirmation.correctness_passed = false),
    (e) => (e.confirmation.seed = e.timing.comparison.seed),
    (e) => (e.confirmation.actual_strides = [1, 1]),
  ]) {
    const record = fixture();
    mutate(record.experiments.find((e) => e.id === record.winner_id));
    assert.throws(() => validateRecord(record));
  }
});
test("a nonwinner cannot be presented as accepted", () => {
  const record = fixture();
  record.experiments[0].status = "accepted";
  assert.throws(() => validateRecord(record));
});
test("nonfinite values, oversized sample arrays, and inconsistent reported speedups are rejected", () => {
  let record = fixture();
  record.environment.invalid = Infinity;
  assert.throws(() => validateRecord(record), /non-finite/);
  record = fixture();
  record.experiments[0].timing.candidate_us = Array(10001).fill(1);
  assert.throws(() => validateRecord(record));
  record = fixture();
  record.experiments.find(
    (e) => e.id === record.winner_id,
  ).confirmation.comparison.speedup = 100;
  assert.throws(() => validateRecord(record), /promotion gate/);
});

test("optional text fields are rejected before activating any view", () => {
  for (const mutate of [
    (r) => (r.experiments[0].intervention = {}),
    (r) => (r.experiments[0].parent_id = []),
    (r) => (r.hypotheses[0].experiment = {}),
  ]) {
    const record = fixture();
    mutate(record);
    assert.throws(() => validateRecord(record));
  }
});

test("nonwinner timings cannot silently truncate an unpaired observation", () => {
  const record = fixture();
  record.experiments[0].timing.candidate_us.pop();
  assert.throws(() => validateRecord(record), /matched/);
});
