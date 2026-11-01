import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { lineDiff } from "../../lab/js/line-diff.js";
import {
  comparisonRows,
  comparisonDomain,
  timingCSV,
  readLocation,
  locationHash,
  resultNarrative,
} from "../../lab/js/evidence-model.js";

const fixture = () =>
  JSON.parse(
    readFileSync(
      new URL("../../lab/data/control-softmax.json", import.meta.url),
    ),
  );

function reconstruct(diff) {
  return [
    diff.rows
      .filter((r) => r.kind !== "add")
      .map((r) => r.text)
      .join("\n"),
    diff.rows
      .filter((r) => r.kind !== "remove")
      .map((r) => r.text)
      .join("\n"),
  ];
}
test("line diff preserves repeated lines, whitespace, blank lines and trailing newlines", () => {
  for (const [left, right] of [
    ["", ""],
    ["a\na\nb\n", "a\nb\na\n"],
    ["    x\n\nreturn x", "  x\n\nreturn x\n"],
    ["a\nb\nc", "a\nc"],
    ["x", "x\ny"],
    ["<script>unsafe()</script>", "ordinary text"],
  ]) {
    const diff = lineDiff(left, right);
    assert.deepEqual(reconstruct(diff), [left, right]);
    let old = 0,
      next = 0;
    for (const row of diff.rows) {
      assert.equal(row.old, row.kind === "add" ? null : ++old);
      assert.equal(row.new, row.kind === "remove" ? null : ++next);
    }
  }
  assert.equal(lineDiff("x\nx", "x").removals, 1);
});
test("bounded diff is still an exact replacement with explicit fallback status", () => {
  const left =
    "start\n" +
    Array.from({ length: 1000 }, (_, i) => `left-${i}`).join("\n") +
    "\nend";
  const right =
    "start\n" +
    Array.from({ length: 1000 }, (_, i) => `right-${i}`).join("\n") +
    "\nend";
  const diff = lineDiff(left, right, 100);
  assert.equal(diff.bounded, true);
  assert.equal(diff.additions, 1000);
  assert.equal(diff.removals, 1000);
  assert.deepEqual(reconstruct(diff), [left, right]);
});
test("comparison preserves record order, phase, rejection and matching reference arrays", () => {
  const record = fixture(),
    rows = comparisonRows(record);
  assert.deepEqual(
    rows.map((r) => r.experiment.id),
    record.experiments.map((e) => e.id),
  );
  const winner = rows.find((r) => r.experiment.id === record.winner_id);
  assert.equal(winner.accepted, true);
  assert.equal(winner.phase, "Independent confirmation");
  assert.deepEqual(winner.baseline, winner.experiment.confirmation.baseline_us);
  const rejected = rows.find((r) => r.experiment.id !== record.winner_id);
  assert.equal(rejected.accepted, false);
  assert.equal(rejected.phase, "Discovery measurement");
  assert.deepEqual(rejected.baseline, rejected.experiment.timing.baseline_us);
  const [low, high] = comparisonDomain(rows);
  assert.ok(low < 1 && high > 1);
  for (const row of rows)
    for (const value of [...row.interval, row.ratio])
      assert.ok(value > low && value < high);
  assert.deepEqual(comparisonDomain([]), [0.88, 1.12]);
});
test("CSV exports exact matched observations without rounded numbers", () => {
  const row = comparisonRows(fixture()).find((r) => r.accepted);
  const lines = timingCSV(row).trim().split("\n");
  assert.equal(lines.length, row.baseline.length + 1);
  assert.equal(lines[0], "pair,reference_us,candidate_us,phase");
  assert.equal(
    lines[1],
    `1,${row.baseline[0]},${row.candidate[0]},confirmation`,
  );
});
test("deep links round trip reserved and hostile characters as data", () => {
  const hash = locationHash("code", "record/#?& 😀", 'cpu-"<script>');
  assert.deepEqual(readLocation(hash), {
    view: "code",
    record: "record/#?& 😀",
    candidate: 'cpu-"<script>',
  });
  assert.deepEqual(readLocation("#unknown"), {
    view: "overview",
    record: null,
    candidate: null,
  });
  assert.deepEqual(readLocation("#research"), {
    view: "research",
    record: null,
    candidate: null,
  });
});
test("result narrative distinguishes accepted, rejected, missing and numerically invalid", () => {
  const record = fixture(),
    winner = record.experiments.find((e) => e.id === record.winner_id);
  assert.match(resultNarrative(record, winner).title, /latency reduction/);
  const rejected = record.experiments.find((e) => e.id !== record.winner_id);
  assert.match(resultNarrative(record, rejected).title, /not promoted/);
  assert.match(
    resultNarrative(record, { correctness: { passed: false } }).title,
    /Correctness/,
  );
  assert.match(
    resultNarrative(record, { correctness: { passed: true } }).title,
    /measurement/,
  );
});
