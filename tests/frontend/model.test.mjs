import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  median,
  quantile,
  histogram,
  selectedExperiment,
  measurement,
  searchRecords,
  scientific,
  selectionFinding,
} from "../../lab/js/model.js";
const record = JSON.parse(
  fs.readFileSync(
    new URL("../../lab/data/control-softmax.json", import.meta.url),
  ),
);

test("even/odd quantiles and empty measurements retain actual values", () => {
  assert.equal(median([4, 1, 2, 3]), 2.5);
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([]), undefined);
  assert.equal(quantile([1, 5], 0.25), 2);
});
test("histogram conserves all observations including both bounds and a degenerate range", () => {
  for (const values of [
    [1, 2, 3, 4],
    [10, 10, 10],
  ]) {
    const result = histogram(values, [values[0]], 7);
    assert.equal(
      result.baseline.reduce((sum, value) => sum + value, 0),
      values.length,
    );
    assert.equal(
      result.candidate.reduce((sum, value) => sum + value, 0),
      1,
    );
    assert.ok(result.high > result.low);
  }
  assert.equal(histogram([], []), null);
});
test("accepted display uses its own confirmation pair, rejected display uses discovery", () => {
  const winner = selectedExperiment(record);
  const accepted = measurement(record, winner);
  assert.equal(winner.id, record.winner_id);
  assert.deepEqual(accepted.baseline, winner.confirmation.baseline_us);
  assert.deepEqual(accepted.candidate, winner.confirmation.candidate_us);
  assert.equal(
    accepted.comparison.speedup,
    median(accepted.baseline) / median(accepted.candidate),
  );
  const rejected = selectedExperiment(record, "cpu-32");
  assert.equal(measurement(record, rejected).accepted, false);
  assert.deepEqual(
    measurement(record, rejected).baseline,
    rejected.timing.baseline_us,
  );
});
test("search distinguishes evidence and shape without changing stored records", () => {
  const original = JSON.stringify(record);
  assert.equal(searchRecords([record], "SOFTMAX  ").length, 1);
  assert.equal(searchRecords([record], "", "cuda_run").length, 0);
  assert.equal(JSON.stringify(record), original);
});
test("small nonzero errors remain nonzero on screen", () => {
  assert.equal(scientific(1.2e-7), "1.20e-7");
  assert.equal(scientific(0), "0");
});
test("research conclusion follows actual aggregate values rather than a fixed marketing claim", () => {
  const report = (value) => ({
    splits: {
      family: {
        aggregates: [
          { budget: 1, policy: "learned_ucb", mean_best_reward: value },
          { budget: 1, policy: "heuristic", mean_best_reward: 0.3 },
        ],
      },
    },
  });
  assert.match(selectionFinding(report(0.3)).title, /matched/);
  assert.match(selectionFinding(report(0.2)).title, /simple rule led/);
  assert.match(selectionFinding(report(0.4)).title, /Learned selection led/);
});

test("lineage grouping preserves real parents including missing and cyclic references", async () => {
  const { lineageRelations } = await import("../../lab/js/model.js");
  const nodes = [
    { id: "a", parent_id: null },
    { id: "b", parent_id: "a" },
    { id: "c", parent_id: "missing" },
  ];
  const graph = lineageRelations(nodes);
  assert.equal(graph.flat, false);
  assert.deepEqual(
    graph.groups.map((g) => [g.parent, g.known]),
    [
      [null, true],
      ["a", true],
      ["missing", false],
    ],
  );
  assert.equal(
    lineageRelations([
      { id: "x", parent_id: "y" },
      { id: "y", parent_id: "x" },
    ]).groups.length,
    2,
  );
});
test("large profiler traces have bounded extrema computation without argument spreading", async () => {
  const { traceBounds } = await import("../../lab/js/model.js");
  const events = Array.from({ length: 150000 }, (_, i) => ({
    start_us: i * 2,
    duration_us: 1,
  }));
  assert.deepEqual(traceBounds(events), {
    origin: 0,
    end: 299999,
    extent: 299999,
  });
});
