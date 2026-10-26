/** Pure data transforms. Every visual value derives from the imported record. */
export const arr = (value) => (Array.isArray(value) ? value : []);
export const finite = (value) =>
  typeof value === "number" && Number.isFinite(value);
export const samples = (value) =>
  arr(value).filter((value) => finite(value) && value > 0);
export const num = (value, digits = 2) =>
  finite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "—";
export const format = (value) =>
  value == null
    ? "Not recorded"
    : typeof value === "object"
      ? JSON.stringify(value)
      : String(value);
export function quantile(values, fraction) {
  const sorted = values.filter(finite).toSorted((a, b) => a - b);
  if (!sorted.length) return undefined;
  const index = (sorted.length - 1) * fraction;
  const lower = Math.floor(index);
  return (
    sorted[lower] + (sorted[Math.ceil(index)] - sorted[lower]) * (index - lower)
  );
}
export const median = (values) => quantile(values, 0.5);
export const evidenceLabel = (record) =>
  ({
    cuda_run: "CUDA run",
    cpu_control: "CPU control",
    reference: "Reference",
  })[record?.evidence_kind] || "No record";
export const recordName = (record) =>
  `${record.workload.family} / ${record.workload.shape.join(" × ")}`;
export const operationName = (family) =>
  ({
    sum: "Row reduction",
    softmax: "Softmax",
    layernorm: "Layer normalization",
    fused: "Fused elementwise",
  })[family] || "Investigation";
export function selectedExperiment(record, selectedId) {
  const experiments = arr(record?.experiments);
  return (
    experiments.find((item) => item.id === selectedId) ||
    experiments.find((item) => item.id === record?.winner_id) ||
    experiments
      .filter(
        (item) =>
          item.correctness?.passed && samples(item.timing?.candidate_us).length,
      )
      .sort(
        (a, b) =>
          median(samples(a.timing.candidate_us)) -
          median(samples(b.timing.candidate_us)),
      )[0] ||
    experiments[0]
  );
}
export function measurement(record, experiment) {
  const phase =
    experiment?.status === "accepted" && experiment.confirmation?.comparison
      ? experiment.confirmation
      : experiment?.timing;
  const baseline = samples(phase?.baseline_us);
  return {
    baseline: baseline.length
      ? baseline
      : samples(record?.baseline?.samples_us),
    candidate: samples(phase?.candidate_us),
    comparison: phase?.comparison,
    phase:
      phase === experiment?.confirmation
        ? "Independent confirmation"
        : "Discovery measurement",
    accepted:
      experiment?.id === record?.winner_id &&
      experiment?.status === "accepted" &&
      experiment.correctness?.passed === true &&
      experiment.confirmation?.correctness_passed === true,
  };
}
export function histogram(baseline, candidate, bins = 30) {
  const values = [...baseline, ...candidate].filter(finite);
  if (!values.length) return null;
  let low = Math.min(...values),
    high = Math.max(...values);
  if (low === high) {
    low *= 0.95;
    high *= 1.05;
  }
  const padding = (high - low) * 0.045;
  low = Math.max(0, low - padding);
  high += padding;
  const count = (list) => {
    const result = Array(bins).fill(0);
    for (const value of list) {
      const index = Math.max(
        0,
        Math.min(bins - 1, Math.floor(((value - low) / (high - low)) * bins)),
      );
      result[index]++;
    }
    return result;
  };
  return { low, high, baseline: count(baseline), candidate: count(candidate) };
}
export function decisionText(record, experiment) {
  if (!record) return "Awaiting evidence";
  if (record.status === "blocked") return "Hardware unavailable";
  if (experiment?.status === "invalid") return "Numerical check failed";
  if (experiment?.status === "failed") return "Execution failed";
  if (experiment?.id === record.winner_id) return "Accepted in record";
  return "Not promoted";
}
export function searchRecords(records, query, kind = "") {
  const normalized = query.toLowerCase().trim();
  return records.filter(
    (record) =>
      (!kind || record.evidence_kind === kind) &&
      JSON.stringify([
        record.id,
        record.title,
        record.workload,
        record.device,
        arr(record.experiments).map((item) => item.intervention),
      ])
        .toLowerCase()
        .includes(normalized),
  );
}

export function scientific(value) {
  if (!finite(value)) return "Not recorded";
  if (value === 0) return "0";
  return Math.abs(value) < 0.001 ? value.toExponential(2) : num(value, 5);
}
export function intervalLabel(comparison) {
  return finite(comparison?.confidence_level)
    ? `${num(comparison.confidence_level * 100, 0)}% interval`
    : "Reported interval";
}
export function selectionFinding(evaluation) {
  const rows = arr(evaluation?.splits?.family?.aggregates).filter(
    (row) => row.budget === 1,
  );
  const learned = rows.find((row) => row.policy === "learned_ucb");
  const rule = rows.find((row) => row.policy === "heuristic");
  if (!learned || !rule)
    return {
      title: "A comparison needs its baseline.",
      body: "This evaluation does not contain a matched learned/rule comparison at a one-candidate budget.",
    };
  const gap = learned.mean_best_reward - rule.mean_best_reward;
  if (Math.abs(gap) < 1e-9)
    return {
      title: "Learning matched the simple rule.",
      body: "The learned selector and fixed heuristic achieved the same mean first-candidate reward in this recorded replay. Added learning benefit was not established.",
    };
  return {
    title:
      gap > 0
        ? "Learned selection led this replay."
        : "The simple rule led this replay.",
    body: "This is an offline comparison in the recorded candidate pool. It does not establish prospective optimization benefit or transfer to other hardware.",
  };
}

/** Preserve topology as parent groups; unknown parents never become baseline edges. */
export function lineageRelations(experiments) {
  const groups = new Map();
  const ids = new Set(experiments.map((experiment) => experiment.id));
  for (const experiment of experiments) {
    const parent = experiment.parent_id ?? null;
    if (!groups.has(parent)) groups.set(parent, []);
    groups.get(parent).push(experiment);
  }
  return {
    flat: experiments.every((experiment) => experiment.parent_id == null),
    groups: [...groups].map(([parent, children]) => ({
      parent,
      children,
      known: parent === null || ids.has(parent),
    })),
  };
}
export function traceBounds(events) {
  let origin = Infinity,
    end = -Infinity;
  for (const event of events) {
    const start = event.start_us ?? event.ts,
      duration = event.duration_us ?? event.dur;
    if (!finite(start) || !finite(duration) || duration < 0) continue;
    origin = Math.min(origin, start);
    end = Math.max(end, start + duration);
  }
  return {
    origin,
    end,
    extent: Number.isFinite(end - origin) ? end - origin || 1 : 1,
  };
}
