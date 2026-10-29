/** Pure presentation transforms. These summarize reported evidence, never promote it. */
import { arr, finite, measurement, median, num } from "./model.js";

export function comparisonRows(record) {
  return arr(record?.experiments).map((experiment, index) => {
    const data = measurement(record, experiment);
    const comparison = data.comparison;
    const interval =
      comparison &&
      finite(comparison.ci_low) &&
      finite(comparison.ci_high) &&
      comparison.ci_low > 0 &&
      comparison.ci_high >= comparison.ci_low
        ? [comparison.ci_low, comparison.ci_high]
        : null;
    return {
      experiment,
      index,
      ...data,
      interval,
      ratio: finite(comparison?.speedup) ? comparison.speedup : null,
    };
  });
}

export function comparisonDomain(rows) {
  let low = 1,
    high = 1;
  for (const row of rows) {
    for (const value of [...(row.interval || []), row.ratio]) {
      if (!finite(value) || value <= 0) continue;
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
  }
  const padding = Math.max((high - low) * 0.12, 0.12);
  return [Math.max(0, low - padding), high + padding];
}

export function resultNarrative(record, experiment) {
  const data = measurement(record, experiment);
  const base = median(data.baseline),
    candidate = median(data.candidate);
  if (experiment?.correctness?.passed !== true)
    return {
      title: "Correctness comes first.",
      text: "This attempt has no passing numerical check. Its timing cannot establish a valid optimization.",
      tone: "neutral",
    };
  if (!finite(base) || !finite(candidate))
    return {
      title: "An implementation needs a measurement.",
      text: "No matched candidate timing is available for this attempt.",
      tone: "neutral",
    };
  if (data.accepted) {
    const saving = base - candidate;
    return {
      title: `A recorded ${num((saving / base) * 100, 1)}% latency reduction.`,
      text: `${experiment.id} saves ${num(saving)} µs at the median against its matched reference. Independent confirmation is reported; the browser has not rerun the workload.`,
      tone: "positive",
    };
  }
  return {
    title: "Measured, but not promoted.",
    text:
      (experiment?.timing?.comparison?.reason
        ? `Discovery gate: ${experiment.timing.comparison.reason}. No accepted independent confirmation is attached to this attempt.`
        : null) ||
      experiment?.error ||
      "This attempt has no accepted independent confirmation. Inspect its recorded gate before drawing a conclusion.",
    tone: "neutral",
  };
}

export function timingCSV(data) {
  const count = Math.min(data.baseline.length, data.candidate.length);
  const lines = ["pair,reference_us,candidate_us,phase"];
  for (let index = 0; index < count; index++)
    lines.push(
      `${index + 1},${data.baseline[index]},${data.candidate[index]},${data.phase === "Independent confirmation" ? "confirmation" : "discovery"}`,
    );
  return lines.join("\n") + "\n";
}

export const routeIds = new Set([
  "overview",
  "timeline",
  "bottlenecks",
  "hypotheses",
  "evolution",
  "code",
  "atlas",
  "research",
]);
export function readLocation(hash) {
  const [route, query = ""] = hash.replace(/^#/, "").split("?", 2);
  const params = new URLSearchParams(query);
  return {
    view: routeIds.has(route) ? route : "overview",
    record: params.get("record"),
    candidate: params.get("candidate"),
  };
}
export function locationHash(view, record, candidate) {
  const params = new URLSearchParams();
  if (record) params.set("record", record);
  if (candidate) params.set("candidate", candidate);
  return `#${routeIds.has(view) ? view : "overview"}${params.size ? "?" + params : ""}`;
}
