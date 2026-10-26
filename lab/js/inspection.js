import { renderLineage } from "./lineage.js";
import {
  el,
  svg,
  icon,
  badge,
  section,
  definition,
  emptyState,
  detail,
} from "./dom.js";
import {
  arr,
  samples,
  num,
  median,
  format,
  selectedExperiment,
  measurement,
  evidenceLabel,
  recordName,
  searchRecords,
  scientific,
  selectionFinding,
  traceBounds,
} from "./model.js";
import { candidateLedger } from "./overview.js";
import { policyChart } from "./charts.js";

export const viewInfo = {
  timeline: [
    "GPU timeline",
    "Recorded device activity, kept separate from benchmark samples.",
  ],
  bottlenecks: [
    "Bottleneck explorer",
    "What is measured, what is plausible, and which observation is still missing.",
  ],
  hypotheses: [
    "Hypothesis laboratory",
    "Keep competing explanations. Name the experiment that could disprove each one.",
  ],
  evolution: [
    "Optimization evolution",
    "Every branch belongs in the history, including the unsuccessful ones.",
  ],
  code: [
    "Implementation comparison",
    "Inspect source, numerical checks, and the exact configuration behind a result.",
  ],
  atlas: [
    "Performance atlas",
    "A searchable collection of recorded investigations.",
  ],
  research: [
    "Research results",
    "Test whether learned selection earns its complexity.",
  ],
};
export function viewHeading(view, extra) {
  const [title, description] = viewInfo[view];
  return el(
    "header",
    { class: "view-heading" },
    el(
      "div",
      {},
      el("div", { class: "eyebrow" }, "INVESTIGATION / " + view.toUpperCase()),
      el("h1", {}, title),
      el("p", {}, description),
    ),
    extra,
  );
}
export function renderTimeline(state) {
  const trace = state.record?.trace;
  if (trace?.status !== "recorded" || !arr(trace.events).length)
    return el(
      "div",
      {},
      viewHeading("timeline"),
      emptyState(
        "No GPU execution trace was collected",
        "Latency samples cannot reveal kernel overlap, launch gaps, or synchronization. Collect a separate profiler pass on a supported NVIDIA device.",
        "timeline",
      ),
      section(
        "Collect the missing evidence",
        "An instrumented profiler pass is not a steady-state latency measurement.",
        definition([
          ["CLI option", "kd investigate --trace"],
          ["Trace state", trace?.status || "Unavailable"],
          ["Evidence scope", evidenceLabel(state.record)],
          [
            "Collection note",
            trace?.reason || "No trace attached to this record.",
          ],
        ]),
      ),
    );
  const events = trace.events.filter(
    (event) =>
      Number.isFinite(event.start_us ?? event.ts) &&
      Number.isFinite(event.duration_us ?? event.dur),
  );
  if (!events.length)
    return emptyState(
      "No interpretable trace events",
      "Expected recorded timestamped duration events in microseconds.",
      "timeline",
    );
  const { origin, extent } = traceBounds(events);
  return el(
    "div",
    {},
    viewHeading("timeline", badge("RECORDED TRACE", "positive")),
    section(
      "Execution interval",
      `${events.length} recorded events · source ${trace.source || "See provenance"}`,
      el(
        "div",
        { class: "timeline" },
        events.slice(0, 300).map((event) => {
          const start = event.start_us ?? event.ts,
            duration = event.duration_us ?? event.dur;
          const bar = el("span", {
            class: "timeline-bar",
            title: `${num(start - origin)} µs start, ${num(duration)} µs duration`,
          });
          bar.style.left = `${((start - origin) / extent) * 100}%`;
          bar.style.width = `${(duration / extent) * 100}%`;
          return el(
            "div",
            { class: "timeline-row" },
            el("span", {}, event.name || "Event"),
            el(
              "div",
              {
                class: "timeline-track",
                role: "img",
                "aria-label": `${event.name || "Event"}: starts at ${num(start - origin)} microseconds, duration ${num(duration)} microseconds`,
              },
              bar,
            ),
            el("span", { class: "mono" }, `${num(duration)} µs`),
          );
        }),
      ),
    ),
    el(
      "p",
      { class: "footnote" },
      `Showing ${Math.min(events.length, 300)} of ${events.length} recorded events. Relative timestamps from the first event; timeline and benchmark samples come from separate passes.`,
    ),
  );
}
export function renderBottlenecks(state, actions) {
  const record = state.record,
    counters = record?.counters || record?.profile?.counters || {};
  const entries = Object.entries(counters).filter(
    ([, value]) => typeof value === "number",
  );
  return el(
    "div",
    {},
    viewHeading("bottlenecks"),
    el(
      "div",
      { class: "instrument-readouts" },
      [
        "Memory traffic",
        "Compute throughput",
        "Achieved occupancy",
        "Launch overhead",
      ].map((title, index) =>
        el(
          "div",
          { class: "instrument-readout" },
          el("span", { class: "eyebrow" }, `0${index + 1} / OBSERVATION`),
          el("h2", {}, title),
          el("strong", {}, "Unmeasured"),
          el(
            "p",
            {},
            [
              "Requires DRAM transactions + sustained bandwidth.",
              "Requires instruction or arithmetic throughput.",
              "Requires occupancy, register and spill counters.",
              "Requires host/device trace or graph-replay control.",
            ][index],
          ),
        ),
      ),
    ),
    section(
      "Counter evidence",
      "Original metric names and units must remain attached to their collection conditions.",
      entries.length
        ? definition(entries.map(([key, value]) => [key, format(value)]))
        : el(
            "div",
            { class: "missing-evidence" },
            icon("bottlenecks"),
            el(
              "div",
              {},
              el("h3", {}, "No hardware counters in this record"),
              el(
                "p",
                {},
                "Tensor dimensions and latency can suggest an experiment. They cannot establish GPU utilization.",
              ),
            ),
          ),
    ),
    section(
      "Roofline readiness",
      "A roofline requires defensible traffic, arithmetic work, and hardware ceilings.",
      el(
        "div",
        { class: "readiness-grid" },
        [
          "Measured arithmetic work",
          "Measured memory traffic",
          "Sustained bandwidth ceiling",
          "Device compute ceiling",
        ].map((label) =>
          el("div", {}, icon("info"), label, el("span", {}, "Missing")),
        ),
      ),
    ),
    el(
      "button",
      { class: "button ghost", onclick: () => actions.navigate("hypotheses") },
      "Inspect the diagnostic leads",
      icon("arrow"),
    ),
  );
}
export function renderHypotheses(state) {
  const hypotheses = arr(state.record?.hypotheses);
  if (!hypotheses.length)
    return el(
      "div",
      {},
      viewHeading("hypotheses"),
      emptyState(
        "No hypotheses recorded",
        "A useful hypothesis names its evidence, missing observation, and a test that could refute it.",
        "hypotheses",
      ),
    );
  return el(
    "div",
    {},
    viewHeading("hypotheses", badge(`${hypotheses.length} OPEN LEADS`)),
    el(
      "div",
      { class: "hypothesis-list" },
      hypotheses.map((hypothesis, index) =>
        el(
          "article",
          { class: "hypothesis-card" },
          el(
            "div",
            { class: "hypothesis-spine" },
            `H${String(index + 1).padStart(2, "0")}`,
            el("span", { class: "spine-line" }),
          ),
          el(
            "div",
            { class: "hypothesis-content" },
            el(
              "div",
              { class: "hypothesis-title" },
              el(
                "div",
                {},
                el("span", { class: "eyebrow" }, hypothesis.id),
                el("h2", {}, hypothesis.title),
              ),
              badge(hypothesis.status || "untested"),
            ),
            el(
              "div",
              { class: "evidence-columns" },
              el(
                "div",
                {},
                el("h3", {}, icon("check"), "Supporting observations"),
                el(
                  "ul",
                  {},
                  arr(hypothesis.evidence).map((value) =>
                    el("li", {}, format(value)),
                  ),
                ),
              ),
              el(
                "div",
                {},
                el("h3", {}, icon("info"), "Still missing"),
                el(
                  "ul",
                  {},
                  arr(hypothesis.missing_evidence).map((value) =>
                    el("li", {}, format(value)),
                  ),
                ),
              ),
            ),
            el(
              "div",
              { class: "falsifier" },
              el("span", { class: "eyebrow" }, "FALSIFIABLE TEST"),
              el("p", {}, hypothesis.experiment),
              el(
                "div",
                { class: "falsifier-result" },
                icon("arrow"),
                el(
                  "p",
                  {},
                  el("strong", {}, "Would refute it: "),
                  hypothesis.falsifier,
                ),
              ),
            ),
            el(
              "div",
              { class: "hypothesis-footer" },
              el(
                "span",
                {},
                `Recorded confidence: ${format(hypothesis.confidence)}`,
              ),
              el("span", {}, "Assessment, not proof of causality"),
            ),
          ),
        ),
      ),
    ),
  );
}
export function renderEvolution(state, actions) {
  const record = state.record,
    experiments = arr(record?.experiments),
    selected = selectedExperiment(record, state.chosen);
  if (!experiments.length)
    return el(
      "div",
      {},
      viewHeading("evolution"),
      emptyState(
        "No optimization attempts recorded",
        "Failed candidates remain part of the history when an investigation is collected.",
        "evolution",
      ),
    );
  return el(
    "div",
    {},
    viewHeading("evolution"),
    section(
      "Experimental lineage",
      "Select an actual attempt. Connector labels preserve its recorded parent.",
      renderLineage(record, selected, actions.selectCandidate),
    ),
    section(
      "Selected attempt",
      selected?.id || "Candidate",
      el(
        "div",
        { class: "selected-attempt" },
        definition([
          ["Decision", selected?.status],
          ["Block size", selected?.config?.block_size],
          ["Warp count", selected?.config?.num_warps],
          [
            "Gate result",
            selected?.confirmation?.comparison?.reason ||
              selected?.timing?.comparison?.reason ||
              selected?.error ||
              "Not recorded",
          ],
        ]),
        el(
          "button",
          { class: "button ghost", onclick: () => actions.navigate("code") },
          "Read the implementation",
          icon("code"),
        ),
      ),
    ),
    section(
      "Complete candidate ledger",
      "Reported confirmation values are used for the accepted candidate.",
      candidateLedger(record, selected?.id, actions.selectCandidate),
    ),
  );
}
function sourceLines(source, other) {
  const existing = new Set(
    (other || "")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean),
  );
  return el(
    "pre",
    { class: "source-lines" },
    el(
      "code",
      {},
      String(source || "Source not recorded.")
        .split("\n")
        .map((line, index) =>
          el(
            "span",
            {
              class: `source-line ${line.trim() && !existing.has(line.trim()) ? "changed" : ""}`,
            },
            el(
              "span",
              { class: "line-number", "aria-hidden": "true" },
              index + 1,
            ),
            el("span", {}, line || " "),
          ),
        ),
    ),
  );
}
export function renderCode(state, actions) {
  const record = state.record,
    experiment = selectedExperiment(record, state.chosen),
    experiments = arr(record?.experiments);
  if (!experiment)
    return el(
      "div",
      {},
      viewHeading("code"),
      emptyState(
        "No implementation attached",
        "Import a record containing candidate source and its declared reference.",
        "code",
      ),
    );
  const candidate = el(
    "select",
    {
      "aria-label": "Select implementation candidate",
      onchange: (event) => actions.selectCandidate(event.target.value),
    },
    experiments.map((item) => el("option", { value: item.id }, item.id)),
  );
  candidate.value = experiment.id;
  const reference = experiment.source?.baseline,
    optimized = experiment.source?.candidate;
  return el(
    "div",
    {},
    viewHeading(
      "code",
      el(
        "div",
        { class: "code-selector" },
        candidate,
        badge(
          experiment.status === "accepted"
            ? "Accepted · record"
            : experiment.status,
          experiment.status === "accepted" ? "positive" : "neutral",
        ),
      ),
    ),
    el(
      "div",
      { class: "code-diff" },
      el(
        "section",
        { class: "source-pane" },
        el(
          "header",
          {},
          el("span", { class: "source-dot baseline" }),
          "Declared reference",
          el("span", {}, "READ ONLY"),
        ),
        sourceLines(reference, optimized),
      ),
      el(
        "section",
        { class: "source-pane candidate-source" },
        el(
          "header",
          {},
          el("span", { class: "source-dot candidate" }),
          "Candidate implementation",
          el("span", {}, "READ ONLY"),
        ),
        sourceLines(optimized, reference),
      ),
    ),
    el(
      "p",
      { class: "footnote" },
      "Highlighted lines are unique to that side, not a semantic diff. Imported source is displayed as text and never executed.",
    ),
    el(
      "div",
      { class: "two-column" },
      section(
        "Numerical contract",
        "Precision and behavior must survive the change.",
        definition([
          [
            "Correctness",
            experiment.correctness?.passed
              ? "Passed in record"
              : "Failed / not recorded",
          ],
          ["Input cases", arr(experiment.correctness?.tests).length],
          [
            "Max absolute error",
            scientific(experiment.correctness?.max_abs_error),
          ],
          ["Absolute tolerance", format(experiment.correctness?.atol)],
          ["Relative tolerance", format(experiment.correctness?.rtol)],
          ["Memory safety", experiment.safety?.status || "Unverified"],
        ]),
      ),
      section(
        "Execution configuration",
        "Compilation and measurement remain distinct.",
        definition([
          ["Configuration", format(experiment.config)],
          ["Compile / first dispatch", `${num(experiment.compile_us)} µs`],
          [
            "Discovery interval",
            experiment.timing?.comparison
              ? `${num(experiment.timing.comparison.ci_low)}–${num(experiment.timing.comparison.ci_high)}×`
              : "Not recorded",
          ],
          [
            "Confirmation",
            experiment.confirmation?.comparison?.accepted
              ? "Passed in record"
              : "Not established",
          ],
          ["Safety note", experiment.safety?.reason || "Not recorded"],
        ]),
      ),
    ),
    detail(
      "All numerical check cases",
      el(
        "pre",
        { class: "json-source" },
        JSON.stringify(experiment.correctness?.tests || [], null, 2),
      ),
    ),
  );
}
export function renderAtlas(state, actions) {
  const search = el("input", {
    type: "search",
    placeholder: "Operation, shape, device, intervention…",
    "aria-label": "Search investigations",
  });
  const kind = el(
    "select",
    { "aria-label": "Filter evidence" },
    el("option", { value: "" }, "All evidence"),
    el("option", { value: "cpu_control" }, "CPU controls"),
    el("option", { value: "cuda_run" }, "CUDA runs"),
    el("option", { value: "reference" }, "Reference records"),
  );
  const results = el("div", {}),
    count = el("span", { class: "mono muted", "aria-live": "polite" });
  const draw = () => {
    const records = searchRecords(state.records, search.value, kind.value);
    count.textContent = `${records.length} RECORD${records.length === 1 ? "" : "S"}`;
    results.replaceChildren(
      records.length
        ? el(
            "div",
            { class: "atlas-grid" },
            records.map((record) => {
              const experiment = selectedExperiment(record),
                data = measurement(record, experiment);
              return el(
                "button",
                {
                  class: "atlas-record",
                  onclick: () => actions.activate(record),
                },
                el(
                  "div",
                  { class: "atlas-record-top" },
                  icon("file"),
                  badge(evidenceLabel(record), "scope"),
                ),
                el("span", { class: "eyebrow" }, record.id.slice(0, 12)),
                el("h2", {}, record.workload.family),
                el(
                  "div",
                  { class: "atlas-shape mono" },
                  record.workload.shape.join(" × "),
                ),
                el(
                  "div",
                  { class: "atlas-record-meta" },
                  el("span", {}, record.workload.dtype),
                  el("span", {}, record.device.name),
                ),
                el(
                  "div",
                  { class: "atlas-record-bottom" },
                  el(
                    "span",
                    {},
                    record.status === "accepted"
                      ? "Accepted in record"
                      : "No promotion",
                  ),
                  el("strong", {}, `${num(data.comparison?.speedup)}×`),
                ),
                el(
                  "span",
                  { class: "atlas-open" },
                  "Open investigation",
                  icon("arrow"),
                ),
              );
            }),
          )
        : emptyState(
            "No matching investigations",
            "Try a different operation, shape, or evidence filter.",
            "search",
          ),
    );
  };
  search.addEventListener("input", draw);
  kind.addEventListener("change", draw);
  draw();
  return el(
    "div",
    {},
    viewHeading("atlas", count),
    el(
      "div",
      { class: "atlas-toolbar" },
      el("label", { class: "search-field" }, icon("search"), search),
      kind,
    ),
    results,
  );
}
export function renderResearch(state) {
  const evaluation = state.evaluation;
  if (!evaluation)
    return el(
      "div",
      {},
      viewHeading("research"),
      emptyState(
        "No evaluation attached",
        "A learned model needs held-out comparisons against simpler strategies.",
        "research",
      ),
    );
  const finding = selectionFinding(evaluation);
  const aliases = {
    learned_ucb: "Learned UCB",
    heuristic: "Fixed rule",
    random: "Random",
    no_context: "No context",
    no_uncertainty: "No uncertainty",
    bayesian_ucb: "Bayesian GP-UCB",
  };
  const split = el(
    "select",
    { "aria-label": "Evaluation holdout split" },
    Object.entries(evaluation.splits || {})
      .filter(([, value]) => arr(value.aggregates).length)
      .map(([key]) => el("option", { value: key }, `Held-out ${key}`)),
  );
  const budget = el("select", { "aria-label": "Evaluation candidate budget" }),
    results = el("div", {});
  const draw = () => {
    const current = evaluation.splits[split.value];
    const rows = arr(current?.aggregates).filter(
      (row) => row.budget === Number(budget.value),
    );
    if (!rows.length) {
      results.replaceChildren(
        emptyState(
          "No aggregates for this selection",
          "Select a recorded split and budget.",
          "research",
        ),
      );
      return;
    }
    results.replaceChildren(
      el(
        "p",
        { class: "chart-context" },
        `${current.folds} holdout groups · ${budget.value} candidate${budget.value === "1" ? "" : "s"} per task · mean best log(speedup)`,
      ),
      policyChart(rows, aliases),
      el(
        "div",
        {
          class: "table-scroll",
          tabindex: 0,
          "aria-label": "Scrollable experiment table",
        },
        el(
          "table",
          {},
          el(
            "thead",
            {},
            el(
              "tr",
              {},
              [
                "Policy",
                "Workloads",
                "Mean reward",
                "Oracle gap",
                "Cost / s",
              ].map((label) => el("th", { scope: "col" }, label)),
            ),
          ),
          el(
            "tbody",
            {},
            rows.map((row) =>
              el(
                "tr",
                {},
                el("td", {}, aliases[row.policy] || row.policy),
                el("td", { class: "mono" }, row.workloads),
                el("td", { class: "mono" }, num(row.mean_best_reward, 4)),
                el("td", { class: "mono" }, num(row.mean_oracle_gap, 4)),
                el("td", { class: "mono" }, num(row.mean_cost, 4)),
              ),
            ),
          ),
        ),
      ),
    );
  };
  const setBudget = () => {
    const values = [
      ...new Set(
        arr(evaluation.splits[split.value]?.aggregates).map(
          (row) => row.budget,
        ),
      ),
    ].sort((a, b) => a - b);
    budget.replaceChildren(
      ...values.map((value) =>
        el("option", { value }, `${value} candidate${value === 1 ? "" : "s"}`),
      ),
    );
    draw();
  };
  split.addEventListener("change", setBudget);
  budget.addEventListener("change", draw);
  setBudget();
  return el(
    "div",
    {},
    viewHeading(
      "research",
      badge(
        evaluation.device_kind === "cpu" ? "CPU REPLAY" : "RECORDED REPLAY",
        "scope",
      ),
    ),
    el(
      "div",
      { class: "research-finding" },
      el("div", { class: "eyebrow" }, "THE RECORDED RESULT"),
      el("h2", {}, finding.title),
      el("p", {}, finding.body),
    ),
    section(
      "Held-out selection performance",
      `${evaluation.observations} actual recorded interventions. Larger mean reward is better.`,
      el(
        "div",
        {},
        el("div", { class: "evaluation-controls" }, split, budget),
        results,
      ),
    ),
    el(
      "p",
      { class: "footnote" },
      "The bundled result concerns CPU controls. Offline replay reveals observed candidate rewards; it is not prospective GPU search or a causal estimate. Oracle gap refers only to the measured pool.",
    ),
    section(
      "Research boundaries",
      "Missing comparisons remain visible.",
      el(
        "div",
        { class: "readiness-grid" },
        [
          ["CUDA execution", "Not established"],
          ["Triton autotuning", "Not reproduced"],
          ["Hardware holdout", "Not established"],
          ["Agent baseline", "Not reproduced"],
        ].map(([label, status]) =>
          el("div", {}, icon("info"), label, el("span", {}, status)),
        ),
      ),
    ),
    detail(
      "Evaluation protocol & limitations",
      el(
        "ul",
        { class: "limit-list" },
        arr(evaluation.limitations).map((value) => el("li", {}, value)),
      ),
      true,
    ),
  );
}
