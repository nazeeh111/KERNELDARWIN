import { investigationFinding, candidateComparison } from "./comparison.js";
import {
  el,
  icon,
  badge,
  section,
  definition,
  detail,
  emptyState,
} from "./dom.js";
import {
  arr,
  samples,
  num,
  median,
  measurement,
  selectedExperiment,
  operationName,
  decisionText,
  scientific,
  intervalLabel,
} from "./model.js";
import { latencyPlot, latencyRange } from "./charts.js";

export function candidateLedger(record, selectedId, onSelect) {
  const experiments = arr(record?.experiments);
  if (!experiments.length)
    return emptyState(
      "No interventions recorded",
      "An investigation keeps unsuccessful attempts alongside its accepted candidate.",
      "evolution",
    );
  return el(
    "div",
    {
      class: "table-scroll",
      tabindex: 0,
      "aria-label": "Scrollable experiment table",
    },
    el(
      "table",
      { class: "candidate-table" },
      el(
        "thead",
        {},
        el(
          "tr",
          {},
          [
            "Candidate",
            "Intervention",
            "Numerical check",
            "Median / µs",
            "Speedup",
            "Decision",
          ].map((label) => el("th", { scope: "col" }, label)),
        ),
      ),
      el(
        "tbody",
        {},
        experiments.map((experiment, index) => {
          const data = measurement(record, experiment);
          return el(
            "tr",
            { "data-selected": String(experiment.id === selectedId) },
            el(
              "td",
              {},
              el(
                "button",
                {
                  class: "candidate-button",
                  "data-candidate": experiment.id,
                  "aria-pressed": String(experiment.id === selectedId),
                  onclick: () => onSelect(experiment.id),
                },
                el(
                  "span",
                  { class: "candidate-index" },
                  String(index + 1).padStart(2, "0"),
                ),
                experiment.id,
                icon("chevron"),
              ),
            ),
            el(
              "td",
              {},
              experiment.intervention?.replaceAll("_", " ") || "Configuration",
            ),
            el(
              "td",
              {},
              el(
                "span",
                {
                  class: experiment.correctness?.passed
                    ? "check-pass"
                    : "muted",
                },
                icon(experiment.correctness?.passed ? "check" : "close"),
                experiment.correctness?.passed ? "Passed" : "Failed / not run",
              ),
            ),
            el("td", { class: "mono" }, num(median(data.candidate))),
            el("td", { class: "mono" }, `${num(data.comparison?.speedup)}×`),
            el(
              "td",
              {},
              badge(
                experiment.status === "accepted"
                  ? "Accepted · record"
                  : experiment.status || "Not recorded",
                experiment.status === "accepted" ? "positive" : "neutral",
              ),
            ),
          );
        }),
      ),
    ),
  );
}
function checkRow(title, value, note, pass) {
  return el(
    "div",
    { class: "verification-row" },
    el(
      "span",
      { class: `verification-icon ${pass ? "pass" : ""}` },
      icon(pass ? "check" : "info"),
    ),
    el("div", {}, el("span", {}, title), el("small", {}, note)),
    el("strong", { class: pass ? "pass" : "" }, value),
  );
}
export function evidenceRail(record, experiment, actions) {
  const data = measurement(record, experiment);
  const tests = arr(experiment?.correctness?.tests);
  return el(
    "aside",
    { class: "evidence-rail" },
    el(
      "div",
      { class: "rail-title" },
      icon("lock"),
      el("h2", {}, "Verification ledger"),
    ),
    el(
      "div",
      { class: "decision-lockup" },
      badge(
        data.accepted ? "Accepted in record" : "Not promoted",
        data.accepted ? "positive" : "neutral",
      ),
      el(
        "p",
        {},
        data.accepted
          ? "The candidate cleared the recorded numerical and independent timing gates."
          : "A faster timing alone does not earn promotion.",
      ),
    ),
    checkRow(
      "Numerical checks",
      experiment?.correctness?.passed ? "Passed" : "Not established",
      `${tests.length} recorded input cases`,
      experiment?.correctness?.passed,
    ),
    checkRow(
      "Independent timing",
      experiment?.confirmation?.comparison?.accepted
        ? "Passed"
        : "Not established",
      "Fresh input + separate timing pass",
      experiment?.confirmation?.comparison?.accepted,
    ),
    checkRow(
      "Memory safety",
      experiment?.safety?.status === "not_applicable"
        ? "CPU control"
        : experiment?.safety?.status || "Unverified",
      experiment?.safety?.status === "not_applicable"
        ? "CUDA sanitizer does not apply"
        : "Inspect independent sanitizer logs",
      false,
    ),
    el(
      "div",
      { class: "rail-facts" },
      definition([
        [
          "Input tolerance",
          experiment?.correctness?.atol != null
            ? `${experiment.correctness.atol} absolute`
            : "See reference contract",
        ],
        [
          "Maximum error",
          experiment?.correctness?.max_abs_error != null
            ? scientific(experiment.correctness.max_abs_error)
            : "Not recorded",
        ],
        [
          "Source provenance",
          record?.provenance?.source_sha256 ? "Hash recorded" : "Not recorded",
        ],
      ]),
    ),
    el(
      "button",
      {
        class: "button full-width ghost",
        onclick: () => actions.navigate("code"),
      },
      icon("code"),
      "Inspect implementation",
      icon("arrow"),
    ),
    detail(
      "Validate this record",
      el(
        "div",
        {},
        el(
          "p",
          {},
          "The viewer checks structural consistency. The CLI independently recomputes promotion statistics.",
        ),
        el(
          "code",
          { class: "command" },
          `kd validate ${record?.id || "RECORD_ID"}.json`,
        ),
      ),
    ),
  );
}
export function renderOverview(state, actions) {
  const record = state.record;
  if (!record)
    return emptyState(
      "Start with a measured workload",
      "Import an investigation record to inspect its latency, interventions, and verification history.",
      "overview",
      el(
        "button",
        { class: "button primary", onclick: actions.importRecord },
        "Import record",
      ),
    );
  const experiment = selectedExperiment(record, state.chosen);
  const data = measurement(record, experiment);
  const comp = data.comparison;
  const heading = el(
    "header",
    { class: "investigation-heading" },
    el(
      "div",
      {},
      el(
        "div",
        { class: "eyebrow" },
        "PERFORMANCE INVESTIGATION",
        el("span", {}, `RUN ${record.id.slice(0, 8)}`),
      ),
      el(
        "h1",
        {},
        operationName(record.workload.family),
        el(
          "span",
          { class: "heading-shape" },
          record.workload.shape.join(" × "),
        ),
      ),
      el(
        "p",
        { class: "heading-description" },
        record.evidence_kind === "cpu_control"
          ? "NumPy reference → row-batched implementation"
          : "Declared reference → authored candidate",
      ),
    ),
    el(
      "div",
      { class: "improvement-lockup" },
      el(
        "span",
        { class: "eyebrow" },
        data.accepted ? "RECORDED IMPROVEMENT" : "MEASURED RATIO",
      ),
      el(
        "div",
        { class: "improvement-value", "data-accepted": String(data.accepted) },
        num(comp?.speedup),
        el("span", {}, "×"),
      ),
      el(
        "span",
        { class: "improvement-interval" },
        comp
          ? `${intervalLabel(comp)} ${num(comp.ci_low)}–${num(comp.ci_high)}`
          : "No comparison recorded",
      ),
    ),
  );
  const scope = el(
    "div",
    { class: "scope-line" },
    icon("info"),
    el(
      "p",
      {},
      record.evidence_kind === "cpu_control"
        ? "CPU control. These measurements validate the workflow. GPU performance is not established."
        : record.evidence_kind === "cuda_run"
          ? "CUDA record. Inspect collection provenance and independent verification before using the result."
          : "Reference record. No independently reproduced optimization is established.",
    ),
  );
  const plot = section(
    "Latency fingerprint",
    "Every timing sample, with the selected candidate’s matched reference.",
    el(
      "div",
      { class: "plot-content" },
      latencyPlot(data),
      el(
        "div",
        { class: "latency-summary" },
        latencyRange("REFERENCE", data.baseline, "baseline"),
        el("div", { class: "comparison-arrow" }, icon("arrow")),
        latencyRange(
          experiment?.id || "CANDIDATE",
          data.candidate,
          "candidate",
        ),
      ),
    ),
    badge(
      data.phase === "Independent confirmation" ? "CONFIRMATION" : "DISCOVERY",
      "neutral",
    ),
  );
  const context = el(
    "div",
    { class: "context-strip" },
    definition([
      ["Precision", record.workload.dtype],
      [
        "Input stride",
        `${record.workload.stride} element${record.workload.stride === 1 ? "" : "s"}`,
      ],
      ["Input seed", record.workload.seed],
      ["Measurement", record.provenance?.measurement || "Not recorded"],
      ["Device", record.device.name],
    ]),
  );
  const leads = arr(record.hypotheses).slice(0, 3);
  const diagnostics = section(
    "What could explain the runtime?",
    "Diagnostic leads remain untested until a controlled experiment distinguishes them.",
    el(
      "div",
      { class: "lead-list" },
      leads.length
        ? leads.map((hypothesis, index) =>
            el(
              "button",
              {
                class: "lead-row",
                onclick: () => actions.navigate("hypotheses"),
              },
              el(
                "span",
                { class: "lead-index" },
                `H${String(index + 1).padStart(2, "0")}`,
              ),
              el(
                "div",
                {},
                el("strong", {}, hypothesis.title),
                el("span", {}, hypothesis.status || "untested"),
              ),
              icon("arrow"),
            ),
          )
        : el("p", { class: "empty-inline" }, "No hypotheses recorded."),
    ),
    el(
      "button",
      { class: "text-button", onclick: () => actions.navigate("hypotheses") },
      "Open hypothesis lab",
      icon("arrow"),
    ),
  );
  return el(
    "div",
    { class: "overview-view" },
    heading,
    scope,
    investigationFinding(record, experiment, actions),
    el(
      "div",
      { class: "investigation-grid" },
      el("div", { class: "analysis-main" }, plot, context),
      evidenceRail(record, experiment, actions),
    ),
    candidateComparison(record, experiment?.id, actions.selectCandidate),
    detail(
      `Complete intervention ledger · ${arr(record.experiments).length} attempts`,
      candidateLedger(record, experiment?.id, actions.selectCandidate),
    ),
    diagnostics,
    detail(
      "Limitations & collection provenance",
      el(
        "div",
        { class: "provenance-grid" },
        el(
          "ul",
          { class: "limit-list" },
          arr(record.diagnosis?.limitations).map((value) =>
            el("li", {}, value),
          ),
        ),
        definition([
          ["Record", record.id],
          ["Created", record.created_at],
          ["Python", record.environment?.python],
          [
            "NumPy / PyTorch",
            record.environment?.numpy ||
              record.environment?.torch ||
              "Not recorded",
          ],
          ["GPU execution", String(record.provenance?.gpu_execution === true)],
        ]),
      ),
    ),
  );
}
