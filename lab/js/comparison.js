import { el, svg, icon, section, badge } from "./dom.js";
import { num, intervalLabel, median } from "./model.js";
import {
  comparisonRows,
  comparisonDomain,
  resultNarrative,
} from "./evidence-model.js";

export function investigationFinding(record, experiment, actions) {
  const finding = resultNarrative(record, experiment);
  return el(
    "div",
    { class: `finding ${finding.tone}` },
    el(
      "div",
      { class: "finding-symbol", "aria-hidden": "true" },
      icon(finding.tone === "positive" ? "check" : "info"),
    ),
    el(
      "div",
      {},
      el("span", { class: "eyebrow" }, "THE RESULT"),
      el("h2", {}, finding.title),
      el("p", {}, finding.text),
    ),
    el(
      "button",
      { class: "text-button", onclick: () => actions.navigate("code") },
      "Inspect the change",
      icon("arrow"),
    ),
  );
}

export function candidateComparison(record, selectedId, onSelect) {
  const rows = comparisonRows(record);
  if (!rows.length) return null;
  const [low, high] = comparisonDomain(rows);
  const selected =
    rows.find((row) => row.experiment.id === selectedId) || rows[0];
  const x = (value) => 12 + ((value - low) / (high - low)) * 576;
  const referenceX = x(1);
  const axis = svg("svg", {
    viewBox: "0 0 600 32",
    class: "comparison-axis",
    role: "img",
    "aria-label":
      "Speedup relative to each matched reference. One times means equal median latency.",
  });
  for (let index = 0; index <= 4; index++) {
    const value = low + ((high - low) * index) / 4;
    axis.append(
      svg(
        "text",
        {
          x: x(value),
          y: 14,
          "text-anchor": index === 0 ? "start" : index === 4 ? "end" : "middle",
          class: "chart-axis",
        },
        `${num(value, 2)}×`,
      ),
    );
  }
  const marker = (row) => {
    const chart = svg("svg", {
      viewBox: "0 0 600 38",
      class: "interval-chart",
      "aria-hidden": "true",
    });
    chart.append(
      svg("line", { x1: 12, x2: 588, y1: 19, y2: 19, class: "interval-track" }),
      svg("line", {
        x1: referenceX,
        x2: referenceX,
        y1: 0,
        y2: 38,
        class: "reference-rule",
      }),
    );
    if (row.interval) {
      const [a, b] = row.interval.map(x);
      chart.append(
        svg("line", {
          x1: a,
          x2: b,
          y1: 19,
          y2: 19,
          class: "interval-whisker",
        }),
        ...[a, b].map((p) =>
          svg("line", { x1: p, x2: p, y1: 13, y2: 25, class: "interval-cap" }),
        ),
      );
    }
    if (row.ratio > 0)
      chart.append(
        svg("circle", {
          cx: x(row.ratio),
          cy: 19,
          r: 4.5,
          class: "interval-point",
        }),
      );
    return chart;
  };
  return section(
    "The whole search, in perspective.",
    "A common 1× reference. Each interval belongs to its own matched measurement; phases remain separate.",
    el(
      "div",
      { class: "comparison-content" },
      el(
        "div",
        { class: "comparison-table-head" },
        el("span", {}, "ATTEMPT / PHASE"),
        axis,
        el("span", {}, "RATIO / INTERVAL"),
      ),
      rows.map((row) =>
        el(
          "button",
          {
            class: "comparison-row",
            "data-candidate": row.experiment.id,
            "aria-pressed": String(row.experiment.id === selectedId),
            "aria-label": `${row.experiment.id}, ${row.phase}, ${row.ratio == null ? "no timing" : num(row.ratio) + " times"}, ${row.interval ? intervalLabel(row.comparison) + " " + row.interval.map((v) => num(v)).join(" to ") : "no interval"}, ${row.experiment.status}. Inspect evidence`,
            onclick: () => onSelect(row.experiment.id),
          },
          el(
            "span",
            { class: "comparison-name" },
            el(
              "span",
              { class: "attempt-number" },
              String(row.index + 1).padStart(2, "0"),
            ),
            el(
              "span",
              {},
              el("strong", {}, row.experiment.id),
              el(
                "small",
                {},
                row.phase === "Independent confirmation"
                  ? "Confirmation"
                  : "Discovery",
              ),
            ),
          ),
          marker(row),
          el(
            "span",
            { class: "comparison-value" },
            el("strong", {}, row.ratio == null ? "—" : `${num(row.ratio)}×`),
            el(
              "small",
              {},
              row.interval
                ? `${num(row.interval[0])}–${num(row.interval[1])}`
                : "No interval",
            ),
          ),
          el(
            "span",
            { class: "comparison-status" },
            row.accepted ? icon("check") : icon("chevron"),
          ),
        ),
      ),
      el(
        "div",
        { class: "comparison-selection" },
        el(
          "div",
          {},
          el(
            "span",
            { class: "eyebrow" },
            `SELECTED / ${selected.experiment.id}`,
          ),
          el(
            "p",
            {},
            `${num(median(selected.baseline))} µs reference → ${num(median(selected.candidate))} µs candidate`,
            el(
              "small",
              {},
              `${selected.candidate.length} matched pairs · ${selected.phase.toLowerCase()} · ${selected.experiment.status}`,
            ),
          ),
        ),
        el(
          "button",
          {
            class: "text-button",
            onclick: () => {
              const plot = document.querySelector(".plot-canvas");
              if (plot) {
                plot.tabIndex = -1;
                plot.focus({ preventScroll: true });
                plot.scrollIntoView({ block: "center", behavior: "instant" });
              }
            },
          },
          "Inspect samples",
          icon("arrow"),
        ),
      ),
      el(
        "div",
        { class: "comparison-note" },
        el(
          "span",
          {},
          el("i", { class: "reference-tick" }),
          "1× = no median change",
        ),
        el(
          "span",
          {},
          "Intervals are record-reported. A ratio alone does not satisfy the promotion gate.",
        ),
      ),
    ),
    badge(`${rows.length} ATTEMPTS`, "neutral"),
  );
}
