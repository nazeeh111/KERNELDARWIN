import { timingCSV } from "./evidence-model.js";
import { downloadText } from "./download.js";
import { el, svg, icon } from "./dom.js";
import { histogram, median, quantile, num } from "./model.js";

const geometry = () => ({
  width: innerWidth < 760 ? Math.max(280, innerWidth - 80) : 760,
  height: 260,
  left: 32,
  right: 16,
  top: 28,
  bottom: 40,
});
function axis(width, height, left, right, top, bottom, low, high) {
  const nodes = [];
  for (let index = 0; index <= 4; index++) {
    const x = left + ((width - left - right) * index) / 4;
    nodes.push(
      svg("line", {
        x1: x,
        y1: top,
        x2: x,
        y2: height - bottom,
        class: "chart-grid",
      }),
    );
    nodes.push(
      svg(
        "text",
        { x, y: height - 15, "text-anchor": "middle", class: "chart-axis" },
        num(low + ((high - low) * index) / 4, 0),
      ),
    );
  }
  return nodes;
}
function timingDistribution(baseline, candidate) {
  const data = histogram(baseline, candidate, innerWidth < 760 ? 20 : 34);
  const { width, height, left, right, top, bottom } = geometry();
  const chart = svg("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "data-chart",
    role: "img",
    "aria-label": `Measured latency distribution. Baseline median ${num(median(baseline))} microseconds; candidate median ${num(median(candidate))} microseconds.`,
  });
  if (!data) return chart;
  const plotWidth = width - left - right,
    plotHeight = height - top - bottom;
  const x = (value) =>
    left + ((value - data.low) / (data.high - data.low)) * plotWidth;
  const max = Math.max(...data.baseline, ...data.candidate, 1);
  chart.append(
    ...axis(width, height, left, right, top, bottom, data.low, data.high),
  );
  chart.append(
    svg(
      "text",
      { x: left - 8, y: top + 7, "text-anchor": "end", class: "chart-axis" },
      max,
    ),
    svg(
      "text",
      {
        x: left - 8,
        y: height - bottom,
        "text-anchor": "end",
        class: "chart-axis",
      },
      "0",
    ),
  );
  for (const [series, counts] of [
    ["baseline", data.baseline],
    ["candidate", data.candidate],
  ]) {
    counts.forEach((count, index) => {
      const barWidth = plotWidth / counts.length;
      const barHeight = (count / max) * plotHeight;
      const bar = svg("rect", {
        x: left + index * barWidth + 1,
        y: height - bottom - barHeight,
        width: Math.max(1, barWidth - 2),
        height: barHeight,
        class: `histogram-bar ${series}`,
      });
      bar.append(
        svg(
          "title",
          {},
          `${series}: ${count} samples in ${num(data.low + (index * (data.high - data.low)) / counts.length, 1)} µs bin`,
        ),
      );
      chart.append(bar);
    });
  }
  for (const [name, list] of [
    ["baseline", baseline],
    ["candidate", candidate],
  ]) {
    if (!list.length) continue;
    const value = median(list);
    chart.append(
      svg("line", {
        x1: x(value),
        y1: top - 8,
        x2: x(value),
        y2: height - bottom,
        class: `median-line ${name}`,
      }),
    );
  }
  chart.append(
    svg("text", { x: left, y: 15, class: "chart-axis" }, "SAMPLE FREQUENCY"),
  );
  chart.append(
    svg(
      "text",
      { x: width - right, y: 15, "text-anchor": "end", class: "chart-axis" },
      "LATENCY / µs",
    ),
  );
  return chart;
}
function pairedChart(baseline, candidate, selected) {
  const { width, height, left, right, top, bottom } = geometry();
  const count = Math.min(baseline.length, candidate.length);
  const high = Math.max(...baseline, ...candidate) * 1.1;
  const x = (index) =>
    left + ((width - left - right) * index) / Math.max(1, count - 1);
  const y = (value) =>
    height - bottom - ((height - top - bottom) * value) / high;
  const chart = svg("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "data-chart pair-chart",
    role: "img",
    "aria-label":
      "Paired measurements in collection order. Use the measurement inspector below to read each timing pair.",
  });
  for (let index = 0; index <= 3; index++) {
    const value = (high * index) / 3;
    chart.append(
      svg("line", {
        x1: left,
        y1: y(value),
        x2: width - right,
        y2: y(value),
        class: "chart-grid",
      }),
    );
    chart.append(
      svg(
        "text",
        {
          x: left - 6,
          y: y(value) + 4,
          "text-anchor": "end",
          class: "chart-axis",
        },
        num(value, 0),
      ),
    );
  }
  for (const [name, list] of [
    ["baseline", baseline],
    ["candidate", candidate],
  ]) {
    const d = list
      .slice(0, count)
      .map((value, index) => `${index ? "L" : "M"}${x(index)},${y(value)}`)
      .join(" ");
    chart.append(svg("path", { d, class: `sample-line ${name}` }));
    list.slice(0, count).forEach((value, index) =>
      chart.append(
        svg("circle", {
          cx: x(index),
          cy: y(value),
          r: index === selected ? 4 : 2,
          class: `sample-dot ${name}`,
        }),
      ),
    );
  }
  chart.append(
    svg("line", {
      x1: x(selected),
      y1: top,
      x2: x(selected),
      y2: height - bottom,
      class: "inspection-line",
    }),
  );
  chart.append(
    svg("text", { x: left, y: height - 12, class: "chart-axis" }, "PAIR 01"),
  );
  chart.append(
    svg(
      "text",
      {
        x: width - right,
        y: height - 12,
        "text-anchor": "end",
        class: "chart-axis",
      },
      `PAIR ${count}`,
    ),
  );
  return chart;
}
export function latencyPlot(data) {
  const { baseline, candidate, phase } = data;
  const content = el("div", { class: "latency-plot" });
  if (!baseline.length)
    return el(
      "p",
      { class: "empty-inline" },
      "No latency measurements recorded.",
    );
  let mode = "distribution",
    selected = 0;
  const plot = el("div", { class: "plot-canvas" }),
    output = el("output", { class: "pair-readout", "aria-live": "polite" });
  const modes = el("div", {
    class: "segmented",
    "aria-label": "Latency plot view",
    role: "group",
  });
  const buttons = ["distribution", "paired"].map((value) =>
    el(
      "button",
      {
        type: "button",
        "aria-pressed": String(value === mode),
        onclick: () => {
          mode = value;
          buttons.forEach((button, index) =>
            button.setAttribute(
              "aria-pressed",
              String(["distribution", "paired"][index] === mode),
            ),
          );
          draw();
        },
      },
      value === "distribution" ? "Distribution" : "Paired samples",
    ),
  );
  modes.append(...buttons);
  const draw = () => {
    plot.replaceChildren(
      mode === "distribution"
        ? timingDistribution(baseline, candidate)
        : pairedChart(baseline, candidate, selected),
    );
    if (mode === "paired" && candidate.length) {
      output.replaceChildren(
        el("span", {}, `PAIR ${String(selected + 1).padStart(2, "0")}`),
        el("span", { class: "baseline-text" }, `${num(baseline[selected])} µs`),
        icon("arrow"),
        el(
          "span",
          { class: "candidate-text" },
          `${num(candidate[selected])} µs`,
        ),
      );
    } else output.textContent = "Individual observations. No smoothing.";
  };
  const count = Math.min(baseline.length, candidate.length);
  const slider = el("input", {
    type: "range",
    min: 0,
    max: Math.max(0, count - 1),
    value: 0,
    step: 1,
    "aria-label": "Inspect timing pair",
    "aria-valuetext": "Pair 1",
    oninput: (event) => {
      selected = Number(event.target.value);
      mode = "paired";
      buttons[0].setAttribute("aria-pressed", "false");
      buttons[1].setAttribute("aria-pressed", "true");
      event.target.setAttribute(
        "aria-valuetext",
        `Pair ${selected + 1}: baseline ${num(baseline[selected])}, candidate ${num(candidate[selected])} microseconds`,
      );
      draw();
    },
  });
  const inspect = el(
    "div",
    { class: "pair-inspector" },
    el("label", {}, "Inspect pair", slider),
    output,
  );
  content.append(
    el(
      "div",
      { class: "plot-toolbar" },
      el(
        "div",
        { class: "legend" },
        el("span", {}, el("i", { class: "baseline-swatch" }), "Reference"),
        el("span", {}, el("i", { class: "candidate-swatch" }), "Candidate"),
      ),
      modes,
    ),
    plot,
  );
  if (count) content.append(inspect);
  content.append(
    el(
      "div",
      { class: "plot-caption" },
      el("span", {}, phase),
      el(
        "span",
        {},
        `${count || baseline.length} ${count ? "timing pairs" : "reference samples"} · microseconds`,
      ),
      count
        ? el(
            "button",
            {
              class: "text-button sample-download",
              onclick: () =>
                downloadText(
                  timingCSV(data),
                  "kerneldarwin-timing-pairs.csv",
                  "text/csv",
                ),
            },
            "Raw samples",
            icon("download"),
          )
        : null,
    ),
  );
  draw();
  return content;
}
export function latencyRange(label, values, tone) {
  const lo = quantile(values, 0.05),
    mid = median(values),
    hi = quantile(values, 0.95);
  return el(
    "div",
    { class: `latency-range ${tone}` },
    el(
      "div",
      {},
      el("span", { class: "range-label" }, label),
      el("strong", {}, num(mid), el("small", {}, " µs")),
    ),
    el(
      "span",
      { class: "range-spread" },
      `P05 ${num(lo, 1)} / P95 ${num(hi, 1)}`,
    ),
  );
}
export function policyChart(rows, aliases) {
  const max = Math.max(0, ...rows.map((row) => row.mean_best_reward));
  const min = Math.min(0, ...rows.map((row) => row.mean_best_reward));
  const range = max - min || 1;
  return el(
    "div",
    { class: "policy-chart" },
    rows.map((row) => {
      const track = el("div", { class: "policy-track" }),
        bar = el("div", {
          class: `policy-bar ${row.policy === "learned_ucb" ? "learned" : ""}`,
        });
      bar.style.left = `${((Math.min(0, row.mean_best_reward) - min) / range) * 100}%`;
      bar.style.width = `${(Math.abs(row.mean_best_reward) / range) * 100}%`;
      track.append(bar);
      return el(
        "div",
        { class: "policy-row" },
        el("span", {}, aliases[row.policy] || row.policy),
        track,
        el("span", { class: "mono" }, num(row.mean_best_reward, 4)),
      );
    }),
  );
}
