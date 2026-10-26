/** KERNELDARWIN workbench controller. No compute backend and no source evaluation. */
import { el, icon, badge } from "./js/dom.js";
import {
  arr,
  evidenceLabel,
  recordName,
  selectedExperiment,
} from "./js/model.js";
import { validateRecord } from "./js/records.js";
import { reveal } from "./js/motion.js";
import { renderOverview } from "./js/overview.js";
import {
  renderTimeline,
  renderBottlenecks,
  renderHypotheses,
  renderEvolution,
  renderCode,
  renderAtlas,
  renderResearch,
} from "./js/inspection.js";

const $ = (id) => document.getElementById(id);
const routes = [
  ["overview", "Overview", renderOverview],
  ["timeline", "GPU timeline", renderTimeline],
  ["bottlenecks", "Bottlenecks", renderBottlenecks],
  ["hypotheses", "Hypotheses", renderHypotheses],
  ["evolution", "Evolution", renderEvolution],
  ["code", "Implementation", renderCode],
  ["atlas", "Performance atlas", renderAtlas],
  ["research", "Research", renderResearch],
];
const routeMap = new Map(routes.map((route) => [route[0], route]));
const routeFromURL = () =>
  routeMap.has(location.hash.slice(1)) ? location.hash.slice(1) : "overview";
const state = {
  record: null,
  records: [],
  evaluation: null,
  chosen: null,
  view: routeFromURL(),
};
let importSequence = 0;

function feedback(message, error = false) {
  $("feedback").hidden = false;
  $("feedback").className = `feedback ${error ? "error" : "success"}`;
  $("feedback").replaceChildren(
    icon(error ? "info" : "check"),
    el("span", {}, message),
    el(
      "button",
      {
        class: "icon-button",
        "aria-label": "Dismiss notification",
        onclick: () => {
          $("feedback").hidden = true;
        },
      },
      icon("close"),
    ),
  );
}
function navigate(view) {
  if (!routeMap.has(view)) return;
  if (state.view === view) return;
  location.hash = view;
}
function render(animate = true) {
  const record = state.record;
  const route = routeMap.get(state.view);
  const focusedCandidate = document.activeElement?.dataset?.candidate;
  const focusedRecord = document.activeElement?.dataset?.record;
  const focusedSelect =
    document.activeElement?.id === "record-select"
      ? "#record-select"
      : document.activeElement?.matches(".code-selector select")
        ? ".code-selector select"
        : null;

  const focusedKind = document.activeElement?.classList.contains("lineage-node")
    ? "lineage-node"
    : "candidate-button";
  // Build the next view before changing the DOM so a rejected import preserves it.
  const next = route[2](state, actions);
  $("breadcrumb").textContent = route[1];
  document.title = `${route[1]} / KERNELDARWIN LAB`;
  for (const link of document.querySelectorAll(".nav-link")) {
    if (link.dataset.view === state.view)
      link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  $("download").disabled = !record;
  $("record-count").textContent = String(state.records.length).padStart(2, "0");
  const select = el(
    "select",
    {
      id: "record-select",
      "aria-label": "Active investigation",
      onchange: (event) => {
        const selected = state.records.find(
          (item) => item.id === event.target.value,
        );
        if (selected) activate(selected);
      },
    },
    state.records.map((item) =>
      el("option", { value: item.id }, recordName(item)),
    ),
  );
  if (record) select.value = record.id;
  $("record-strip").replaceChildren(
    el(
      "div",
      { class: "record-picker" },
      icon("file"),
      el("label", { for: "record-select" }, "INVESTIGATION"),
      select,
    ),
    el(
      "div",
      { class: "record-meta" },
      badge(
        evidenceLabel(record),
        record?.evidence_kind === "cpu_control" ? "scope" : "neutral",
      ),
      el("span", { class: "mono" }, record?.workload?.dtype || "No precision"),
      el(
        "span",
        { class: "record-date" },
        record?.created_at?.slice(0, 10) || "No date",
      ),
    ),
  );
  $("recent-records").replaceChildren(
    ...state.records.map((item) =>
      el(
        "button",
        {
          class: "recent-record",
          "data-record": item.id,
          "aria-current": item.id === record?.id ? "true" : null,
          onclick: () => activate(item),
        },
        el("span", {
          class:
            "recent-status " + (item.status === "accepted" ? "accepted" : ""),
        }),
        el(
          "span",
          {},
          item.workload.family,
          el("small", {}, item.workload.shape.join(" × ")),
        ),
        icon("chevron"),
      ),
    ),
  );
  $("view").replaceChildren(next);
  if (focusedSelect)
    document.querySelector(focusedSelect)?.focus({ preventScroll: true });
  if (focusedRecord)
    document.querySelectorAll(".recent-record").forEach((button) => {
      if (button.dataset.record === focusedRecord)
        button.focus({ preventScroll: true });
    });
  if (animate) {
    const target = focusedCandidate
      ? next.querySelector(".plot-canvas, .selected-attempt") || next
      : next;
    reveal(target, Boolean(focusedCandidate));
  }
  if (focusedCandidate)
    document.querySelectorAll("[data-candidate]").forEach((button) => {
      if (
        button.dataset.candidate === focusedCandidate &&
        button.classList.contains(focusedKind)
      )
        button.focus({ preventScroll: true });
    });
}
function activate(record, announce = false) {
  validateRecord(record);
  const previous = {
    record: state.record,
    chosen: state.chosen,
    records: state.records,
  };
  const records = state.records.filter((item) => item.id !== record.id);
  const index = state.records.findIndex((item) => item.id === record.id);
  records.splice(index < 0 ? records.length : index, 0, record);
  Object.assign(state, { record, chosen: null, records });
  try {
    render();
  } catch (error) {
    Object.assign(state, previous);
    throw error;
  }
  if (announce)
    feedback(
      `Loaded ${recordName(record)}. Record claims remain reported; kd validate recomputes statistical gates.`,
    );
}
const actions = {
  navigate,
  activate: (record) => {
    activate(record);
    navigate("overview");
  },
  importRecord: () => $("file").click(),
  selectCandidate: (id) => {
    if (!state.record?.experiments.some((item) => item.id === id)) return;
    state.chosen = id;
    render();
  },
};
async function importFile(file) {
  if (!file) return;
  const sequence = ++importSequence;
  try {
    if (file.size > 20 * 1024 * 1024)
      throw Error("The 20 MB record limit was exceeded.");
    const record = validateRecord(JSON.parse(await file.text()));
    if (sequence === importSequence) activate(record, true);
  } catch (error) {
    if (sequence === importSequence)
      feedback(
        `Could not import record: ${error.message} The previous investigation is preserved.`,
        true,
      );
  } finally {
    $("file").value = "";
  }
}
function downloadRecord() {
  if (!state.record) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(state.record, null, 2) + "\n"], {
      type: "application/json",
    }),
  );
  const link = el("a", { href: url, download: `${state.record.id}.json` });
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
for (const [view, label] of routes) {
  const link = el(
    "a",
    {
      href: `#${view}`,
      class: "nav-link",
      "data-view": view,
      "aria-label": label,
      title: label,
    },
    icon(view),
    el("span", {}, label),
  );
  $("navigation").append(link);
}
$("import").prepend(icon("plus"));
$("download").prepend(icon("download"));
$("import").addEventListener("click", actions.importRecord);
$("file").addEventListener("change", () => importFile($("file").files[0]));
$("download").addEventListener("click", downloadRecord);
window.addEventListener("hashchange", () => {
  state.view = routeFromURL();
  render();
  window.scrollTo(0, 0);
  $("main").focus({ preventScroll: true });
});
// Redraw SVG geometry only when the responsive chart width changes.
let compact = innerWidth < 760;
window.addEventListener("resize", () => {
  const next = innerWidth < 760;
  if (next !== compact) {
    compact = next;
    render(false);
  }
});
let dragDepth = 0;
window.addEventListener("dragenter", (event) => {
  if (!arr([...event.dataTransfer.types]).includes("Files")) return;
  event.preventDefault();
  dragDepth++;
  $("drop-overlay").hidden = false;
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("drop-overlay").hidden = true;
});
window.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  $("drop-overlay").hidden = true;
  importFile(event.dataTransfer.files[0]);
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    $("drop-overlay").hidden = true;
    dragDepth = 0;
  }
});
render(false);

async function localJSON(path, base = new URL("data/", location.href)) {
  const url = new URL(path, base);
  if (url.origin !== location.origin)
    throw Error("External record URLs are not allowed.");
  const response = await fetch(url);
  if (!response.ok) throw Error(`HTTP ${response.status}`);
  const raw = await response.text();
  if (raw.length > 20 * 1024 * 1024)
    throw Error("Bundled file exceeds the record limit.");
  return JSON.parse(raw);
}
async function loadLibrary() {
  try {
    const manifest = await localJSON("index.json");
    const results = await Promise.allSettled(
      arr(manifest.records)
        .slice(0, 100)
        .map(async (path) => validateRecord(await localJSON(path))),
    );
    const records = results
      .filter((result) => result.status === "fulfilled")
      .map((result) => result.value);
    // Preserve an investigation imported while the library was loading.
    for (const record of records)
      if (!state.records.some((item) => item.id === record.id))
        state.records.push(record);
    let failed = results.filter(
      (result) => result.status === "rejected",
    ).length;
    if (typeof manifest.evaluation === "string") {
      try {
        const evaluation = await localJSON(manifest.evaluation);
        if (
          evaluation.schema_version !== 1 ||
          !["cpu", "cuda"].includes(evaluation.device_kind) ||
          !evaluation.splits
        )
          throw Error("Invalid evaluation schema");
        state.evaluation = evaluation;
      } catch {
        failed++;
      }
    }
    if (!state.record && state.records.length) activate(state.records[0]);
    else render(false);
    if (failed)
      feedback(
        `${failed} bundled file${failed === 1 ? "" : "s"} could not be read. You can still import a valid local record.`,
        true,
      );
  } catch (error) {
    feedback(
      location.protocol === "file:"
        ? "Serve this folder with kd lab to load bundled records. Local JSON import still works."
        : `The record library could not load (${error.message}). Import a local JSON record to continue.`,
      true,
    );
  }
}
loadLibrary();
