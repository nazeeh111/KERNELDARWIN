import { readLocation, locationHash } from "./js/evidence-model.js";
import { installCommands } from "./js/commands.js";
import { downloadText } from "./js/download.js";
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
const initialLocation = readLocation(location.hash);
const state = {
  record: null,
  records: [],
  evaluation: null,
  chosen: null,
  view: initialLocation.view,
};
let importSequence = 0;
const bundledRecords = new Map();

function syncLocation(push = false) {
  const hash = locationHash(state.view, state.record?.id, state.chosen);
  if (location.hash !== hash)
    history[push ? "pushState" : "replaceState"](null, "", hash);
}
function focusView() {
  window.scrollTo(0, 0);
  $("main").focus({ preventScroll: true });
}

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
  state.view = view;
  syncLocation(true);
  render();
  focusView();
}
function render(animate = true) {
  const record = state.record;
  const route = routeMap.get(state.view);
  const openedDetails = new Set(
    [...$("view").querySelectorAll("details[open]")].map(
      (node) => node.querySelector("summary")?.textContent,
    ),
  );
  const focusedCandidate = document.activeElement?.dataset?.candidate;
  const focusedRecord = document.activeElement?.dataset?.record;
  const focusedSelect =
    document.activeElement?.id === "record-select"
      ? "#record-select"
      : document.activeElement?.matches(".code-selector select")
        ? ".code-selector select"
        : null;

  const focusedKind = [
    "lineage-node",
    "candidate-button",
    "comparison-row",
  ].find((name) => document.activeElement?.classList.contains(name));
  // Build the next view before changing the DOM so a rejected import preserves it.
  const next = route[2](state, actions);
  $("breadcrumb").textContent = route[1];
  document.title = `${route[1]} / KERNELDARWIN LAB`;
  for (const link of document.querySelectorAll(".nav-link")) {
    link.href = locationHash(link.dataset.view, record?.id, state.chosen);
    if (link.dataset.view === state.view)
      link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  $("download").disabled = !record;
  $("share").disabled = !record;
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
  for (const node of next.querySelectorAll("details"))
    if (openedDetails.has(node.querySelector("summary")?.textContent))
      node.open = true;
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
function activate(
  record,
  announce = false,
  chosen = null,
  updateHistory = true,
) {
  validateRecord(record);
  const previous = {
    record: state.record,
    chosen: state.chosen,
    records: state.records,
  };
  const records = state.records.filter((item) => item.id !== record.id);
  const index = state.records.findIndex((item) => item.id === record.id);
  records.splice(index < 0 ? records.length : index, 0, record);
  Object.assign(state, {
    record,
    chosen: selectedExperiment(record, chosen)?.id || null,
    records,
  });
  try {
    render();
  } catch (error) {
    Object.assign(state, previous);
    throw error;
  }
  if (updateHistory) syncLocation(true);
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
    syncLocation(true);
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
    if (sequence === importSequence) {
      const requested = readLocation(location.hash);
      activate(
        record,
        true,
        requested.record === record.id ? requested.candidate : null,
      );
    }
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
  downloadText(
    JSON.stringify(state.record, null, 2) + "\n",
    `${state.record.id}.json`,
    "application/json",
  );
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
  link.addEventListener("click", (event) => {
    if (
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey
    )
      return;
    event.preventDefault();
    navigate(view);
  });
  $("navigation").append(link);
}
$("import").prepend(icon("plus"));
$("download").prepend(icon("download"));
$("import").addEventListener("click", actions.importRecord);
$("file").addEventListener("change", () => importFile($("file").files[0]));
$("download").addEventListener("click", downloadRecord);
installCommands({
  trigger: $("commands"),
  routes,
  getState: () => state,
  actions,
});
$("share").addEventListener("click", () => {
  if (!state.record) return;
  syncLocation(false);
  $("share-copy-status").textContent = "";
  $("selection-url").value = location.href;
  $("share-note").textContent =
    bundledRecords.get(state.record.id) === state.record
      ? "This link opens the bundled investigation, candidate and view. Measurements remain recorded evidence."
      : "This record was imported locally. The link contains its ID only. The recipient must import the same JSON record; no record contents are uploaded or stored.";
  $("share-dialog").showModal();
  $("selection-url").focus();
  $("selection-url").select();
});
$("share-close").addEventListener("click", () => $("share-dialog").close());
$("copy-selection").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("selection-url").value);
    $("share-copy-status").textContent = "Selection link copied.";
  } catch {
    $("share-copy-status").textContent =
      "Clipboard access was unavailable. Select the link and copy it with your browser's standard controls.";
  }
});
window.addEventListener("hashchange", () => {
  const desired = readLocation(location.hash);
  state.view = desired.view;
  const record =
    state.records.find((item) => item.id === desired.record) || state.record;
  if (record) activate(record, false, desired.candidate, false);
  else render();
  if (desired.record && record?.id !== desired.record)
    feedback(
      "This investigation is not in the local library. Import its JSON record to open the shared selection.",
      true,
    );
  focusView();
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
    for (const record of records) bundledRecords.set(record.id, record);
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
    if (!state.record && state.records.length) {
      const desired = readLocation(location.hash);
      const target =
        state.records.find((item) => item.id === desired.record) ||
        state.records[0];
      activate(target, false, desired.candidate, false);
      if (desired.record && target.id !== desired.record)
        feedback(
          "Shared investigation not found. Import the original JSON record to inspect it. The bundled reference is shown meanwhile.",
          true,
        );
      else syncLocation(false);
    } else render(false);
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
