import { el, icon } from "./dom.js";
import { recordName } from "./model.js";
import { reveal } from "./motion.js";

/** A native modal and ordinary buttons keep focus and activation browser-owned. */
export function installCommands({ trigger, routes, getState, actions }) {
  const results = el("div", { class: "command-results" });
  const count = el("p", { class: "sr-only", role: "status" });
  const search = el("input", {
    type: "search",
    "aria-label": "Search views, investigations and candidates",
    placeholder: "Where does the evidence lead?",
    autocomplete: "off",
    spellcheck: "false",
  });
  const dialog = el(
    "dialog",
    { class: "command-dialog", "aria-labelledby": "command-title" },
    el(
      "div",
      { class: "command-dialog-heading" },
      icon("search"),
      el("h2", { id: "command-title" }, "Go to evidence"),
      el(
        "button",
        {
          class: "icon-button",
          "aria-label": "Close command search",
          onclick: () => dialog.close(),
        },
        icon("close"),
      ),
    ),
    search,
    count,
    results,
    el(
      "div",
      { class: "command-hint" },
      el("span", {}, "Search views, records or candidate IDs"),
      el("span", {}, el("kbd", {}, "esc"), " to close"),
    ),
  );
  document.body.append(dialog);
  const execute = (action) => {
    dialog.close();
    action();
  };
  function draw() {
    const state = getState(),
      query = search.value.trim().toLowerCase();
    const matches = (value) => value.toLowerCase().includes(query);
    const commands = [
      ...routes.map(([view, label]) => ({
        group: "VIEWS",
        label,
        detail: state.view === view ? "Current view" : "Open view",
        name: view,
        action: () => actions.navigate(view),
      })),
      ...state.records.map((record) => ({
        group: "INVESTIGATIONS",
        label: recordName(record),
        detail: record.evidence_kind.replaceAll("_", " "),
        name: record.id,
        action: () => actions.activate(record),
      })),
      ...(state.record?.experiments || []).map((experiment) => ({
        group: "CURRENT CANDIDATES",
        label: experiment.id,
        detail: experiment.status,
        name: experiment.intervention || "",
        action: () => {
          actions.selectCandidate(experiment.id);
          actions.navigate("overview");
        },
      })),
    ].filter((item) => matches(`${item.label} ${item.name}`));
    const nodes = [];
    let group;
    for (const command of commands.slice(0, 60)) {
      if (group !== command.group)
        nodes.push(el("h3", {}, (group = command.group)));
      nodes.push(
        el(
          "button",
          { class: "command-result", onclick: () => execute(command.action) },
          el("span", {}, command.label, el("small", {}, command.detail)),
          icon("arrow"),
        ),
      );
    }
    results.replaceChildren(...nodes);
    if (!commands.length)
      results.append(
        el(
          "p",
          { class: "empty-inline" },
          "No match. Try a view, operation or candidate ID.",
        ),
      );
    count.textContent = `${Math.min(commands.length, 60)} results${commands.length > 60 ? ", refine your search to see more" : ""}`;
  }
  function open() {
    if (dialog.open) return;
    search.value = "";
    draw();
    dialog.showModal();
    search.focus();
    reveal(dialog, true);
  }
  trigger.addEventListener("click", open);
  search.addEventListener("input", draw);
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      const rect = dialog.getBoundingClientRect();
      if (
        event.clientX < rect.left ||
        event.clientX > rect.right ||
        event.clientY < rect.top ||
        event.clientY > rect.bottom
      )
        dialog.close();
    }
  });
  dialog.addEventListener("keydown", (event) => {
    const buttons = [...results.querySelectorAll("button")];
    if (!["ArrowDown", "ArrowUp", "Enter"].includes(event.key)) return;
    if (event.key === "Enter") {
      if (event.target === search && buttons[0]) {
        event.preventDefault();
        buttons[0].click();
      }
      return;
    }
    event.preventDefault();
    const index = buttons.indexOf(document.activeElement);
    const next =
      index < 0
        ? event.key === "ArrowDown"
          ? 0
          : buttons.length - 1
        : index + (event.key === "ArrowDown" ? 1 : -1);
    if (next < 0 || next >= buttons.length) search.focus();
    else buttons[next].focus();
  });
  document.addEventListener("keydown", (event) => {
    if (
      (event.metaKey || event.ctrlKey) &&
      !event.altKey &&
      event.key.toLowerCase() === "k"
    ) {
      event.preventDefault();
      if (dialog.open) dialog.close();
      else open();
    }
  });
  return { open };
}
