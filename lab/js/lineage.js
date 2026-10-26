import { el, icon, badge } from "./dom.js";
import {
  num,
  median,
  samples,
  measurement,
  lineageRelations,
} from "./model.js";

function attemptCard(record, experiment, index, selected, onSelect) {
  const data = measurement(record, experiment);
  return el(
    "button",
    {
      class: `lineage-node ${experiment.status}`,
      "data-candidate": experiment.id,
      "aria-pressed": String(experiment.id === selected?.id),
      onclick: () => onSelect(experiment.id),
    },
    el("span", { class: "lineage-port", "aria-hidden": "true" }),
    el(
      "div",
      { class: "lineage-node-top" },
      el(
        "span",
        { class: "eyebrow" },
        `ATTEMPT ${String(index + 1).padStart(2, "0")}`,
      ),
      badge(
        experiment.status === "accepted"
          ? "Accepted · record"
          : experiment.status,
        experiment.status === "accepted" ? "positive" : "neutral",
      ),
    ),
    el("strong", {}, experiment.id),
    el(
      "span",
      {},
      experiment.intervention?.replaceAll("_", " ") || "Configuration",
    ),
    el(
      "div",
      { class: "lineage-measurement" },
      el("strong", {}, `${num(data.comparison?.speedup)}×`),
      el("span", { class: "mono" }, `${num(median(data.candidate))} µs`),
    ),
    el("small", {}, `Parent: ${experiment.parent_id ?? "baseline"}`),
  );
}
export function renderLineage(record, selected, onSelect) {
  const experiments = record.experiments;
  const topology = lineageRelations(experiments);
  const baseline = el(
    "div",
    { class: "lineage-root" },
    icon("overview"),
    el(
      "div",
      {},
      el("strong", {}, record.baseline?.name || "Reference"),
      el(
        "span",
        { class: "mono" },
        `${num(median(samples(record.baseline?.samples_us)))} µs baseline`,
      ),
    ),
    badge("REFERENCE"),
  );
  const content = el(
    "div",
    { class: `lineage ${topology.flat ? "flat" : "grouped"}` },
    baseline,
  );
  if (topology.flat) {
    content.style.setProperty("--branches", String(experiments.length));
    content.append(
      el(
        "div",
        { class: "lineage-branches" },
        experiments.map((experiment, index) =>
          attemptCard(record, experiment, index, selected, onSelect),
        ),
      ),
    );
  } else {
    content.append(
      el(
        "p",
        { class: "lineage-note" },
        "Grouped by the parent IDs in this record. No inferred connections are drawn.",
      ),
      ...topology.groups.map((group) =>
        el(
          "div",
          { class: "lineage-parent-group" },
          el(
            "div",
            { class: "lineage-parent-label" },
            icon("evolution"),
            el(
              "span",
              {},
              group.parent === null
                ? "Children of baseline"
                : `Children of ${group.parent}`,
            ),
            !group.known && badge("PARENT NOT IN RECORD", "scope"),
          ),
          el(
            "div",
            { class: "lineage-group-cards" },
            group.children.map((experiment) =>
              attemptCard(
                record,
                experiment,
                experiments.indexOf(experiment),
                selected,
                onSelect,
              ),
            ),
          ),
        ),
      ),
    );
  }
  return el(
    "div",
    {
      class: "lineage-scroll",
      tabindex: 0,
      "aria-label":
        "Experimental lineage. Scroll horizontally to inspect larger candidate pools.",
    },
    content,
  );
}
