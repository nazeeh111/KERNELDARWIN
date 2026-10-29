import { el, icon } from "./dom.js";
import { lineDiff } from "./line-diff.js";
import { downloadText } from "./download.js";

function excessiveLines(text) {
  let count = 1;
  for (let index = 0; index < text.length; index++)
    if (text[index] === "\n" && ++count > 10000) return true;
  return false;
}
export function sourceComparison(reference, candidate, id) {
  if (typeof reference !== "string" || typeof candidate !== "string")
    return el(
      "p",
      { class: "empty-inline" },
      "Both source texts are required for a line comparison. No missing implementation is inferred.",
    );
  if (
    reference.length + candidate.length > 1000000 ||
    excessiveLines(reference) ||
    excessiveLines(candidate)
  )
    return el(
      "div",
      { class: "source-comparison" },
      el(
        "p",
        { class: "diff-disclaimer" },
        "These sources exceed the interactive comparison limit (one million total characters or 10,000 lines per source). Download the complete text for external review; no source was executed or discarded.",
      ),
      el(
        "div",
        { class: "diff-footer" },
        el(
          "button",
          {
            class: "text-button",
            onclick: () => downloadText(reference, `${id}-reference.txt`),
          },
          "Reference source",
          icon("download"),
        ),
        el(
          "button",
          {
            class: "text-button",
            onclick: () => downloadText(candidate, `${id}-candidate.txt`),
          },
          "Candidate source",
          icon("download"),
        ),
      ),
    );
  const diff = lineDiff(reference, candidate);
  const limit = 1600;
  const rows = diff.rows.slice(0, limit);
  const container = el("div", { class: "source-comparison" });
  const body = el("div", { class: "diff-body" });
  const output = el("output", {
    class: "diff-position",
    "aria-live": "polite",
  });
  let changesOnly = false,
    current = -1;
  const draw = () => {
    current = -1;
    output.textContent = "";
    body.replaceChildren(
      ...rows
        .filter((row) => !changesOnly || row.kind !== "same")
        .map((row) =>
          el(
            "div",
            {
              class: `diff-line ${row.kind}`,
              "data-changed": row.kind === "same" ? null : "true",
              tabindex: row.kind === "same" ? null : -1,
            },
            el(
              "span",
              {
                class: "diff-old",
                "aria-label":
                  row.old == null
                    ? "No reference line"
                    : `Reference line ${row.old}`,
              },
              row.old ?? "",
            ),
            el(
              "span",
              {
                class: "diff-new",
                "aria-label":
                  row.new == null
                    ? "No candidate line"
                    : `Candidate line ${row.new}`,
              },
              row.new ?? "",
            ),
            el(
              "span",
              {
                class: "diff-sign",
                "aria-label": {
                  same: "Unchanged",
                  add: "Inserted",
                  remove: "Deleted",
                }[row.kind],
              },
              { same: " ", add: "+", remove: "−" }[row.kind],
            ),
            el("code", {}, row.text.slice(0, 5000) || " "),
          ),
        ),
    );
    if (!body.childNodes.length)
      body.append(el("p", { class: "empty-inline" }, "No textual changes."));
  };
  const go = (direction) => {
    const changes = [...body.querySelectorAll('[data-changed="true"]')];
    if (!changes.length) return;
    current =
      current < 0
        ? direction > 0
          ? 0
          : changes.length - 1
        : (current + direction + changes.length) % changes.length;
    changes.forEach((node) => node.classList.remove("current-change"));
    const target = changes[current];
    target.classList.add("current-change");
    target.focus({ preventScroll: true });
    target.scrollIntoView({
      block: "nearest",
      inline: "nearest",
      behavior: "instant",
    });
    output.textContent = `Changed line ${current + 1} of ${changes.length}`;
  };
  const toggle = el(
    "button",
    {
      class: "button ghost",
      "aria-pressed": "false",
      onclick: () => {
        changesOnly = !changesOnly;
        toggle.setAttribute("aria-pressed", String(changesOnly));
        draw();
      },
    },
    "Changes only",
  );
  container.append(
    el(
      "div",
      { class: "diff-toolbar" },
      el(
        "div",
        { class: "diff-summary" },
        el("span", { class: "diff-added" }, `+${diff.additions}`),
        el("span", { class: "diff-deleted" }, `−${diff.removals}`),
        el("span", {}, "LINE DIFF"),
      ),
      el(
        "div",
        { class: "diff-controls" },
        toggle,
        el(
          "button",
          {
            class: "icon-button",
            "aria-label": "Previous changed line",
            disabled: !diff.additions && !diff.removals,
            onclick: () => go(-1),
          },
          icon("arrow", "reverse"),
        ),
        el(
          "button",
          {
            class: "icon-button",
            "aria-label": "Next changed line",
            disabled: !diff.additions && !diff.removals,
            onclick: () => go(1),
          },
          icon("arrow"),
        ),
      ),
    ),
    el(
      "div",
      { class: "diff-labels" },
      el("span", {}, "REF"),
      el("span", {}, "NEW"),
      el("span", {}, "DECLARED REFERENCE → CANDIDATE / READ ONLY"),
    ),
    el(
      "div",
      {
        class: "diff-scroll",
        tabindex: 0,
        "aria-label": "Scrollable source line comparison",
      },
      body,
    ),
    el(
      "div",
      { class: "diff-footer" },
      output,
      el(
        "button",
        {
          class: "text-button",
          onclick: () => downloadText(reference, `${id}-reference.txt`),
        },
        "Reference source",
        icon("download"),
      ),
      el(
        "button",
        {
          class: "text-button",
          onclick: () => downloadText(candidate, `${id}-candidate.txt`),
        },
        "Candidate source",
        icon("download"),
      ),
    ),
    el(
      "p",
      { class: "diff-disclaimer" },
      "Exact line comparison, not a semantic or correctness check. Imported source is never executed.",
      diff.bounded
        ? " Large input: minimal alignment skipped; changed middle displayed as replacement."
        : "",
      diff.rows.length > limit
        ? ` Preview limited to ${limit} of ${diff.rows.length} lines. Download the complete sources.`
        : "",
      rows.some((row) => row.text.length > 5000)
        ? " Lines longer than 5,000 characters are truncated in this preview."
        : "",
    ),
  );
  draw();
  return container;
}
