/** Explicit DOM construction. Imported strings are always text, never markup. */
export function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value == null || value === false) continue;
    if (name === "class") node.className = value;
    else if (name.startsWith("on") && typeof value === "function")
      node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child != null && child !== false)
      node.append(
        child instanceof Node ? child : document.createTextNode(String(child)),
      );
  }
  return node;
}
export function svg(tag, attributes = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes))
    node.setAttribute(name, String(value));
  for (const child of children.flat(Infinity))
    if (child != null)
      node.append(
        child instanceof Node ? child : document.createTextNode(String(child)),
      );
  return node;
}
const paths = {
  overview: ["M3 13h4l3-8 4 14 3-8h4"],
  timeline: ["M4 5h16M4 19h16M6 9h7v3H6zM15 14h5v3h-5z"],
  bottlenecks: ["M5 19V9m7 10V4m7 15v-7M3 21h18"],
  hypotheses: [
    "M9 3h6m-5 0v6L4.7 18a2 2 0 0 0 1.7 3h11.2a2 2 0 0 0 1.7-3L14 9V3M8 14h8",
  ],
  evolution: [
    "M6 3v12a4 4 0 0 0 4 4h7M6 8h7a4 4 0 0 0 4-4",
    "M3 3h6v5H3zM15 2h5v5h-5zM16 16h5v5h-5z",
  ],
  code: ["m8 6-6 6 6 6m8-12 6 6-6 6M14 3l-4 18"],
  atlas: ["M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z"],
  research: ["M4 19V5m0 14h17M7 14l4-5 4 3 5-7"],
  arrow: ["M5 12h14m-5-5 5 5-5 5"],
  external: ["M14 3h7v7m0-7L10 14M10 3H3v18h18v-7"],
  plus: ["M12 5v14M5 12h14"],
  download: ["M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"],
  check: ["m5 12 4 4L19 6"],
  close: ["m6 6 12 12M6 18 18 6"],
  chevron: ["m9 5 7 7-7 7"],
  info: ["M12 11v6M12 7h.01", "M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0"],
  clock: ["M12 6v6l4 2", "M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0"],
  search: ["M21 21l-5-5", "M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0"],
  file: ["M14 2H5v20h14V7zM14 2v6h5M8 12h8M8 16h6"],
  lock: ["M6 10V7a6 6 0 0 1 12 0v3M4 10h16v12H4z"],
};
export function icon(name, className = "") {
  return svg(
    "svg",
    {
      viewBox: "0 0 24 24",
      width: 18,
      height: 18,
      fill: "none",
      stroke: "currentColor",
      "stroke-width": 1.5,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      "aria-hidden": "true",
      class: `icon ${className}`,
    },
    (paths[name] || paths.info).map((d) => svg("path", { d })),
  );
}
export function badge(text, tone = "neutral") {
  return el(
    "span",
    { class: `badge ${tone}` },
    el("span", { class: "badge-dot", "aria-hidden": "true" }),
    text,
  );
}
export function section(title, subtitle, content, action) {
  return el(
    "section",
    { class: "section" },
    el(
      "div",
      { class: "section-heading" },
      el("div", {}, el("h2", {}, title), subtitle && el("p", {}, subtitle)),
      action,
    ),
    content,
  );
}
export function definition(entries, className = "") {
  return el(
    "dl",
    { class: `definition ${className}` },
    entries.map(([key, value]) => [
      el("dt", {}, key),
      el("dd", {}, value ?? "Not recorded"),
    ]),
  );
}
export function emptyState(title, description, name = "file", action) {
  return el(
    "div",
    { class: "empty-state" },
    el("div", { class: "empty-icon" }, icon(name)),
    el("h2", {}, title),
    el("p", {}, description),
    action,
  );
}
export function detail(title, content, open = false) {
  return el(
    "details",
    { class: "disclosure", open: open ? "" : null },
    el("summary", {}, title, icon("chevron")),
    el("div", { class: "disclosure-body" }, content),
  );
}
