const byId = (id) => document.getElementById(id);
const setText = (id, value) => { byId(id).textContent = value; };
const acceptedStatuses = new Set(["confirmed"]);
const guardedExamples = {
  matched: ["Eligible for a plan rule", "A candidate can run only when the installed artifact passed numerical checks and independent performance confirmation. Eligibility alone is not a speedup claim."],
  shape: ["Compiled MLX fallback", "Only an exact query-count, document-count and width tuple named in the installed scoring plan can select a candidate. An unfamiliar tuple uses the complete compiled library query; runtime does not search for a replacement."],
  dtype: ["Compiled MLX fallback", "The authored candidates accept float32. Float16 and bfloat16 use stable float32 accumulation in the compiled library operation, then return the original dtype."],
  identity: ["Plan rejected; library fallback", "The installed source hashes, MLX version, operating system, compiler settings, and device identity must match the artifact. A stale or malformed plan cannot select a candidate."],
  strides: ["Read actual device strides", "An exact-shape float32 rule covers MLX layouts. The Metal kernel reads the input strides directly. The adapter does not request an implicit contiguous copy."],
  autodiff: ["Explicit error", "The callable supports inference only. A request for another mode raises an error. Use ordinary MLX library operations when you need automatic differentiation."],
};

byId("guard-example").addEventListener("change", (event) => {
  const example = guardedExamples[event.target.value];
  if (!example) return;
  setText("guard-decision", example[0]);
  setText("guard-explanation", example[1]);
});

byId("consumer").addEventListener("change", (event) => {
  const vectors = event.target.value === "vectors";
  setText("run-code", vectors ? "kd workload search VECTORS.npz --k 5" : "kd workload demo");
  setText("consumer-help", vectors
    ? "Prepare a NumPy archive with documents and queries arrays. Use the same float32 or float16 dtype, two dimensions, and matching feature widths."
    : "Start with the bundled corpus. Results are lexical matches from hashed token counts.");
  setText("run-note", vectors
    ? "Replace VECTORS.npz with your local file. Uses a compatible bundled artifact if present, otherwise the compiled MLX fallback. Override with --artifact PATH."
    : "Uses a compatible bundled artifact if present, otherwise the compiled MLX fallback. Override with --artifact PATH.");
});

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const source = byId(button.dataset.copy);
    if (!source) return;
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(source.textContent);
      setText("copy-feedback", "Copied. Paste the command into your local environment.");
      button.textContent = "Copied";
      window.setTimeout(() => { button.textContent = button.dataset.copy === "install-code" ? "Copy commands" : button.dataset.copy === "python-code" ? "Copy Python" : "Copy command"; }, 1800);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(source);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
      setText("copy-feedback", "Clipboard access is unavailable. The command is selected; use your system Copy shortcut.");
    }
  });
});

function safeLink(id, value) {
  const element = byId(id);
  element.hidden = true;
  element.removeAttribute("href");
  if (typeof value !== "string" || !value.trim()) return false;
  try {
    const url = new URL(value, document.baseURI);
    if (!["http:", "https:"].includes(url.protocol)) return false;
    element.href = url.href;
    element.hidden = false;
    return true;
  } catch { return false; }
}

function finitePositive(value) { return typeof value === "number" && Number.isFinite(value) && value > 0; }

function metric(prefix, data, pending) {
  setText(`${prefix}-value`, "Pending");
  setText(`${prefix}-ci`, "No accepted confidence interval.");
  if (pending || !data || !finitePositive(data.speedup)) return;
  setText(`${prefix}-value`, `${data.speedup.toFixed(2)}×`);
  if (Array.isArray(data.ci) && data.ci.length === 2 && data.ci.every(finitePositive) && data.ci[0] <= data.ci[1]) {
    setText(`${prefix}-ci`, `Reported interval: ${data.ci[0].toFixed(2)}–${data.ci[1].toFixed(2)}×`);
  } else {
    setText(`${prefix}-ci`, "Confidence interval not supplied.");
  }
}

function renderEvidence(data) {
  if (!data || Array.isArray(data) || typeof data !== "object" || typeof data.status !== "string" || typeof data.headline !== "string" || typeof data.summary !== "string") {
    throw new Error("Evidence file does not match the workload schema");
  }
  const pending = !acceptedStatuses.has(data.status.toLowerCase());
  const measuredNull = data.status === "measured_null";
  setText("evidence-headline", data.headline);
  setText("evidence-summary", data.summary);
  setText("evidence-status", data.status.replaceAll("_", " "));
  setText("measurement-device", typeof data.device === "string" ? data.device : "Device details not supplied");
  if (typeof data.version === "string") setText("version-label", `WORKLOAD ${data.version}`);
  safeLink("evidence-link", data.evidence_url);
  const artifact = !pending && safeLink("artifact-link", data.artifact_url);
  if (!artifact) safeLink("artifact-link", null);
  setText("artifact-note", artifact ? "Inspect the manifest and Metal files before installing. A mismatch selects the compiled library fallback." : "A confirmed artifact is not available yet. The fallback remains usable.");
  metric("normalization", data.normalization, pending);
  metric("application", data.application, pending);
  if (measuredNull) {
    for (const prefix of ["normalization", "application"]) {
      setText(`${prefix}-value`, "Not confirmed");
      setText(`${prefix}-ci`, "The frozen three-session criterion failed.");
    }
  }
  setText("methods-caption", measuredNull
    ? "Descriptive mean per four-task sequence across 120 samples. These costs do not establish a speedup; inspect the independent session intervals."
    : "Reported comparison methods; missing costs are pending.");
  const tbody = byId("methods-body");
  tbody.replaceChildren();
  const methods = Array.isArray(data.methods) ? data.methods.filter((method) => method && typeof method.name === "string").slice(0, 30) : [];
  if (methods.length) {
    methods.forEach((method) => {
      const row = document.createElement("tr");
      const cost = measuredNull ? method.observed_cost_ms : !pending ? method.cost_ms : null;
      [method.name, typeof cost === "number" && Number.isFinite(cost) && cost >= 0 ? `${cost.toFixed(3)} ms` : "Pending", typeof method.status === "string" ? method.status : "Not supplied"].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      tbody.append(row);
    });
  } else {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 3;
    cell.textContent = "Method results await accepted measurements.";
    row.append(cell);
    tbody.append(row);
  }
  if (Array.isArray(data.limits) && data.limits.some((value) => typeof value === "string")) {
    const list = byId("evidence-limits");
    list.replaceChildren();
    data.limits.filter((value) => typeof value === "string").slice(0, 30).forEach((value) => {
      const item = document.createElement("li");
      item.textContent = value;
      list.append(item);
    });
  }
}

let loading = false;
async function loadEvidence() {
  if (loading) return;
  loading = true;
  byId("evidence").setAttribute("aria-busy", "true");
  byId("retry-evidence").hidden = true;
  setText("data-feedback", "");
  setText("evidence-status", "Loading evidence");
  try {
    const response = await fetch("data/workload.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Evidence request failed (${response.status})`);
    renderEvidence(await response.json());
  } catch {
    setText("evidence-status", "Evidence unavailable");
    setText("evidence-headline", "Performance evidence could not be loaded");
    setText("evidence-summary", "The local install and search instructions remain available. No timing or artifact claim is shown.");
    setText("data-feedback", "Serve this page with kd lab, then retry. If the problem persists, inspect lab/data/workload.json in the installed package.");
    metric("normalization", null, true);
    metric("application", null, true);
    safeLink("artifact-link", null);
    safeLink("evidence-link", null);
    byId("retry-evidence").hidden = false;
  } finally {
    byId("evidence").setAttribute("aria-busy", "false");
    loading = false;
  }
}
byId("retry-evidence").addEventListener("click", loadEvidence);
loadEvidence();

async function loadRecordedDemo() {
  try {
    const response = await fetch("data/workload-demo.json", { cache: "no-store" });
    if (!response.ok) throw new Error("Recording unavailable");
    const recording = await response.json();
    const probe = recording?.rows?.[0]?.result;
    if (recording.kind !== "recorded_installed_gpu_cli_execution" || recording.live_browser_compute !== false || recording.performance_claim !== false || probe?.metal_executed !== true || probe.probe_sum_squares !== 690880) throw new Error("Invalid recording");
    const queries = recording.rows.slice(1, 3);
    if (queries.length !== 2 || queries.some(row => typeof row.command !== "string" || !Array.isArray(row.result?.matches) || row.result.matches.length !== 3 || !row.result.matches.every(x => typeof x === "string") || !Array.isArray(row.result?.scores?.[0]) || row.result.scores[0].length !== 3 || !row.result.scores[0].every(x => typeof x === "number" && Number.isFinite(x)) || row.result.artifact_status !== "library_only" || row.result.fallback_calls !== 1)) throw new Error("Invalid query recording");
    setText("recorded-note", `Actual installed CLI output, recorded ${recording.recorded_at}. Metal probe passed: sum of squares = 690880. The browser does not execute GPU code.`);
    const render = () => {
      const row = queries[Number(byId("recorded-query").value)] ?? queries[0];
      const lines = row.result.matches.map((match, index) => `${row.result.scores[0][index].toFixed(6)}  ${match}`);
      setText("recorded-output", `$ ${row.command}\n\n${lines.join("\n")}\n\nartifact_status: ${row.result.artifact_status}\nfallback_calls: ${row.result.fallback_calls}`);
    };
    byId("recorded-query").disabled = false;
    byId("recorded-query").addEventListener("change", render);
    render();
  } catch {
    setText("recorded-note", "The recording could not be loaded. Run kd workload demo locally to execute the search.");
    setText("recorded-output", "Recording unavailable. Local install and search instructions remain available.");
  }
}
loadRecordedDemo();
