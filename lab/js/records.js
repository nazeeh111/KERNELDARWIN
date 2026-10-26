/** Structural record validation. Statistical claims are independently replayed by kd validate. */
import { arr, finite, median } from "./model.js";
const object = (value) =>
  value !== null && typeof value === "object" && !Array.isArray(value);
function validatePhase(phase, label) {
  if (!object(phase)) throw Error(`${label} requires paired timing evidence.`);
  const b = phase.baseline_us,
    c = phase.candidate_us,
    comp = phase.comparison;
  if (
    !Array.isArray(b) ||
    !Array.isArray(c) ||
    b.length !== c.length ||
    b.length < 20 ||
    [...b, ...c].some((v) => !finite(v) || v <= 0)
  )
    throw Error(`${label} requires at least 20 positive matched timing pairs.`);
  if (
    !object(comp) ||
    ![comp.speedup, comp.ci_low, comp.ci_high, comp.min_speedup ?? 1.03].every(
      finite,
    )
  )
    throw Error(`${label} needs finite comparison statistics.`);
  const threshold = comp.min_speedup ?? 1.03,
    observed = median(b) / median(c);
  if (
    comp.accepted !== true ||
    threshold < 1.03 ||
    comp.ci_low <= threshold ||
    comp.ci_high < comp.ci_low ||
    comp.n !== b.length ||
    Math.abs(comp.speedup - observed) >
      Math.max(1e-12, Math.abs(observed) * 1e-9)
  )
    throw Error(`${label} does not support the declared promotion gate.`);
}
export function validateRecord(r) {
  if (
    !object(r) ||
    r.schema_version !== 1 ||
    typeof r.id !== "string" ||
    !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(r.id) ||
    !object(r.workload)
  )
    throw Error(
      "Expected a version 1 KERNELDARWIN investigation record with a safe identifier.",
    );
  if (!["accepted", "no_improvement", "blocked", "failed"].includes(r.status))
    throw Error("The record has an unsupported investigation status.");
  if (!["cuda_run", "cpu_control", "reference"].includes(r.evidence_kind))
    throw Error(
      "The record needs an explicit CUDA, CPU control, or reference evidence label.",
    );
  const w = r.workload;
  if (
    !["sum", "softmax", "layernorm", "fused"].includes(w.family) ||
    !Array.isArray(w.shape) ||
    w.shape.length !== 2 ||
    w.shape.some((v) => !Number.isSafeInteger(v) || v < 1)
  )
    throw Error(
      "The workload must declare a supported family and two positive dimensions.",
    );
  if (
    !["float32", "float16", "bfloat16"].includes(w.dtype) ||
    !Number.isSafeInteger(w.stride) ||
    w.stride < 1 ||
    !finite(w.eps) ||
    w.eps <= 0 ||
    !Number.isSafeInteger(w.seed) ||
    w.seed < 0
  )
    throw Error(
      "The workload has invalid precision, stride, epsilon or seed metadata.",
    );
  if (!object(r.device) || !["cpu", "cuda"].includes(r.device.kind))
    throw Error("The record must declare a CPU or CUDA device.");
  if (
    w.actual_strides !== undefined &&
    (!Array.isArray(w.actual_strides) ||
      w.actual_strides.length !== 2 ||
      w.actual_strides.some((v) => !Number.isSafeInteger(v) || v < 1) ||
      w.actual_strides[1] !== w.stride)
  )
    throw Error("Actual strides must contain two positive element strides.");
  for (const key of [
    "environment",
    "provenance",
    "trace",
    "diagnosis",
    "budget",
    "selection",
    "baseline",
    "compiled_baseline",
  ])
    if (r[key] !== undefined && !object(r[key]))
      throw Error(`${key} must be an object.`);
  if (
    r.evidence_kind === "cuda_run" &&
    (r.device.kind !== "cuda" || r.provenance?.gpu_execution !== true)
  )
    throw Error(
      "A CUDA evidence label requires a CUDA device and explicit GPU execution provenance.",
    );
  if (r.evidence_kind === "cpu_control" && r.device.kind !== "cpu")
    throw Error("A CPU control must declare a CPU device.");
  if (r.evidence_kind === "reference" && r.status === "accepted")
    throw Error("A reference record cannot claim an accepted optimization.");
  for (const key of ["experiments", "hypotheses"])
    if (
      r[key] !== undefined &&
      (!Array.isArray(r[key]) || r[key].some((x) => !object(x)))
    )
      throw Error(`${key} must be an array of objects.`);
  if (
    r.trace?.events !== undefined &&
    (!Array.isArray(r.trace.events) || r.trace.events.some((x) => !object(x)))
  )
    throw Error("Trace events must be an array of objects.");
  const checkSamples = (a, label) => {
    if (
      a !== undefined &&
      (!Array.isArray(a) ||
        a.length > 10000 ||
        a.some((v) => !finite(v) || v <= 0))
    )
      throw Error(`${label} must be positive finite timing samples.`);
  };
  checkSamples(r.baseline?.samples_us, "Baseline samples");
  checkSamples(r.compiled_baseline?.samples_us, "Compiled baseline samples");
  for (const hypothesis of arr(r.hypotheses)) {
    for (const key of ["id", "title", "status", "experiment", "falsifier"]) {
      if (hypothesis[key] !== undefined && typeof hypothesis[key] !== "string")
        throw Error(`Hypothesis ${key} must be text.`);
    }
    for (const key of ["evidence", "missing_evidence"]) {
      if (hypothesis[key] !== undefined && !Array.isArray(hypothesis[key]))
        throw Error(`Hypothesis ${key} must be an array.`);
    }
  }
  const ids = new Set();
  let winner = null;
  for (const e of arr(r.experiments)) {
    if (typeof e.id !== "string" || !e.id || ids.has(e.id))
      throw Error("Each experiment needs a unique string identifier.");
    ids.add(e.id);
    for (const key of ["intervention", "error", "parent_id"]) {
      if (e[key] != null && typeof e[key] !== "string")
        throw Error(`Experiment ${key} must be text.`);
    }
    if (!["accepted", "rejected", "invalid", "failed"].includes(e.status))
      throw Error("Invalid experiment decision status.");

    for (const key of [
      "config",
      "correctness",
      "timing",
      "confirmation",
      "source",
      "safety",
    ])
      if (e[key] !== undefined && !object(e[key]))
        throw Error(`Experiment ${key} must be an object.`);
    if (
      e.config !== undefined &&
      (!Number.isSafeInteger(e.config.block_size) ||
        e.config.block_size < 1 ||
        !Number.isSafeInteger(e.config.num_warps) ||
        e.config.num_warps < 1)
    )
      throw Error(
        "Candidate configuration requires positive integer block size and warp count.",
      );
    if (
      e.source &&
      ["baseline", "candidate"].some(
        (key) =>
          e.source[key] !== undefined && typeof e.source[key] !== "string",
      )
    )
      throw Error("Candidate sources must be text.");
    checkSamples(e.timing?.baseline_us, "Candidate baseline samples");
    checkSamples(e.timing?.candidate_us, "Candidate samples");
    if (
      e.timing?.baseline_us !== undefined ||
      e.timing?.candidate_us !== undefined
    ) {
      if (
        !Array.isArray(e.timing.baseline_us) ||
        !Array.isArray(e.timing.candidate_us) ||
        e.timing.baseline_us.length !== e.timing.candidate_us.length
      )
        throw Error(
          "Candidate timings must contain matched baseline/candidate pairs.",
        );
    }

    if (
      e.status === "accepted" &&
      (e.id !== r.winner_id ||
        r.status !== "accepted" ||
        e.correctness?.passed !== true ||
        e.timing?.comparison?.accepted !== true)
    )
      throw Error(
        "An accepted experiment requires passing correctness and performance gates.",
      );
    if (e.id === r.winner_id) winner = e;
  }
  if (r.winner_id != null && (!winner || winner.status !== "accepted"))
    throw Error("The winner must identify an accepted experiment.");
  if (r.status === "accepted" && !winner)
    throw Error("An accepted investigation requires a winner.");
  if (winner) {
    validatePhase(winner.timing, "Discovery");
    validatePhase(winner.confirmation, "Independent confirmation");
    const confirmation = winner.confirmation;
    if (confirmation.correctness_passed !== true)
      throw Error(
        "Independent confirmation needs a recorded correctness pass.",
      );
    if (
      !Number.isSafeInteger(confirmation.seed) ||
      confirmation.seed === w.seed ||
      arr(r.experiments).some(
        (experiment) =>
          experiment.timing?.comparison?.seed === confirmation.seed,
      )
    )
      throw Error("Independent confirmation requires a different random seed.");
    if (
      !Array.isArray(confirmation.actual_strides) ||
      confirmation.actual_strides.length !== 2 ||
      confirmation.actual_strides.some(
        (v) => !Number.isSafeInteger(v) || v < 1,
      ) ||
      JSON.stringify(confirmation.actual_strides) !==
        JSON.stringify(w.actual_strides)
    )
      throw Error(
        "Independent confirmation strides must match the declared workload.",
      );
  }
  const check = (value, depth = 0) => {
    if (depth > 40) throw Error("The record is nested too deeply.");
    if (typeof value === "number" && !Number.isFinite(value))
      throw Error("The record contains a non-finite number.");
    if (Array.isArray(value)) value.forEach((x) => check(x, depth + 1));
    else if (value && typeof value === "object")
      Object.values(value).forEach((x) => check(x, depth + 1));
  };
  check(r);
  return r;
}
