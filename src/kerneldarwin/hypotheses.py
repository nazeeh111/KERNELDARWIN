"""Falsifiable hypotheses; analytical hints are not causal diagnoses."""
from __future__ import annotations


def diagnose(spec: dict, baseline: dict, device: dict, counters: dict | None = None) -> list[dict]:
    counters = counters or {}
    rows, columns = spec["shape"]
    width = 4 if spec.get("dtype", "float32") == "float32" else 2
    total_bytes = rows * columns * width
    median = baseline.get("summary", baseline).get("median_us")
    timing = f"Measured baseline median is {median:.4g} us." if isinstance(median, (int, float)) else "Baseline timing is unavailable."
    stride = spec.get("stride", 1)
    common = ["Timing alone does not identify a bottleneck.", "Peak bandwidth is not measured sustained bandwidth."]
    results = [{
        "id": "memory-traffic", "title": "Memory traffic may limit throughput", "status": "untested",
        "evidence": [f"Input contains {rows * columns} elements ({total_bytes} logical bytes).", timing],
        "missing_evidence": ["Measured DRAM bytes and sustained bandwidth", "Cache behavior", *common],
        "falsifier": "A variant reducing memory traffic fails to improve independently measured latency while compute-only variants improve it.",
        "experiment": "Compare fused versus eager execution at fixed shape/dtype; inspect DRAM traffic with Nsight Compute when available.",
        "confidence": "low"}, {
        "id": "launch-overhead", "title": "Launch overhead may dominate small workloads", "status": "untested",
        "evidence": [f"Workload has {rows} rows and {columns} columns.", timing],
        "missing_evidence": ["Host/device trace with launch gaps", "Normal-launch versus CUDA-graph timings"],
        "falsifier": "Increasing useful work substantially raises runtime proportionally and graph replay provides little change.",
        "experiment": "Sweep shapes at fixed configuration and compare normal-launch versus graph replay timing separately.",
        "confidence": "low"}]
    if stride > 1:
        results.append({"id": "strided-access", "title": "Strided reads may waste memory transactions", "status": "untested",
                        "evidence": [f"Declared last-dimension element stride is {stride}."],
                        "missing_evidence": ["Memory transaction efficiency", "Copy cost for contiguous layout"],
                        "falsifier": "A contiguous-input control, including any copy cost, has no repeatable advantage.",
                        "experiment": "Compare strided input with an independently allocated contiguous equivalent; report copy cost separately.",
                        "confidence": "low"})
    if spec["family"] in ("sum", "softmax", "layernorm"):
        results.append({"id": "occupancy-registers", "title": "Reduction configuration may constrain occupancy", "status": "untested",
                        "evidence": [f"Row reduction spans {columns} elements."],
                        "missing_evidence": ["Registers per thread", "Achieved occupancy", "Spill traffic", "Instruction throughput"],
                        "falsifier": "Warp-count or tile changes alter occupancy without improving independently measured latency.",
                        "experiment": "Vary num_warps at fixed logical work; use profiler counters to distinguish occupancy, spills, and instruction cost.",
                        "confidence": "low"})
    if counters:
        for item in results:
            item["evidence"].append("Profiler counters are recorded; inspect units and collection conditions before causal interpretation.")
    if device.get("kind") != "cuda":
        for item in results:
            item["missing_evidence"].append("CUDA execution is unavailable in this CPU control; GPU hypothesis remains untested.")
    return results
