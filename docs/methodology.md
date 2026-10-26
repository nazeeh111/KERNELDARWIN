# Measurement and verification protocol

This document defines how to interpret an investigation and what is required before accepting a performance claim. It includes follow-up research requirements beyond the bounded preview implementation. A field marked unavailable or not run is a limit, not a pass. Read the record and current source alongside this protocol.

## Record the contract first

Record operation semantics, full shape, strides, dtype, epsilon where relevant, seed, input-generation method, source/configuration identity, device model/architecture and software versions. Preserve compiler settings and timing mode. Capture driver/runtime, clocks, power and concurrent activity where observable; explicitly list unavailable conditions. Do not silently substitute CPU execution for a requested CUDA run.

Keep PyTorch eager as the declared CUDA reference. Measure `torch.compile` separately if it succeeds, retaining any failure reason. Separate first-call compilation/setup from steady-state latency. A template candidate has the same output shape, dtype and declared numerical semantics as its reference. Unsupported shapes and layouts fail clearly rather than falling back invisibly.

## Correctness precedes speed

Use independent fresh inputs across seeds, zeros, constants, negative values, small and large finite values, odd sizes, boundary widths, supported noncontiguous strides and each declared dtype. Compare against the reference with recorded absolute and relative tolerances appropriate to the dtype and operation. Report maximum error and case count; any mismatched case invalidates that candidate.

A first numerical suite is not proof over all inputs. Specifically test that changed inputs change outputs as expected, previous inputs are not cached as results, output storage is fresh where promised, and input tensors remain unmodified. Keep the tested source/configuration identity with the result. Never accept a precision change, missing computation, stale output or hidden CPU fallback as an optimization. NaN/Inf semantics need a declared contract and tests before support is claimed.

Compute Sanitizer is a separate diagnostic route on supported CUDA installations. Run `memcheck` before relying on `racecheck`; synchronization and initialization checks cover other failure classes. [NVIDIA documents](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/) that shared-memory race detection is not general memory checking. Preserve exact invocation, exit code and logs. A missing tool must remain `not_run` or unavailable.

## Timing

Warm the exact callable and synchronize as required. Collect repeated GPU measurements using CUDA events for device work, keeping measurement setup and synchronization consistent. Interleave baseline and candidate measurements with randomized order within each pair to reduce drift bias. Store raw per-pair microseconds and the order/seed; do not retain only a rounded headline.

Use normal launches as the default timing mode. CUDA graph replay is a distinct mode and must be labeled and compared like for like. Host wall-time or application timing answers a different question from isolated device-event timing. Count dispatch, transfers, allocations or graph capture only where the declared benchmark includes them. Retain compile/setup cost separately.

Summarize median, spread, number of observations and coefficient of variation. The preview's paired bootstrap reports a 95% interval for the ratio of baseline and candidate medians; it resamples complete pairs. The basic acceptance threshold requires at least 20 pairs and an interval lower bound exceeding the configured minimum speedup, normally 1.03. This is a practical noise gate, not an automatic guarantee of an unbiased experiment or a universal significance test.

Searching many candidates creates selection bias. After selecting a winner, independently rerun that source/configuration with new timing samples and correctness inputs. For a headline, reproduce in a fresh process/session and report failures and variability. Where candidates are nearly tied, preserve the simpler or established implementation rather than overstating a fractional gain. Never aggregate unrelated tasks as though repeated samples were additional workloads.

## Diagnosis and counters

Collect counters separately from fair latency timing. [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) may replay kernels/applications, control clocks and alter cache conditions; the report must state those settings. A trace establishes recorded events, not an inferred hardware utilization percentage. In the absence of actual counter/trace data, the LAB must show that absence.

Use controlled interventions to distinguish explanations. Change one relevant factor at a time where practical. Separate observed speedup from a hypothesized mechanism. Keep failed experiments and unresolved alternatives. Analytical bandwidth or arithmetic-intensity calculations must expose assumptions and byte/FLOP accounting. Do not fabricate hardware ceilings or claim a roofline bottleneck without suitable evidence.

## Budget, isolation and integration

Specify maximum candidates, timeout and workload-size limits before the run. Record invalid candidates and timeouts as costs. Run only bounded trusted templates in the preview; imported JSON is data and never executable code. A subprocess timeout and sanitized environment reduce operational risk but are not a security sandbox. Arbitrary generated GPU programs require a separately provisioned worker/container with enforced resource limits, restricted mounts and no credentials or unnecessary network.

An exported accepted kernel must be callable from a supported application and enforce its supported input contract. Reproduce the result in that integration and measure the complete application path against eager/compiled alternatives. Microbenchmark speedup does not imply application speedup. Preserve the unoptimized implementation as a fallback where contract support is narrower.

## Reproduction package

Retain versioned raw records, source/configuration, seeds, paired samples, numerical results, failures, timing mode and environment information. Separate `cpu_control`, actual CUDA measurements and external/reference records. Include training/test group identities and policy parameters with learned-selection reports. No CPU or replay result may be relabeled as a new CUDA run.

The development host for this preview is Apple M4/macOS. It cannot establish CUDA correctness, CUDA-event performance, NVIDIA counter observations or Compute Sanitizer outcomes. Those validations must be completed on compatible Linux/NVIDIA hardware before GPU success is claimed.
