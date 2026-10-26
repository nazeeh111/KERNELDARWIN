# Demonstration verification

The final `brag.mp4` is 20.000 seconds, 1920 × 1080, H.264 yuv420p, 30 fps, 600 frames and intentionally contains no audio stream. `ffprobe.json` records the exported media properties. A full FFmpeg decode completed without errors. The settled Overview scene at 5.6 seconds supplies `brag.jpg` and is baked into frame 0; the hook begins immediately afterward.

The final composition used the LAB agent's final current-source screenshots, confirmed after native Chrome checks. Source hashes are preserved in `source-manifest.json`; no measurement or interface panel was invented. CPU CONTROL / RECORDED PLAYBACK and no-live-GPU labels remain visible across the film. The ending states CUDA validation pending.

HyperFrames 0.8.33 lint/runtime/layout checks passed with zero errors and warnings. Layout sampling covered 9 times, with transition sampling as well. All 40 automated text contrast checks passed. Crossfade text overlap during the 0.35 second handoffs and cropped screenshot boundaries are intentional. Hero scenes and transition frames were inspected directly after rendering. `check.json` contains the compact check result.

Limits: the automated motion module was disabled. The separate animation-map helper could not load `@hyperframes/producer`; no new package installation was attempted. Sampled visual inspection and full decode do not assert every possible playback platform or accessibility conformance. No GPU kernel was executed by the demonstration, and its CPU timings do not establish CUDA improvement.

Editable composition, local media assets, upstream GSAP license notice, and narrative/design sources are included. Drafts,raw render and expanded QA evidence are preserved in the ignored task-local `.work/demo-qa` directory.
