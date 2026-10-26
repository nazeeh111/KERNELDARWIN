# Recorded technical demonstration

The 20-second, 1920 × 1080 film is intentionally silent. It shows screenshots of actual KERNELDARWIN LAB CPU control records. The entire film carries `CPU CONTROL / RECORDED PLAYBACK`; it does not show live GPU execution or establish a CUDA performance result.

- `brag.mp4`: final video, with the settled product poster baked into frame 0.
- `brag.jpg`: poster image.
- `brag-plan.md`, `composition-brief.md`, `DESIGN.md`: narrative and design sources.
- `composition/index.html`: editable HyperFrames/GSAP composition.
- `source-manifest.json`: current UI screenshot identity and render provenance.
- `verification.md`: checks and inspection limits.
- `share-copy.txt`: optional caption. No message or social post was sent.

Re-render with a local HyperFrames installation, then inspect the result:

```sh
hyperframes lint composition
hyperframes check composition --samples 9
hyperframes render composition --output new-render.mp4 --fps 30 --quality standard --workers 1
ffprobe new-render.mp4
ffmpeg -v error -i new-render.mp4 -f null -
```

The development render uses an existing pinned HyperFrames 0.8.33 runner and native Chrome, with no model API, cloud renderer or audio provider. Commands may differ in newer versions. The source is deterministic and uses local screenshots, mark and GSAP runtime. The bundled GSAP runtime retains its upstream license header; its distribution terms are in `composition/assets/GSAP-LICENSE.txt` and https://gsap.com/standard-license/.
