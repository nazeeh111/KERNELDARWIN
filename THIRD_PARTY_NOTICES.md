# Third-party notices and research attribution

KERNELDARWIN's initial application/core sources are separately authored. The prior-art review did not vendor competitor implementations, benchmark tasks, figures, datasets or pretrained weights. Research attribution does not imply endorsement or reproduced performance.

Optional dependencies are installed by users and retain their own licenses:

| Project | License/source | Use |
|---|---|---|
| PyTorch | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE), including BSD-style terms and historical component notices | Optional tensor reference and compiled baseline |
| Triton | [MIT](https://github.com/triton-lang/triton/blob/main/LICENSE), copyright Philippe Tillet and OpenAI | Optional GPU language/compiler |
| BoTorch | [MIT](https://github.com/meta-pytorch/botorch/blob/main/LICENSE) | Investigated optional comparison baseline; not required by core |
| NVIDIA CUDA, Nsight Compute, Compute Sanitizer | NVIDIA's applicable distribution/tool terms | User-provided optional runtime/profiling/safety tools; binaries not redistributed |

Research foundations, reviewed 26 October 2026:

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench): MIT, Scaling Intelligence Lab/Stanford authors. Ouyang et al., [*KernelBench: Can LLMs Write Efficient GPU Kernels?*](https://arxiv.org/abs/2502.10517), 2025.
- [KernelAgent](https://github.com/meta-pytorch/KernelAgent): Apache-2.0, Meta/PyTorch contributors. Prior hardware-guided optimization and verified composition.
- [Kernel Forge](https://github.com/TheJoshBrod/KernelForge): MIT, copyright 2026 Joshua Brodsky. Brodsky et al., [*Kernel Forge: An Agent Harness for LLM-based Generation and Optimization of CUDA Kernels*](https://arxiv.org/abs/2607.24762), 2026.
- Chen et al., [*Learning to Optimize Tensor Programs*](https://arxiv.org/abs/1805.08166), 2018, and Zheng et al., [*Ansor*](https://arxiv.org/abs/2006.06762), 2020: learned program-cost models/search precedents.
- [Kernel Tuning Toolkit](https://github.com/HiPerCoRe/KTT): MIT. Hozzová et al., [counter-guided autotuning data article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8633859/), 2021. [Dataset DOI 10.17632/nn53dskr7z.1](https://data.mendeley.com/datasets/nn53dskr7z/1), CC BY 4.0, is a potential external replay dataset, not bundled or remeasured here.
- [PaperVizAgent](https://github.com/google-research/papervizagent), formerly PaperBanana: Apache-2.0, Google Research contributors. Investigated illustration workflow; no pipeline output or upstream figure is copied.
- [/brag](https://github.com/latent-spaces/brag): MIT, copyright 2026 Shunit Haviv Hakimi. Investigated launch-video workflow; the locally installed skill may be used for release presentation.

If code or media is later copied or adapted, add exact revision and original copyright, ship the full applicable license and preserve required notices beside the artifact. Installed dependency licenses are not replaced by KERNELDARWIN's license. Authored SVG assets do not contain upstream logos or research figures.

## Launch demonstration runtime

The editable HyperFrames composition under `docs/demo/composition` includes GSAP 3.14.2 for local animation. Its original header is retained. GSAP is available without charge under the [GSAP Standard License](https://gsap.com/standard-license/), which is separate from this repository's Apache-2.0 license and is not an OSI license. See `docs/demo/composition/assets/GSAP-LICENSE.txt` for the source and license notice. No GSAP paid service or plugin is used. HyperFrames and FFmpeg are local rendering tools, not bundled runtime dependencies of KERNELDARWIN's Python core or LAB.
