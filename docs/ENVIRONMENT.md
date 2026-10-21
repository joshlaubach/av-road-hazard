Purpose: compute and environment facts for av-road-hazard: candidate dev machines, Docker GPU rules, CARLA host, Ollama, paths, versions, disk sizes, timings.

# Environment

Items marked UNVERIFIED came from memory or inference, not a checked source.

## Candidate dev machines

| Machine | Facts |
|---|---|
| macOS MacBook (the machine the docs were drafted on) | macOS, arm64 (Apple silicon), device name "joshs-macbook-pro-2-local", Claude desktop app 3.1623.0. Its home directory contains tool folders including `.codex`, `.ollama`, `.kaggle`, `.conda`, `miniconda3`, `.nvm`, `.streamlit` (names observed; contents unverified). <!-- F:E-01 --> |
| Windows 11 + NVIDIA RTX 5080 (per source READMEs) | The drivenet-carla-av and defect-detector READMEs describe development on Windows 11 with an RTX 5080, CUDA 12.8, PyTorch CUDA 12.8 wheels (needed for 50-series), and CARLA run with `-dx12`. **[CRITICAL]** It is UNCONFIRMED whether this is a separate desktop or whether the owner now works on the MacBook. This decides where training runs and whether Docker GPU passthrough is possible. Blocks Sprint 1; tracked in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md). <!-- F:E-02 --> |

## Docker GPU rules

GPU passthrough works only on NVIDIA hosts: Windows 11 + WSL2 via Docker Desktop, or Linux with the NVIDIA container toolkit. Docker on macOS cannot use the Apple GPU or an NVIDIA GPU. On a Mac, containers run CPU-only; native (non-Docker) runs can use PyTorch MPS. <!-- F:E-03 -->

## CARLA host

CARLA 0.9.16 runs on the Windows host with DirectX 12, outside Docker. The CARLA frame generator is a host-side script. CARLA does not run on macOS. <!-- F:E-04 -->

## GPU needs by stage

| Stage | GPU need |
|---|---|
| Faster R-CNN detection | GPU for practical speed; CPU takes seconds per frame |
| EfficientNet-B4 defect classification | GPU recommended |
| CLIP-style image embeddings | GPU recommended; CPU feasible for a few thousand frames |
| LLaMA 3.2 Vision via Ollama | Most GPU-intensive. An 11B vision model needs roughly 8-12 GB VRAM at 4-bit quantization, which fits a 16 GB RTX 5080 (UNVERIFIED, approximate) |
| Vector search and metadata filtering | CPU only |
| RAG answer generation | Reuses the LLM; GPU only for the few retrieved frames per query |

<!-- F:E-05 -->

## Practical GPU plan

Run detection, defect classification and embeddings on GPU during indexing. Run the LLM only when needed. Skip LLM report generation during bulk indexing (a `--no-llama` equivalent) and run it for returned frames only. <!-- F:E-06 -->

## Docker plan

Containerize the FastAPI service (both detectors), an Ollama container for LLaMA, Qdrant, and a one-off indexing job; run together with Docker Compose. GPU access via `--gpus all` or a compose `deploy.resources.reservations.devices` entry; Ollama's official image supports this. Use a base image on the CUDA 12.8 line, for example `nvidia/cuda:12.8.x-cudnn-runtime-ubuntu22.04` (exact tag UNVERIFIED), so the PyTorch build and driver match. Do not bake datasets into images; mount volumes; set `NUSCENES_ROOT` via `.env`. <!-- F:E-07 -->

## Hosted demo compute

CPU only (Hugging Face Spaces). EfficientNet-B4 over 64 patches plus Faster R-CNN on CPU is slow, so limit tile count, downscale, cache, or precompute outputs for sample images. <!-- F:E-08 -->

## Timing measurements

Measured by the source repo on an RTX 5080 for one 640x640 RDD2022 frame with `--no-llama --no-dqn`: EfficientNet-B4 over 64 patches about 1.9 s (one patch per forward pass; batching is an easy speed-up); LLaMA 3.2 Vision about 2-5 s per defect patch on Ollama. With a trigger-happy classifier (31 of 64 patches flagged), a frame can take 1-2 minutes end to end. The source tool is framed as batch/post-drive, not real time. <!-- F:E-09 -->

## Disk sizes

| Item | Size |
|---|---|
| RDD2022 | about 13.3 GB download, about 15 GB extracted |
| MVTec AD | about 5 GB (dropped from v1) |
| Source checkpoints | about 80 MB |
| Full source checkout | once reached about 37 GB due to leftover archives (the downloader now deletes archives unless `--keep-archive`) |
| nuScenes full | on the order of hundreds of GB (UNVERIFIED; estimated at ~300 GB+) |

<!-- F:E-10 -->

## Versions

drivenet-carla-av uses Python 3.11 and PyTorch 2.3.1; defect-detector uses Python 3.12. The new repo's Python version is undecided: open question with recommendation 3.12. <!-- F:E-11 -->

## Local services and paths

Ollama on `http://localhost:11434` with model `llama3.2-vision`. Qdrant in Docker locally and Qdrant Cloud free tier when hosted. The Kaggle API token at `~/.kaggle/kaggle.json` is only needed for MVTec AD, which is dropped from v1. Paths come from `.env` (see `.env.example`) and `configs/*.yaml`; none are hardcoded. <!-- F:E-12 -->

## GPU and Docker requirements

GPU and Docker requirements are covered by the sections above: GPU needs by stage, practical GPU plan, Docker GPU rules, Docker plan, and timings. <!-- F:U-18 -->
