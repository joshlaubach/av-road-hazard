Purpose: known failure modes of av-road-hazard and their causes; fixes are completed when the features are implemented.

# Runbook

Stub: nothing is implemented yet. Causes come from the source repos' experience and the environment facts. Fixes marked "to be completed when implemented" are unknown today.

## CUDA mismatch

Cause: PyTorch wheel, container base image, and driver do not match. 50-series GPUs (RTX 5080) need CUDA 12.8 and PyTorch CUDA 12.8 wheels; the Docker base image should be on the CUDA 12.8 line (e.g. `nvidia/cuda:12.8.x-cudnn-runtime-ubuntu22.04`, exact tag UNVERIFIED). Fix: to be completed when implemented.

## Qdrant down

Cause: container not running (local) or wrong `QDRANT_URL` / `QDRANT_API_KEY` (cloud). `/health` reports Qdrant connectivity. Fix: to be completed when implemented.

## Ollama model not pulled

Cause: the model named in `OLLAMA_MODEL` (default `llama3.2-vision`) is not present on the Ollama host (`OLLAMA_HOST`, default `http://localhost:11434`). The source tool degrades gracefully: reports show "Report unavailable: <error>" while classification still completes. Fix: to be completed when implemented.

## Out of memory (OOM)

Cause: an 11B vision model needs roughly 8-12 GB VRAM at 4-bit (UNVERIFIED), competing with detector and classifier on a 16 GB GPU; or batching tiles too large. Mitigations known from the plan: run the LLM only when needed, skip LLM generation during bulk indexing. Fix: to be completed when implemented.

## Docker GPU not visible on macOS

Cause: Docker on macOS cannot use the Apple GPU or an NVIDIA GPU; containers run CPU-only. Run natively to use PyTorch MPS, or train on the NVIDIA machine. Not a bug.

## Other known source-repo behaviors

- Missing EfficientNet checkpoint raises `RuntimeError`.
- `ROIAlign` non-determinism: results reproduce only on the same hardware and library versions.
- Source `dqn.pth` will not load into the 65-output network (DQN is not ported).

Debug playbook: ENGINEERING.md section 4.
