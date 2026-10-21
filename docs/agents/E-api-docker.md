Purpose: self-contained brief for the E API + Docker workstream of av-road-hazard; read it plus `fusion/schema.py` (once it exists) and nothing else.

# Brief: E API + Docker

## Lane
Codex

## Sprint and dependencies
Sprint 2 (days 4-7). Depends on the scaffold, plus A and B checkpoints for real pipeline wiring. Gated by the core-first gate only for parts beyond the API itself.

## Goal
Replace stubs with real calls through a `Pipeline` service object; routes per SPEC; Docker/compose for GPU and CPU; generate the TypeScript client from `openapi.json`.

## Owned paths
- `api/`
- `docker/`
- `tests/e2e/`

Do not touch:
- Model code in `perception/`, `defects/`
- `openapi.json` shape changes need a contract-change PR

## Inputs
- `fusion/schema.py`, SPEC section 5, `openapi.json` from the scaffold
- docs/ENVIRONMENT.md for Docker GPU rules and CUDA 12.8

No other exploration of the source repos is allowed.

## Outputs
- `api/` real `Pipeline` service object (loads models once, lazily)
- `docker/Dockerfile` (CUDA 12.8 base for GPU) and a CPU variant, `docker-compose.yml`, `compose.gpu.yml`
- Script generating the TypeScript client from `openapi.json` into `web/`

## Contract references
- `SPEC.md` section 5 (interfaces), section 4 (repo layout)
- Routes: `/analyze`, `/index`, `/query`, `/records/{id}`, `/health`

## Ordered task list
1. Implement the `Pipeline` service object: lazy model loading, `USE_STUBS` switch, `DEBUG_DUMP_DIR` dumps.
2. Implement routes per SPEC, including `/health` reporting each model, LLM status, Qdrant connectivity, and versions.
3. Add upload validation: type and size, safe decode, EXIF GPS stripping, per-IP rate limit.
4. Docker: CUDA 12.8 base image for GPU, CPU variant for hosting; compose with api, qdrant, ollama; `compose.gpu.yml` for NVIDIA passthrough (`--gpus all` or `deploy.resources.reservations.devices`); weights and data mounted as volumes, never baked in.
5. Generate the TypeScript client from `openapi.json` into `web/` via a script; add the OpenAPI drift check.
6. Structured logging with `request_id`; no image bytes or secrets in logs.

## Tests required
Layers from ENGINEERING.md section 3:
- Contract (responses validate against pydantic schema and `openapi.json`), integration (real pipeline on about 20 fixture images, CPU), e2e (compose up, `/analyze`, `/index`, `/query`)

## Done criteria
- `docker compose up` works on the NVIDIA machine and on CPU-only
- `/health` reports each model and LLM status

## Known pitfalls
- Docker GPU passthrough works only on NVIDIA hosts (Windows 11 + WSL2 or Linux with the NVIDIA container toolkit); not on macOS.
- Base image must be on the CUDA 12.8 line (exact tag UNVERIFIED) so PyTorch and driver match.
- Hosted compute is CPU-only: EfficientNet-B4 over many tiles plus Faster R-CNN is slow; limit tiles, downscale, cache, precompute.
- Never bake datasets into images; no nuScenes images in any bundle.

## Debugging hints
- Reproduce first: capture exact input image, config, seed, git SHA into `tests/fixtures/regressions/`.
- Use `tools/overlay.py` and `DEBUG_DUMP_DIR` before theorizing; most bugs are coordinate bugs (tile to frame, width/height order, resize scale).
- Check `/health` first; see docs/RUNBOOK.md for CUDA mismatch, Qdrant down, Ollama model not pulled, OOM.

## Token-saving notes
- Start a fresh session; read only this brief and `fusion/schema.py`.
- Do not explore beyond the files named in Inputs.
- Small, test-first PRs (under about 400 changed lines).

## Skills to load
Owner to fill in; none enumerated yet.

## PR checklist
- [ ] Branch `agent/api-docker`; PR into `main`; no direct push
- [ ] Spec section implemented is named in the PR
- [ ] Tests added or updated and passing; ruff and mypy clean
- [ ] Config values in `configs/*.yaml` (no hardcoded paths, thresholds, seeds); seeds 42; `num_workers` from config
- [ ] No secrets, data, or weights committed; license boundaries respected
- [ ] Docs and changelog updated if behavior or contract changed (`contract-change` label if so)
- [ ] "How I verified this" note and handoff note in the PR description
- [ ] Maintainer explainer paragraph written

## Reviewer lane
Claude (the other tool), using the ENGINEERING.md section 2 checklist in one short pass.

## Maintainer explainer
Write this paragraph in your PR: what the component does, why, and what could be wrong. Maintainers must be able to explain it in an interview.
