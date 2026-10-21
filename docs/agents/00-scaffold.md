Purpose: self-contained brief for the Scaffold (Phase 0) workstream of av-road-hazard; read it plus `fusion/schema.py` (once it exists) and nothing else.

# Brief: Scaffold (Phase 0)

## Lane
Claude

## Sprint and dependencies
Sprint 0 (day 1). Blocks everyone. No dependencies except the first commit and the remote existing with branch protection enabled. Codex idle or reviews the scaffold PR. Owner machine downloads RDD2022 and nuScenes v1.0-mini in the background.

## Goal
Create the package skeleton, contracts, stubs, stub API, and quality scaffolding so every other workstream can build against stubs. Done when `docker compose up` serves stub `/analyze` and `/health`, CI is green, and all agents can branch from `main`.

## Owned paths
- Everything not owned by a later workstream: `pyproject.toml`, `configs/`, `fusion/schema.py`, stub `perception/infer.py` and `defects/infer.py`, `api/`, `docker/` skeleton, `.github/workflows/`, `tools/`, `tests/`, `eval/baselines.json`, `docs/adr/`, `docs/RUNBOOK.md`

Do not touch:
- Real model code (workstreams A, B); real RAG (D); real UI (F)

## Inputs
- `BUILD_PLAN.md` section 2 (Phase 0), `ENGINEERING.md` (all), `SPEC.md` sections 4 and 5
- No source-repo files needed; ports happen in later workstreams

No other exploration of the source repos is allowed.

## Outputs
- Repo `joshlaubach/av-road-hazard` (private) package skeleton per SPEC section 4
- `pyproject.toml`, ruff + pytest config, mypy config, pre-commit
- GitHub Actions CI (lint, unit tests, CPU only), gitleaks, pip-audit, npm audit, OpenAPI drift check
- `fusion/schema.py` pydantic `Object`, `Defect`, `HazardRecord` exactly per SPEC
- Stub `perception.infer.detect()` and `defects.infer.find_defects()` behind `USE_STUBS=1`
- FastAPI app with all routes returning stub data; `openapi.json` exported
- `configs/default.yaml`, `docker/` skeleton (empty compose with api, qdrant, ollama)
- `tests/{unit,contract,integration,ml,data,e2e}/` with one passing example each; `tests/fixtures/` (license-clean images only)
- `eval/baselines.json` placeholder, `tools/overlay.py`, `DEBUG_DUMP_DIR` support, structured JSON logging with `request_id`
- Data/checkpoint `MANIFEST.json` format, Dependabot config, lockfiles

## Contract references
- SPEC.md section 5: `Object`, `Defect`, `HazardRecord`, API routes
- CONTRACT_CHANGELOG.md baseline v0.0.0

## Ordered task list
1. Create the package skeleton per SPEC section 4 and `pyproject.toml` with ruff, pytest, mypy config; fix Python version per the owner's answer (recommendation 3.12).
2. Write `fusion/schema.py`: `Object`, `Defect`, `HazardRecord` exactly per SPEC; pin `image_size` ordering and add a round-trip test.
3. Add stubs for `detect()` and `find_defects()` returning fixed outputs when `USE_STUBS=1`.
4. Build the FastAPI app with `/analyze`, `/index`, `/query`, `/records/{id}`, `/health` returning stub data; export `openapi.json`.
5. Add `configs/default.yaml` (paths, thresholds, seeds 42, `num_workers`) and make `.env.example` consistent.
6. Add the `docker/` skeleton: compose with api, qdrant, ollama; `compose.gpu.yml` placeholder.
7. Add the test tree with one passing example per layer and license-clean fixtures.
8. Add `tools/overlay.py`, `DEBUG_DUMP_DIR` support, structured JSON logging with `request_id`.
9. Add CI workflows: ruff, mypy, tests, OpenAPI drift check, Docker build (cpu), gitleaks, pip-audit, npm audit; Dependabot; pre-commit; lockfiles.
10. Define the `MANIFEST.json` formats for `data/` and `checkpoints/` and the `eval/baselines.json` placeholder.
11. Add the commands to AGENTS.md section 4 (lint, type-check, test, run API, train, eval).

## Tests required
Layers from ENGINEERING.md section 3:
- Unit, contract (schema round trip, OpenAPI), integration (stub pipeline on fixtures), ML smoke placeholder, data validation placeholder, e2e (compose up, call `/analyze`, `/health`)

## Done criteria
- `docker compose up` serves stub `/analyze` and `/health`
- CI green
- All agents can branch from `main`
- Commands defined in AGENTS.md section 4

## Known pitfalls
- Contracts freeze after this phase: get them right (field names, `image_size` ordering, severity enum).
- No data, weights, or secrets in git; fixtures must be license-clean (no nuScenes or Waymo images; no RDD2022 images until its license is verified).
- Docker on macOS is CPU-only; GPU compose file is for NVIDIA hosts.
- Line endings: Windows development, Linux Docker (`.gitattributes`).

## Debugging hints
- Reproduce first: capture exact input image, config, seed, git SHA into `tests/fixtures/regressions/`.
- Use `tools/overlay.py` and `DEBUG_DUMP_DIR` before theorizing; most bugs are coordinate bugs (tile to frame, width/height order, resize scale).

## Token-saving notes
- Start a fresh session; read only this brief and `fusion/schema.py`.
- Do not explore beyond the files named in Inputs.
- Small, test-first PRs (under about 400 changed lines).
- Keep the scaffold minimal; later workstreams add their own code.

## Skills to load
Owner to fill in; none enumerated yet.

## PR checklist
- [ ] Branch `agent/scaffold`; PR into `main`; no direct push
- [ ] Spec section implemented is named in the PR
- [ ] Tests added or updated and passing; ruff and mypy clean
- [ ] Config values in `configs/*.yaml` (no hardcoded paths, thresholds, seeds); seeds 42; `num_workers` from config
- [ ] No secrets, data, or weights committed; license boundaries respected
- [ ] Docs and changelog updated if behavior or contract changed (`contract-change` label if so)
- [ ] "How I verified this" note and handoff note in the PR description
- [ ] Owner explainer paragraph written

## Reviewer lane
Codex (the other tool), using the ENGINEERING.md section 2 checklist in one short pass.

## Owner explainer
Write this paragraph in your PR: what the component does, why, and what could be wrong. The owner must be able to explain it in an interview.
