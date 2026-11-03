Purpose: agent build plan for av-road-hazard (ground rules, sprints, phases, workstream briefs, merge order).
<!-- F:APP-B -->

# av-road-hazard: Agent Build Plan

Companion to SPEC.md and ENGINEERING.md. Written so each agent can work from its own brief without reading the whole conversation. Suggested skills are placeholders; the owner's own coding skills (not yet enumerated) replace them in each brief.

## 0. Ground rules (every agent)

- One branch per agent: `agent/<workstream>`. Open a PR into `main`; never push to `main` directly.
- Own only your listed paths. Anything outside them goes through the interface contract in SPEC.md section 5 or a PR comment.
- Interfaces are frozen after Phase 0. Changes to `fusion/schema.py` or the API OpenAPI spec need a PR labeled `contract-change` and a note in `CONTRACT_CHANGELOG.md`.
- No data, weights, or secrets in git. Weights go in `checkpoints/` (gitignored); secrets via `.env` (`.env.example` committed).
- Every PR follows the Definition of Done in ENGINEERING.md section 10: tests pass, `ruff` and `mypy` clean, config values in `configs/*.yaml` (no hardcoded paths, thresholds, or seeds), and a short "how I verified this" note.
- Seeds fixed (42); `num_workers` read from config.
- Commit trailers per session attribution settings.

## 1. Execution model: two agents, sequenced sprints

Constraint: one Claude and one Codex plan at $20/mo each, so usage limits, not coordination, are the bottleneck. At most two agents work at once (one per tool), each on one workstream, in a fresh session with only its brief and the schema. GPU training runs on your machine and costs no tokens, so launch them as early as possible and work on other things while they run.

Lanes:
- **Claude lane:** contracts and scaffold, fusion, RAG design, integration, hard debugging, deploy, and review of Codex PRs.
- **Codex lane:** well-specified bulk implementation (ports, tests, Docker, API routes, UI), and review of Claude PRs.
- Every PR is reviewed by the *other* tool. Reviews use the checklist in ENGINEERING.md section 2 and stay under one short pass to save usage.

| Sprint | Days | Claude lane | Codex lane | Your machine (no tokens) |
|---|---|---|---|---|
| 0 | 1 | Scaffold, schema, stubs, CI, ADRs (Phase 0 below) | Idle or review of scaffold PR | Download RDD2022, nuScenes v1.0-mini |
| 1 | 2-4 | C Fusion (stub-driven) | A Perception port, then B Defects port and tiling | Train detector, then classifier (background) |
| 2 | 4-7 | D RAG (Qdrant, embeddings, LLM backends) | E API + Docker (real pipeline wiring, compose, GPU) | Finish training; run metric baselines |
| 3 | 7-9 | G Eval (retrieval, synthetic fusion) | F Web UI against the API | Index nuScenes locally |
| 4 | 9-11 | Integration: swap stubs for models, fix coordinate and schema bugs | Tests and fixtures for gaps, UI fixes | Run full evals; record baselines |
| 5 | 11-14 | Deploy (HF Space, Qdrant Cloud), README, final review | Review pass, polish, walkthrough video support | Record video |

If a sprint overruns, cut in this order: React polish, Claude/LLaMA toggle UI (keep config switch), v1.0-full evaluation.

Token-saving rules:
- One workstream per session; start fresh sessions instead of long ones.
- Give agents their brief plus `fusion/schema.py`, not the whole repo history.
- Agents must not explore the original repos beyond the files named in their brief.
- Debug with `tools/overlay.py` and `DEBUG_DUMP_DIR` yourself first; hand agents a reproducing fixture, not a vague symptom.
- Prefer small, test-first PRs so reviews are cheap.
- Claude API for the hosted demo is billed separately from the subscription; keep the spend cap on.

Dependencies:
```
Phase 0 scaffold ──► C, A, B, D, E, F, G (all build against stubs)
A, B training ──► Phase 2 integration (critical path; start immediately after Sprint 1 ports)
```

## 2. Phase 0: Scaffold (one agent, blocks everyone)

Deliverables:
- Repo `joshlaubach/av-road-hazard` (private), package skeleton per SPEC section 4, `pyproject.toml`, ruff + pytest config, pre-commit, GitHub Actions CI (lint + unit tests, CPU only).
- `fusion/schema.py`: pydantic `Object`, `Defect`, `HazardRecord` exactly per SPEC.
- Stub implementations: `perception.infer.detect()` and `defects.infer.find_defects()` return fixed fake outputs behind a `USE_STUBS=1` flag so every other stream can integrate immediately.
- FastAPI app with all routes returning stub data; export `openapi.json` to repo.
- `configs/default.yaml`, `.env.example`, `docker/` skeleton (empty compose with api, qdrant, ollama services).
- `CONTRACT_CHANGELOG.md`, `CONTRIBUTING.md` containing the ground rules above, and `ENGINEERING.md` (committed as-is).
- Quality scaffolding per ENGINEERING.md: mypy config, `tests/{unit,contract,integration,ml,data,e2e}/` with one passing example each, `tests/fixtures/` (license-clean images only), `eval/baselines.json` placeholder, `tools/overlay.py` debug overlay, `DEBUG_DUMP_DIR` support in the pipeline, structured JSON logging with `request_id`, `docs/adr/` with seed ADRs, `docs/RUNBOOK.md` stub, gitleaks + pip-audit + npm audit in CI, OpenAPI drift check, Dependabot config, data/checkpoint `MANIFEST.json` format.

Done when: `docker compose up` serves stub `/analyze` and `/health`; CI green; all agents can branch from `main`.

## 3. Workstream briefs

### A. Perception (`perception/`, `tests/perception/`)
- Port from `nuscenes-stability` notebook 02 (Faster R-CNN ResNet50 FPN v2, torchvision, torchmetrics AP) into `train.py` and `infer.py`. Reuse `src/data.py`, `src/metrics.py`, `src/utils_nuscenes.py` logic as modules, not notebooks.
- Requires nuScenes v1.0-mini locally (`NUSCENES_ROOT`); build the 2D box cache, train, save to `checkpoints/detector.pt`.
- Implement `detect(frame) -> list[Object]`; batch and device-aware; CPU path for the hosted demo.
- Reproduce the original v1.0-mini metrics (F1 ~0.53, AP@0.50 ~0.31) as a regression check.
- Skills: ML/PyTorch, notebook-to-module refactor.
- Done when: checkpoint trains reproducibly, metrics within tolerance of original, `detect()` unit-tested on a fixture image, full-dataset switch is one config key.

### B. Defects (`defects/`, `tests/defects/`)
- Download RDD2022; retrain EfficientNet-B4 multi-label (D00, D10, D20, D40). Address the 0.92 train / 0.73 val gap (augmentation, stronger regularization, early stopping on val).
- Road-masked tiling (`road_crop.py`, `patches.py`): drivable-road mask (nuScenes map data or a road segmentation model), overlapping tiles on the near-field road at native resolution, upscale to 380x380.
- Auto-labeled negatives: non-road tiles from nuScenes frames added to training as "no defect"; no hand labeling.
- `find_defects(frame) -> list[Defect]` with tile-to-frame box mapping; severity heuristic or optional LLM text (off by default).
- MVTec is excluded from v1.
- Done when: macro-F1 on RDD2022 val reported, negatives reduce false positives on non-road tiles (measure rate), tiling tested on fixtures.

### C. Fusion (`fusion/`, `tests/fusion/`)
- Depends only on schema + stubs. Implement: road mask intersection (`in_road`), nearest-object computation (`near_objects` with IoU and pixel distance), deterministic `hazard_score` formula documented in `fusion/README.md`.
- Pure functions, no model imports. Property-style tests with hand-built boxes.
- Done when: 100% of schema fields populated, edge cases covered (no objects, no defects, overlapping boxes, boxes at frame edge).

### D. RAG (`rag/`, `tests/rag/`)
- CLIP-style image + text embeddings (`embed.py`); Qdrant collection with image vector, text vector, and metadata payload (source, time_of_day, labels, hazard_score, severity).
- `index.py` (batch and single), `retrieve.py` (hybrid: metadata filter then vector search, with simple rerank), `llm.py` with a `LLMBackend` protocol: `OllamaBackend` (LLaMA 3.2 Vision) and `ClaudeBackend`, chosen by `LLM_BACKEND` env/config. Claude key from env only.
- Prompt template grounds answers strictly in retrieved records and cites frame IDs.
- Done when: index 100+ stub records, retrieve with filters, both backends answer a sample query (Claude backend behind a mocked client in CI).

### E. API + Docker (`api/`, `docker/`)
- Replace stubs with real calls via a `Pipeline` service object (loads models once, lazy). Routes per SPEC: `/analyze`, `/index`, `/query`, `/records/{id}`, `/health`.
- Docker: CUDA 12.8 base for GPU, CPU variant for hosting; compose with api, qdrant, ollama; `compose.gpu.yml` for NVIDIA passthrough. Weights/data mounted as volumes.
- Generate TypeScript client from `openapi.json` into `web/` via script.
- Done when: `docker compose up` works on the 5080 and on CPU-only; `/health` reports each model and LLM status.

### F. Web UI (`web/`)
- Next.js + TypeScript; typed client generated from OpenAPI. Views: Analyze (upload, overlay boxes with severity colors, record JSON panel), Search (text + image query, filters, results grid, answer with cited frames), Settings (LLM backend toggle).
- Build against the stub API first. Skills: your frontend/design skill; also use a frontend-design skill for non-templated styling, and dataviz for any charts.
- Done when: runs against stub API, Lighthouse/a11y basics, responsive at phone width, no nuScenes images bundled.

### G. Eval + Data (`eval/`, `data/`)
- Download scripts (RDD2022, nuScenes instructions), nuScenes loader.
- `eval/models.py` (per-model metrics), `eval/retrieval.py` (queries auto-generated from nuScenes GT annotations, recall@k), `eval/fusion_synthetic.py` (paste RDD2022 patches on nuScenes road regions at known positions, verify box/in_road/near_objects).
- Output a single `eval/results.json` and a markdown table for the README.
- Done when: all three evals run end-to-end on stubs and real models and write results.

## 4. Phase 2: Integration (one agent, with a reviewer)

1. Set `USE_STUBS=0`; run `/analyze` on 20 fixture images; fix schema or box-coordinate mismatches (most likely: tile-to-frame coordinates, image size ordering).
2. Index nuScenes v1.0-mini locally; run retrieval eval.
3. Run synthetic fusion test; tune thresholds in config only.
4. Review pass by a separate agent that has not seen the build: checks contract adherence, secrets, hardcoded paths, license boundaries (no nuScenes images in the hosted bundle).

## 5. Phase 3: Deploy and polish

- HF Space (CPU Docker), Qdrant Cloud free tier, Claude key as a Space secret; hosted corpus decided 2026-11-03: 40 non-US RDD2022 samples plus 20 Stellenbosch photos with credits, own photos optional; see [OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md) question 10; precompute and cache outputs for the sample set so the demo stays fast on CPU.
- README: architecture diagram, metrics table, limitations (effective detection range, v1.0-mini confounds), run instructions; 2-3 minute walkthrough video.

## 6. Merge order

Scaffold → C and G data scripts → A and B (once checkpoints exist) → D → E → F → integration PR → deploy PR.

## 7. Open items before agents start

- Push access to create the repo.
- RDD2022 license check.
- Which of your coding skills each agent should load (fill into the briefs).

---

Related: [SPEC.md](SPEC.md), [ENGINEERING.md](ENGINEERING.md), [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md), [docs/AGENT_WORKFLOW.md](docs/AGENT_WORKFLOW.md), [docs/agents/](docs/agents/), [docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md).
