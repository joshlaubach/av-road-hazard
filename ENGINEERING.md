Purpose: engineering practices for av-road-hazard (design, review, testing, debugging, CI/CD, ML hygiene, operations, security, docs, definition of done).
<!-- F:APP-C -->

# Engineering Practices: av-road-hazard

Applies to every contributor, human or agent. Companion to SPEC.md and BUILD_PLAN.md.

## 1. Design and decisions
- Interfaces live in `fusion/schema.py` and `openapi.json`. They are versioned; changes need a PR labeled `contract-change` and an entry in `CONTRACT_CHANGELOG.md`.
- Significant choices get a short ADR in `docs/adr/NNN-title.md` (context, decision, alternatives, consequences). Seed ADRs: Qdrant, road-masked tiling, dropping the DQN gate, switchable LLM backend, nuScenes kept local.
- Every PR states which spec section it implements.

## 2. Version control and review
- Branch per agent/feature (`agent/<workstream>`); PRs only, never push to `main`.
- Small PRs (aim under ~400 changed lines). Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Required to merge: CI green, one review. Reviews are done by an agent or person that did not write the change, checked against: contract adherence, tests present, no hardcoded paths/secrets, license boundaries.
- `main` is always deployable. Tag releases (`v0.1.0`).

## 3. Testing

| Layer | What | Where | When |
|---|---|---|---|
| Unit | Pure logic: tiling, box mapping, fusion, hazard score | `tests/<pkg>/` | every PR |
| Contract | API responses validate against pydantic schema and `openapi.json`; schema round-trips | `tests/contract/` | every PR |
| Integration | Real pipeline on ~20 fixture images in `tests/fixtures/` (small, license-clean) | `tests/integration/` | every PR (CPU) |
| ML smoke | Model overfits 8 images to near-zero loss; forward pass shapes; checkpoint load | `tests/ml/` | every PR |
| Data validation | Label ranges, box sanity, no scene leakage between train/val | `tests/data/` | when data code changes, nightly |
| Metric regression | F1/AP/macro-F1 must not fall below `eval/baselines.json` (with tolerance) | `eval/` | on model or data changes |
| Retrieval eval | recall@k on auto-generated queries | `eval/` | on RAG changes |
| Synthetic fusion | pasted-defect ground truth | `eval/` | on fusion/defects changes |
| LLM | Mocked client in CI; small live check run manually or nightly with cost cap | `tests/llm/` | CI mocked, nightly live |
| E2E | Docker Compose up, call `/analyze`, `/index`, `/query` | `tests/e2e/` | before release |

Rules: a bug fix starts with a failing test; tests are deterministic (seeded); no network in CI except the nightly job; coverage target 80% on non-model code (informational, not a gate for model code).

## 4. Debugging playbook
1. **Reproduce**: capture the exact input image, config, seed, git SHA. Add it to `tests/fixtures/regressions/`.
2. **Localize by stage**: every request carries a `request_id`; the pipeline can dump intermediates with `DEBUG_DUMP_DIR` (road mask, tiles, tile boxes, detections, fused record).
3. **Visual overlays first**: most bugs here are coordinate bugs (tile to frame, width/height order, resize scale). `tools/overlay.py` draws tiles, boxes, and masks on the source frame.
4. **Isolate**: run stages independently with stubs; bisect with `git bisect` once a failing test exists.
5. **Fix and lock in**: regression test, then the fix, then note the cause in the PR.
6. ML-specific checks: suspect data before model (label ranges, class balance, leakage), then thresholds, then training config; compare against the last good run in experiment tracking.

## 5. CI/CD
**On every PR** (GitHub Actions, CPU): ruff lint + format check, mypy, unit/contract/integration/ML-smoke tests, OpenAPI drift check (regenerate and diff), TypeScript typecheck + lint + build for `web/`, Docker build (api, cpu variant), secrets scan (gitleaks), dependency audit (pip-audit, npm audit).
**Nightly**: data validation, live LLM check (capped), metric regression if checkpoints are available.
**On tag**: build and push images, deploy to Hugging Face Space, smoke test the deployed `/health`. Keep the previous image tag for rollback.
Local GPU jobs (training, full eval) run on your machine; CI never needs a GPU.

## 6. ML hygiene
- Track every run (MLflow or W&B): config, git SHA, data version, seed, metrics, checkpoint hash.
- Datasets and weights are not in git. A `data/MANIFEST.json` and `checkpoints/MANIFEST.json` record versions and SHA-256 hashes; download scripts verify them.
- Split by scene, never by frame. Tune only on validation; touch the test split once per release.
- Report confidence intervals or multiple seeds where cheap; document confounds (v1.0-mini night/location correlation, no rain).
- Fixed seeds, deterministic flags where possible; note non-deterministic ops (ROIAlign) in the README.
- Pin library versions; record CUDA/PyTorch versions with each run.

## 7. Observability and operations
- Structured JSON logs (`request_id`, `frame_id`, stage, latency, counts). No image bytes or secrets in logs.
- `/health` reports per-model status, LLM backend, Qdrant connectivity, and versions.
- Track per-stage latency and confidence distributions; shifts in score distributions are the earliest drift signal.
- Hosted demo: per-IP rate limit, upload size and type limits, Claude API monthly spend cap with a hard cutoff and a fallback message.
- `docs/RUNBOOK.md`: common failures (CUDA mismatch, Qdrant down, Ollama model not pulled, OOM) with the fix.

## 8. Security and licensing
- Secrets only in env or the hosting platform's secret store; `.env` gitignored; rotate if ever committed.
- Validate uploads (type, size, decode safely), strip EXIF GPS from stored images.
- nuScenes and Waymo images never ship in images, bundles, or the hosted corpus. RDD2022 images only after its license is confirmed.
- Dependabot or Renovate for updates; lockfiles committed.

## 9. Documentation
- README: what it is, architecture diagram, quickstart (local and Docker), metrics table, honest limitations (effective detection range, domain gap, v1.0-mini confounds).
- Docstrings on public functions; `fusion/README.md` documents the hazard score formula.
- Keep SPEC.md, ADRs, and CONTRACT_CHANGELOG.md current; stale docs are a bug.

## 10. Definition of done (per PR)
Tests added or updated and passing; lint/type checks clean; config not code for tunables; docs and changelog updated if behavior or contract changed; no secrets, data, or weights committed; reviewer from outside the author checked it.

---

Related: [SPEC.md](SPEC.md), [BUILD_PLAN.md](BUILD_PLAN.md), [CONTRIBUTING.md](CONTRIBUTING.md), [docs/RUNBOOK.md](docs/RUNBOOK.md), [docs/EVALUATION.md](docs/EVALUATION.md), [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md), [docs/adr/](docs/adr/).
