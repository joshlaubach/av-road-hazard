Purpose: where av-road-hazard stands today, the one thing in progress, the ordered list of what to build next with exit criteria, and what "production-ready" means.

# Roadmap

Last updated: 2026-10-26. Update this file in the same PR that finishes or changes any row. If a row here disagrees with the code, the code wins and this file is wrong.

Order follows the proposed core-first gate ([ADR 0011](adr/0011-core-first-gate.md)): the core (detector, defect classifier, fusion, synthetic fusion evaluation) must produce reportable numbers before RAG, the web UI or deployment go past stubs.

## Done

| Piece | State | Evidence |
|---|---|---|
| Contracts, stub API, CI, Docker skeleton | Done | `fusion/schema.py`, `openapi.json`, `.github/workflows/ci.yml`. The Docker build and compose have never been run ([OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)). |
| Object detector | Done, zero-shot | COCO Faster R-CNN with pinned weights ([ADR 0014](adr/0014-object-detector.md)). On nuScenes mini at score 0.5 the default MobileNet finds 0.38 of pedestrians (range 0.35-0.42) and 0.21 of cyclists (0.14-0.31, only 57 objects). CPU speed passes p95 <= 2 s at 24, 4 and 2 threads. |
| Road mask | Done | comma10k model, pinned and hash-checked, `min_road` 0.4 ([ADR 0013](adr/0013-road-mask-model.md)). Wired into `api/pipeline.py`. |
| Defect classifier | Trained, thresholds provisional | EfficientNet-B4 on RDD2022 tiles. `find_defects` masks objects and off-road tiles and uses per-class thresholds from config. Class meanings pinned by a test (S-11). |
| Fusion | Done | `fusion/fuse.py`: `in_road`, `near_objects`, deterministic `hazard_score`. Severity stays "unknown" (see Next, item 4). |
| Evaluation tooling | Done | `eval/`, notebooks `03_eval_classifier` and `04_eval_detector`, provenance block, Wilson intervals, recall guard. |
| Hard-negative training | Option merged, result not yet accepted | `extra_negatives` in the train config. Repeat 20 cut nuScenes alerts from 80 to 46 per 404 frames (0.198 to 0.114 per frame) but D10 validation recall fell 0.586 to 0.513, so the recall guard FAILS. |

Not started: RAG (`rag/` is empty), `web/`, retrieval evaluation, synthetic fusion evaluation, hosted deployment.

## Now

**Retrain with `extra_negatives_repeat: 10`** (new run id, about 2-3 h on the GPU, keep the laptop awake), then run `03_eval_classifier`.

Exit when: the recall guard prints PASS for the new checkpoint (no class loses more than 3 points of validation recall; at least 5 of the 6 reviewed cracks still flagged), or the guard still fails and you decide to keep the baseline checkpoint. Either way, record the outcome in item 1 below.

## Next, in order

Do not start an item before the one above it meets its exit.

| # | Build | Exit when | Branch (suggested) |
|---|---|---|---|
| 0 | Decide ADR 0011 (core-first gate): accept or override | Status in the ADR changes from proposed; open question 9 closed | none, your decision |
| 1 | Pick the classifier checkpoint and thresholds | One checkpoint named in config; its thresholds set from `03_eval_classifier` and listed in `configs/`; guard result recorded here | `agent/defects-ship-checkpoint` |
| 2 | Synthetic fusion evaluation (`eval/fusion_synthetic.py`, a tool, tests) | Pastes RDD2022 patches on nuScenes road regions at known spots; reports detection rate vs distance, box IoU, `in_road` accuracy and `near_objects` accuracy; pass criteria live in config. This also gives the effective detection range the README must state. | `agent/eval-synthetic-fusion` |
| 3 | Real `/analyze` end to end with stubs off | 20 fixture images give valid HazardRecords; no stub path left in `api/pipeline.py`; overlay checked by eye on all 20 | `agent/api-real-pipeline` |
| 4 | Turn on severity | RDD2022 class meanings are pinned, so map D00/D10/D20/D40 to severity in config and feed `hazard_score`; contract unchanged, tests updated | `agent/fusion-severity` |
| 5 | Core report | README states the numbers from items 2-3, the limitations, and the effective range; ADR 0011 gate is met | `agent/docs-core-report` |
| 6 | Licences | Close D1, R1, R2 and the RDD2022 licence question; decide the hosted corpus (open question 10) | none, your decision |
| 7 | RAG: embeddings, Qdrant, retrieval, switchable LLM, `/index` and `/query` | Retrieval evaluation recall@k reported on queries generated from nuScenes annotations; both LLM backends answer a sample query (Claude behind a mock in CI) | `agent/rag-*` |
| 8 | Web UI (`web/`, Next.js plus the generated TypeScript client) | Analyze and Search views work against the real API; phone width; no nuScenes images bundled | `agent/web-ui` |
| 9 | Docker for real | `docker compose up` serves the API on the GPU host and on CPU-only; `/health` shows each model | `agent/docker-verify` |
| 10 | Deploy | Hosted demo up (HF Space plus Qdrant Cloud), spend cap on the Claude key, cached outputs for the sample set | `agent/deploy` |
| 11 | Final review | A reviewer who has not seen the build checks contracts, secrets, hardcoded paths and licence boundaries; walkthrough video recorded | none |

## Production-ready means all of these

- [ ] Core numbers (items 2 and 3) are in the README with their limits, not only in notebooks.
- [ ] Classifier checkpoint and thresholds are chosen and come from a PASSing recall guard.
- [ ] Stubs are off by default; `USE_STUBS` exists only for tests.
- [ ] Licences cleared for everything the hosted demo shows (RDD2022, COCO weights, road model); no nuScenes-derived content is shown.
- [ ] Docker build and compose have been run; CI green on `main`.
- [ ] The hosted demo meets the CPU latency target and the Claude spend cap works.
- [ ] Provisional values (classifier thresholds, `min_road`, detector score 0.5) are either re-measured on more data or labelled provisional in the README.

## Parked (deliberately not now)

| Item | Why parked |
|---|---|
| nuScenes v1.0-full and re-measuring the detector score | About 300 GB of disk; only matters if mini results are too thin to report |
| Fine-tuning the detector on nuScenes | Needs licence approval and a measured gap (D2) |
| Shoulder-edge crack loss, mask dilation (R3), Nordic snow check (R5) | Measure after the core report |
| Second human reviewer for the nuScenes alert labels | Current labels are one reviewer; do it if the guard result is borderline |
| `pip-audit` for the `ml` extra (R7) | Before deploy, not before the core report |
| Waymo, DQN gate, CARLA | Out of v1 ([ADR 0003](adr/0003-drop-dqn-gate-v1.md), [ADR 0012](adr/0012-drop-carla.md)) |

---

Related: [SPEC.md](../SPEC.md), [BUILD_PLAN.md](../BUILD_PLAN.md), [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md), [EVALUATION.md](EVALUATION.md), [QUALITY_BAR.md](QUALITY_BAR.md).
