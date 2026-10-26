Purpose: where av-road-hazard stands today, what is being worked on now, what to build next and in what order, and what "production-ready" means for this project.

# Roadmap

Last updated: 2026-10-26. Whoever finishes or changes a row should update this file in the same PR. If a row here disagrees with the code, the code is right and this file is out of date.

The order follows the core-first gate ([ADR 0011](adr/0011-core-first-gate.md)). The core (detector, defect classifier, fusion and the synthetic fusion evaluation) has to give reportable numbers before RAG, the web UI or deployment go beyond stubs.

## Done

| Piece | State | Evidence |
|---|---|---|
| Contracts, stub API, CI, Docker skeleton | Done | `fusion/schema.py`, `openapi.json`, `.github/workflows/ci.yml`. The Docker build and compose file have never been run (see [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)). |
| Object detector | Done, used zero-shot | Pretrained COCO Faster R-CNN with pinned weights ([ADR 0014](adr/0014-object-detector.md)). On nuScenes mini at score 0.5, the default MobileNet model finds 38% of pedestrians (90% range 35-42%) and 21% of cyclists (14-31%, from only 57 objects). Its CPU speed meets the p95 limit of 2 s at 24, 4 and 2 threads. |
| Road mask | Done | comma10k model, pinned and hash-checked, with `min_road` set to 0.4 ([ADR 0013](adr/0013-road-mask-model.md)). It is wired into `api/pipeline.py`. |
| Defect classifier | Trained, thresholds still provisional | EfficientNet-B4 on RDD2022 tiles. `find_defects` skips tiles covered by objects or off the road and uses per-class thresholds from config. The class names are verified against the paper and the annotation files, and pinned by a test. |
| Fusion | Done | `fusion/fuse.py` sets `in_road` and `near_objects` and computes the deterministic `hazard_score`. Severity is still "unknown" (see Next, row 4). |
| Evaluation tooling | Done | Code in `eval/`, plus notebooks `03_eval_classifier` and `04_eval_detector`. They report 90% Wilson intervals, a provenance block and a recall guard. |
| Hard-negative training | Option merged, result not accepted yet | Adding clean tiles with `extra_negatives` (repeated 20 times) cut nuScenes alerts from 80 to 46 over 404 frames. D10 validation recall also fell from 0.586 to 0.513, so the recall guard fails. |
| Core-first gate | Accepted 2026-10-26 | [ADR 0011](adr/0011-core-first-gate.md): RAG, the web UI and deployment stay at stub level until the core produces reportable numbers. Building `web/` against the stub API is allowed. |

Not started: RAG (`rag/` is empty), `web/`, the retrieval evaluation, the synthetic fusion evaluation and the hosted deployment.

## Now

Retrain with `extra_negatives_repeat: 10`, using a new run id (about 2-3 hours on the GPU; keep the laptop awake). The checked-in `configs/defects_train.yaml` has `extra_negatives: null`, so the run needs a config that points `extra_negatives` at the confirmed-clean tile file. Afterwards, check that the run's `best.json` shows a nonzero `extra_negative_tiles`, otherwise the run had no hard negatives at all.

Then run `03_eval_classifier` with the nuScenes review inputs. Without them the guard cannot check the reviewed cracks and can print PASS without looking at any. A real result lists how many of the 6 reviewed cracks are still flagged.

There are two valid outcomes, and both finish this step:
- The new checkpoint passes the guard: no class loses more than 3 points of validation recall, and at least 5 of the 6 reviewed cracks are still flagged.
- It fails, and you keep the baseline checkpoint. The baseline is the reference the guard compares against, so it cannot fail the guard. Choosing it is a recorded decision.

Write the outcome next to row 1 below.

## Next

Do these in order. A row should not start until the one above it has met its exit condition.

| # | What to build | Finished when | Suggested branch |
|---|---|---|---|
| 1 | Choose the classifier checkpoint and its thresholds | One checkpoint is named in config, its thresholds come from `03_eval_classifier`, and the guard result (or the decision to keep the baseline) is written here | `agent/defects-ship-checkpoint` |
| 2 | Run `/analyze` end to end with the stubs off. Code is done: `python -m tools.smoke_analyze FOLDER --out DIR` (needs `USE_STUBS=0`), `/health` reporting the real model states, and a 503 when the classifier is missing. Still to do: run it on the 20 fixtures with the real models, after the retrain frees the GPU. | 20 fixture images give valid HazardRecords and no stub path is left in `api/pipeline.py`. The road model must load, and designated fixtures must give a non-null `in_road`. Failures are counted separately from valid records, because a failed road model still produces valid records with an unknown `in_road`. Overlays for all 20 are checked by eye against the source images, since the schema only checks box corner order and shifted tile-to-frame coordinates would still validate. | `agent/api-real-pipeline` |
| 3 | Synthetic fusion evaluation (`eval/fusion_synthetic.py`, a tool and tests) | RDD2022 patches are pasted onto nuScenes road regions at known positions, and the report gives detection rate by position and size in the image, box IoU, `in_road` accuracy and `near_objects` accuracy. The pass criteria are in config. `in_road` is scored against road ground truth that does not come from the road model being tested (for example nuScenes map data). Positions are image coordinates, so a range in metres needs a stated way to assign distance, with its uncertainty; otherwise the README reports image-space size, not physical range. It runs through the real pipeline from row 2. | `agent/eval-synthetic-fusion` |
| 3b | Construction and work-zone labels | Step 1 (in review): the labels, provisional weights, and the notebook report of what the detector cannot name. Step 2: run notebook 04 to record the gap numbers here. Step 3 (done, recorded 2026-10-26): Motional confirmed that a portfolio project counts as non-commercial if it is not used in actual production, and that a public demo may serve a model fine-tuned on nuScenes (see [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md)). Step 4: fine-tune the detector on nuScenes for the new labels and report its recall. Weights stay out of the repo and carry the nuScenes terms. | `agent/construction-labels`, then `agent/detector-finetune` |
| 4 | Turn on severity | The class names are verified (D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole). Map them to a severity in config and feed `hazard_score`. The contract does not change. | `agent/fusion-severity` |
| 5 | Write up the core results | `eval/results.json` holds the per-model and synthetic results, including the false-positive rate on non-road tiles. The README states those numbers, the limitations and the effective range, and the gate in ADR 0011 is met. | `agent/docs-core-report` |
| 6 | Clear the licences | D1, R1, R2 and the RDD2022 licence question (partly done: working rule recorded 2026-10-26, authors still to be asked) are closed, and the hosted corpus is chosen (open question 10) | none, this is a decision |
| 6b | Collect the own-photos corpus (the plan while RDD2022 publication is blocked) | 30 to 50 dashcam-angle photos of cracks, potholes and clean road are in `data/own_photos/`, which git already ignores. Location data is stripped (re-saving the image with Pillow drops EXIF), faces and number plates are cropped or blurred, and a small file lists each photo with what it shows (crack, pothole or clean) so retrieval can be checked against known answers. The photos are your own work, so there is no licence to clear; the README credits you. | `agent/own-photos` (tooling only; the photos are never committed) |
| 7 | RAG: embeddings, Qdrant, retrieval, switchable LLM, `/index` and `/query` | Recall@k is reported on queries generated from nuScenes annotations and added to `eval/results.json`, retrieval on the hosted corpus (row 6b) is checked against known relevant results, and both LLM backends answer a sample query (the Claude one behind a mock in CI). Uploads are indexed and retrieved per session, reading a record checks access, and a test shows one session cannot see another's upload (today `api/main.py` keeps all uploads in one shared dictionary) | `agent/rag-*` |
| 8 | Web UI in `web/` (Next.js with the generated TypeScript client) | The Analyze and Search views work against the real API at phone width, with no nuScenes images bundled. Plain-language (ELI16) explanations are the default text in the UI: what the pipeline does, why a result was flagged, and how to read the hazard score. Technical detail sits behind an expandable section and is written as mathematics rendered with KaTeX (definitions, formulas and units), not as code or computer-science terms, because the owner's background is mathematics. An animated walkthrough shows what happens to an uploaded photo, step by step: the detector boxes, the road mask, the tile grid sliding over the road, tiles skipped under objects, each tile's probability against its threshold, and the fusion into a hazard score. It plays from the real record for that image, so what it shows is what the pipeline did. The hazard score is shown to people as a whole number from 0 to 100 (the stored `hazard_score` stays between 0 and 1, so the contract does not change). | `agent/web-ui` |
| 9 | Run Docker for real | On both the GPU host and a CPU-only host, `USE_STUBS` is 0, the checkpoints are present, `/health` reports every model as ready, and a real `/analyze` request succeeds. The compose file currently sets `USE_STUBS: "1"`, so starting it proves nothing yet. | `agent/docker-verify` |
| 10 | Harden the API for hosting | Everything here is built and tested before the release review. There is a per-IP rate limit and a size limit on the whole request body, applied before the body is parsed, for every endpoint including the JSON image field of `/query`. File and image limits (bytes, type, pixels) hold, `text` and `filters` are bounded, and oversized requests are tested. EXIF GPS is stripped from stored images. The Claude spend cap has a hard cutoff and a fallback message. `pip-audit` has been run on an environment installed with the `ml` extra, and every finding is either fixed or recorded as a reviewed exception with an end date (R7). | `agent/api-hardening` |
| 11 | Release review, before anything goes online | Someone who has not seen the build checks contracts, secrets, hardcoded paths and licence boundaries in the exact code and bundle that will be deployed. Any change after the review, including fixes for its findings, needs a fresh look at that change. | none |
| 12 | Deploy | That reviewed bundle is running on an HF Space with Qdrant Cloud, outputs for the sample set are cached, a smoke check passes on the live demo, and the walkthrough video is recorded | `agent/deploy` |

## Production-ready means

- [ ] The core numbers from rows 3 and 5 are in the README with their limits, not only in notebooks.
- [ ] The classifier checkpoint and thresholds are chosen, and the choice is backed by a recorded guard result.
- [ ] The stubs are off by default, and `USE_STUBS` is used only in tests.
- [ ] Everything the hosted demo shows has a cleared licence (RDD2022, COCO weights, road model), and nothing derived from nuScenes is shown.
- [ ] The Docker build and compose file have been run with real models, and CI is green on `main`.
- [ ] The hosted demo meets the CPU latency target, and the rate limit, request and upload limits and Claude spend cap are tested.
- [ ] The provisional values (classifier thresholds, `min_road`, detector score 0.5) are either re-measured on more data or marked provisional in the README.

## Parked

| Item | Why it is parked |
|---|---|
| nuScenes v1.0-full and re-measuring the detector score | It needs about 300 GB of disk, and it only matters if the mini results are too thin to report |
| Fine-tuning the detector on nuScenes | It needs licence approval and a measured gap first (D2) |
| Shoulder-edge crack loss and mask dilation (R3), Nordic snow check (R5) | Measure these after the core write-up |
| A second human reviewer for the nuScenes alert labels | The labels come from one reviewer, so do this if the guard result is close |

---

Related: [SPEC.md](../SPEC.md), [BUILD_PLAN.md](../BUILD_PLAN.md), [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md), [EVALUATION.md](EVALUATION.md), [QUALITY_BAR.md](QUALITY_BAR.md).
