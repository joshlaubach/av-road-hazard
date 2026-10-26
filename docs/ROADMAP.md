Purpose: where av-road-hazard stands today, what is being worked on now, what to build next and in what order, and what "production-ready" means for this project.

# Roadmap

Last updated: 2026-10-26 (finished rows moved to Done; Next renumbered). Whoever finishes or changes a row should update this file in the same PR. If a row here disagrees with the code, the code is right and this file is out of date.

The order follows the core-first gate ([ADR 0011](adr/0011-core-first-gate.md)). The core (detector, defect classifier, fusion and the synthetic fusion evaluation) has to give reportable numbers before RAG, the web UI or deployment go beyond stubs.

## Done

| Piece | State | Evidence |
|---|---|---|
| Contracts, stub API, CI, Docker skeleton | Done | `fusion/schema.py`, `openapi.json`, `.github/workflows/ci.yml`. The Docker build and compose file have never been run (see [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)). |
| Object detector | Done, used zero-shot | Pretrained COCO Faster R-CNN with pinned weights ([ADR 0014](adr/0014-object-detector.md)). On nuScenes mini at score 0.5, the default MobileNet model finds 38% of pedestrians (90% range 35-42%) and 21% of cyclists (14-31%, from only 57 objects). Its CPU speed meets the p95 limit of 2 s at 24, 4 and 2 threads. |
| Road mask | Done | comma10k model, pinned and hash-checked, with `min_road` set to 0.4 ([ADR 0013](adr/0013-road-mask-model.md)). It is wired into `api/pipeline.py`. |
| Defect classifier (was row 1) | Checkpoint chosen 2026-10-26, thresholds still provisional | Run `tile-fold0-hn2` (SHA-256 starts `762f4b38`), EfficientNet-B4 on RDD2022 tiles. `find_defects` skips tiles covered by objects or off the road and uses per-class thresholds from config. The class names are verified against the paper and the annotation files, and pinned by a test. |
| Fusion | Done | `fusion/fuse.py` sets `in_road` and `near_objects` and computes the deterministic `hazard_score`. Severity is still "unknown" (see Next, row 3). |
| Evaluation tooling | Done | Code in `eval/`, plus notebooks `03_eval_classifier` and `04_eval_detector`. They report 90% Wilson intervals, a provenance block and a recall guard. |
| Hard-negative training | Done, repeat 10 accepted 2026-10-26 | `extra_negatives` repeated 10 times (run `tile-fold0-hn2`, 680 extra tiles) cut nuScenes alerts from 80 to 53 over 404 frames (0.198 to 0.131 per frame) and the clean probe tiles still flagged from 23 to 13 of 23. The recall guard passes: validation recall changed by -1.9 (D00), +3.3 (D10), +2.8 (D20) and +0.9 (D40) points, and 5 of the 6 reviewed cracks are still flagged. Repeat 20 failed the guard (D10 lost 6.6 points, 5 of 6 cracks). Limits: one fold, one reviewer, 5 of 6 is the lowest passing score, and 13 of the 53 alerts have not been reviewed yet. |
| Real pipeline end to end (was row 2) | Done on local images 2026-10-26 | `python -m tools.smoke_analyze` with `USE_STUBS=0` and the chosen checkpoint, on 20 local images (12 nuScenes frames, day and night, and 8 RDD2022 images from Japan, Czech, Norway and India; none are committed). All 20 gave valid records, 0 failures, and the road model loaded for every image (no unknown `in_road`). Overlays for all 20 checked by eye: detector boxes sit on the cars, trucks, buses and people, and defect boxes line up with the tile grid on the road, so no shifted coordinates. Defects were raised on 3 of 20 images. Findings: (1) one night frame has a large "car" box around headlight glare, and one Czech/India frame has a box on the car's own hood or bumper edge, so the detector gives occasional false boxes; (2) a Norwegian frame gave 15 overlapping defect boxes, because neighboring flagged tiles are not merged into one defect; (3) the app reads real environment variables only, not `.env`, so `DEFECT_CHECKPOINT` must be set in the shell. `api/pipeline.py` keeps its stub switch for tests only. Still open: the 20 license-clean fixture images for CI (see Next, row 6). |
| Core-first gate | Accepted 2026-10-26 | [ADR 0011](adr/0011-core-first-gate.md): RAG, the web UI and deployment stay at stub level until the core produces reportable numbers. Building `web/` against the stub API is allowed. |

Not started: RAG (`rag/` is empty), `web/`, the retrieval evaluation, the synthetic fusion evaluation and the hosted deployment.

## Now

- Review the 13 nuScenes alerts the chosen classifier raised that nobody has judged (exported as an annotation batch by `tools/eval_classifier.py`). They show whether the real false-alarm drop is bigger or smaller than the Done table says.
- Start Next, row 1.

## Next

Do these in order. A row should not start until the one above it has met its exit condition.

| # | What to build | Finished when | Suggested branch |
|---|---|---|---|
| 1 | Synthetic fusion evaluation (`eval/fusion_synthetic.py`, a tool and tests) | RDD2022 patches are pasted onto nuScenes road regions at known positions, and the report gives detection rate by position and size in the image, box IoU, `in_road` accuracy and `near_objects` accuracy. The pass criteria are in config. `in_road` is scored against road ground truth that does not come from the road model being tested (for example nuScenes map data). Positions are image coordinates, so a range in meters needs a stated way to assign distance, with its uncertainty; otherwise the README reports image-space size, not physical range. It runs through the real pipeline (see Done). | `agent/eval-synthetic-fusion` |
| 2 | Construction and work-zone labels | Step 1 (done, merged): the labels, provisional weights, and the notebook report of what the detector cannot name. Step 2: run notebook 04 to record the gap numbers here. Step 3 (done, recorded 2026-10-26): Motional confirmed that a portfolio project counts as non-commercial if it is not used in actual production, and that a public demo may serve a model fine-tuned on nuScenes (see [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md)). Step 4: fine-tune the detector on nuScenes for the new labels and report its recall. Weights stay out of the repo and carry the nuScenes terms. | `agent/detector-finetune` |
| 3 | Merge neighboring flagged tiles, then turn on severity | First, neighboring flagged tiles become one defect box instead of many overlapping ones (one Norwegian frame gave 15). Then map each class name (D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole) to a severity in config and feed `hazard_score`. The contract does not change. | `agent/fusion-severity` |
| 4 | Write up the core results | `eval/results.json` holds the per-model and synthetic results, including the false-positive rate on non-road tiles. The README states those numbers, the limitations and the effective range, and the gate in ADR 0011 is met. | `agent/docs-core-report` |
| 5 | Clear the licenses | D1, R1, R2 and the RDD2022 license question (partly done: working rule recorded 2026-10-26, authors still to be asked) are closed, and the hosted corpus is chosen (open question 10) | none, this is a decision |
| 6 | Collect license-clean images: the own-photos corpus (the plan while RDD2022 publication is blocked) and the 20 CI fixtures | The CI fixtures are 20 small images in `tests/fixtures/` that may be committed (the owner's own photos, or CC0 or public-domain photos with the source and license recorded; the CC0 Stellenbosch pothole dataset is a candidate), and the integration test runs the real pipeline on them. The corpus is 30 to 50 dashcam-angle photos of cracks, potholes and clean road are in `data/own_photos/`, which git already ignores. Location data is stripped (re-saving the image with Pillow drops EXIF), faces and number plates are cropped or blurred, and a small file lists each photo with what it shows (crack, pothole or clean) so retrieval can be checked against known answers. The photos are your own work, so there is no license to clear; the README credits you. | `agent/own-photos` (tooling only; the photos are never committed) |
| 7 | RAG: embeddings, Qdrant, retrieval, switchable LLM, `/index` and `/query` | Recall@k is reported on queries generated from nuScenes annotations and added to `eval/results.json`, retrieval on the hosted corpus (row 6) is checked against known relevant results, and both LLM backends answer a sample query (the Claude one behind a mock in CI). Uploads are indexed and retrieved per session, reading a record checks access, and a test shows one session cannot see another's upload (today `api/main.py` keeps all uploads in one shared dictionary) | `agent/rag-*` |
| 8 | Web UI in `web/` (Next.js with the generated TypeScript client) | The Analyze and Search views work against the real API at phone width, with no nuScenes images bundled. Plain-language (ELI16) explanations are the default text in the UI: what the pipeline does, why a result was flagged, and how to read the hazard score. Technical detail sits behind an expandable section and is written as mathematics rendered with KaTeX (definitions, formulas and units), not as code or computer-science terms, because the owner's background is mathematics. An animated walkthrough shows what happens to an uploaded photo, step by step: the detector boxes, the road mask, the tile grid sliding over the road, tiles skipped under objects, each tile's probability against its threshold, and the fusion into a hazard score. It plays from the real record for that image, so what it shows is what the pipeline did. The hazard score is shown to people as a whole number from 0 to 100 (the stored `hazard_score` stays between 0 and 1, so the contract does not change), drawn as a speedometer-style half-circle gauge: 0 at the left end, 100 at the right, with a blue, green, yellow, orange, red gradient. The sample image in the walkthrough must be one we may publish (the owner's own photo, or a CC0 or public-domain photo with its source recorded), never a nuScenes or RDD2022 image. | `agent/web-ui` |
| 9 | Run Docker for real | On both the GPU host and a CPU-only host, `USE_STUBS` is 0, the checkpoints are present, `/health` reports every model as ready, and a real `/analyze` request succeeds. The compose file currently sets `USE_STUBS: "1"`, so starting it proves nothing yet. | `agent/docker-verify` |
| 10 | Harden the API for hosting | Everything here is built and tested before the release review. There is a per-IP rate limit and a size limit on the whole request body, applied before the body is parsed, for every endpoint including the JSON image field of `/query`. File and image limits (bytes, type, pixels) hold, `text` and `filters` are bounded, and oversized requests are tested. EXIF GPS is stripped from stored images. The Claude spend cap has a hard cutoff and a fallback message. `pip-audit` has been run on an environment installed with the `ml` extra, and every finding is either fixed or recorded as a reviewed exception with an end date (R7). | `agent/api-hardening` |
| 11 | Release review, before anything goes online | Someone who has not seen the build checks contracts, secrets, hardcoded paths and license boundaries in the exact code and bundle that will be deployed. Any change after the review, including fixes for its findings, needs a fresh look at that change. | none |
| 12 | Deploy | That reviewed bundle is running on an HF Space with Qdrant Cloud, outputs for the sample set are cached, a smoke check passes on the live demo, and the walkthrough video is recorded | `agent/deploy` |

## Production-ready means

- [ ] The core numbers from rows 1 and 4 are in the README with their limits, not only in notebooks.
- [ ] The classifier checkpoint and thresholds are chosen, and the choice is backed by a recorded guard result.
- [ ] The stubs are off by default, and `USE_STUBS` is used only in tests.
- [ ] Everything the hosted demo shows has a cleared license (RDD2022, COCO weights, road model), and nothing derived from nuScenes is shown.
- [ ] The Docker build and compose file have been run with real models, and CI is green on `main`.
- [ ] The hosted demo meets the CPU latency target, and the rate limit, request and upload limits and Claude spend cap are tested.
- [ ] The provisional values (classifier thresholds, `min_road`, detector score 0.5) are either re-measured on more data or marked provisional in the README.

## Parked

| Item | Why it is parked |
|---|---|
| nuScenes v1.0-full and re-measuring the detector score | It needs about 300 GB of disk, and it only matters if the mini results are too thin to report |
| Fine-tuning the detector on nuScenes | It needs license approval and a measured gap first (D2) |
| Shoulder-edge crack loss and mask dilation (R3), Nordic snow check (R5) | Measure these after the core write-up |
| A second human reviewer for the nuScenes alert labels | The labels come from one reviewer, so do this if the guard result is close |

---

Related: [SPEC.md](../SPEC.md), [BUILD_PLAN.md](../BUILD_PLAN.md), [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md), [EVALUATION.md](EVALUATION.md), [QUALITY_BAR.md](QUALITY_BAR.md).
