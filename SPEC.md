Purpose: project specification for av-road-hazard v1 (goal, decisions, architecture, contracts, workstreams, evaluation, risks).
<!-- F:APP-A -->

# av-road-hazard: Project Spec (v1)

Private repo: `joshlaubach/av-road-hazard`. Target: demoable v1 in 1-2 weeks using parallel agents. Audience: portfolio / recruiters.

## 1. Goal

One end-to-end system that takes a front-camera driving frame and answers two questions: what is in the scene (objects) and is the road damaged (defects). It fuses both into a per-frame hazard record, indexes records and images for multimodal RAG, and exposes everything through an API and a web UI. It replaces two partial demos (nuscenes-stability and defect-detector, both run on Kaggle-style public datasets) with one deployed product.

## 2. Decisions locked in

| Area | Decision |
|---|---|
| Sources | Port code from `nuscenes-stability` and `defect-detector`; originals untouched |
| Weights | None exist; retrain both locally (RTX 5080) |
| Data | nuScenes v1.0-mini first, then v1.0-full; RDD2022; MVTec AD dropped from v1 |
| Domain gap | Road-masked overlapping tiling at native resolution (bottom third, near-field), auto-labeled non-road negatives added to training, measure with synthetic paste test |
| RAG store | Qdrant (Docker locally, Qdrant Cloud free tier hosted) |
| LLM | Switchable via config: LLaMA 3.2 Vision (Ollama, local) or Claude API (hosted; key available) |
| Front end | React/Next.js, backed by FastAPI |
| Hosting | Hugging Face Spaces (CPU) + Qdrant Cloud free tier; budget up to ~$20/mo |
| Hosted corpus | OPEN: RDD2022 samples (verify RDD2022 license before publishing any images), the owner's own photos, or both ([OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md)). |
| nuScenes images | Local only; never shown or redistributed in the hosted demo |
| Eval | No hand labeling (see section 7) |
| Local runtime | Docker Compose; GPU passthrough only on an NVIDIA host (Windows 11 + WSL2 or Linux). Training machine UNCONFIRMED: the source READMEs say Windows 11 + RTX 5080, but the linked dev computer is a macOS arm64 MacBook Pro (no NVIDIA GPU in Docker) |

## 3. Architecture

```
frame ─► perception/ (Faster R-CNN)  ─► objects[]
     └► defects/   (road crop → patch grid → EfficientNet-B4) ─► defects[]
                         │
                  fusion/ (patch→pixel boxes, road mask, overlap/proximity)
                         │
                  HazardRecord (JSON) ─► rag/ (CLIP embed + Qdrant) ─► LLM answer
                         │                                      │
                      FastAPI  ◄───────────────────────────────┘
                         │
                    Next.js UI
```

## 4. Repo layout

```
av-road-hazard/
  api/            # FastAPI app (main.py), request/response models, service pipeline
  common/         # config loader, JSON logging, debug dumps, progress bar, tracking, device, geometry, hashing (shared by all packages)
  configs/        # YAML configs: default.yaml, fusion.yaml, defects_train.yaml, defects_review.yaml
  defects/        # road-masked tiling, tile labels and index, classifier model, training, inference, evaluation, mining
  docker/         # Dockerfile.api, docker-compose.yml, compose.gpu.yml
  docs/           # architecture, evaluation, runbook, sources and licenses; adr/ (decision records); agents/ (workstream briefs)
  eval/           # per-model metrics (models.py), detector evaluation logic (detector.py), baselines.json
  fusion/         # fuse.py (hazard score), config.py, schema.py (HazardRecord contract), README.md
  perception/     # COCO object detector: detector.py (pinned weights, hash-checked load), infer.py (detect()), labels.py; train.py only if the detector is ever fine-tuned (ADR 0014)
  rag/            # package placeholder; planned: embed.py, index.py, retrieve.py, llm.py (switchable)
  web/            # Next.js app (planned, not created yet)
  tools/          # command-line tools run with python -m tools.NAME, e.g. export_openapi, fetch_detector, eval_detector, validate_annotations, review_nuscenes, overlay
  tests/          # unit, contract, integration, ml, data, e2e, defects, eval, fusion, perception, fixtures
  checkpoints/    # MANIFEST.json; model weights live here, never committed
  data/           # MANIFEST.json; datasets are local only, no data committed
  openapi.json    # exported API contract (CI fails on drift)
  SPEC.md  README.md  .env.example
```

Rules: separate packages (the two source repos both have a `src/`); no hardcoded paths; seeds fixed; `num_workers` configurable (source repos hardcode 0 for Windows).

## 5. Interfaces (contract for parallel work)

**Object detector:** `detect(frame: np.ndarray[RGB uint8]) -> list[Object]`, `Object = {label, score, box}`, where `box` is pixel `(x1, y1, x2, y2)` (top-left origin, `x1<=x2`, `y1<=y2`); `image_size` is `(width, height)`.

**Defect classifier:** `find_defects(frame) -> list[Defect]`, `Defect = {cls, name, confidence, box, patch_row, patch_col}`. Multi-label sigmoid over 4 RDD2022 classes (D00, D10, D20, D40), threshold from config.

**HazardRecord** (one per frame):

```json
{
  "frame_id": "str",
  "source": "rdd2022|upload|nuscenes",
  "image_size": [1600, 900],
  "objects": [{"label": "pedestrian", "score": 0.91, "box": [x1,y1,x2,y2]}],
  "defects": [{"name": "pothole", "confidence": 0.84, "box": [x1,y1,x2,y2],
               "severity": "high|medium|low|unknown", "in_road": true,
               "near_objects": [{"label": "pedestrian", "iou": 0.0, "dist_px": 42}]}],
  "hazard_score": 0.0,
  "meta": {"time_of_day": "day|night|unknown", "location": "str|null"}
}
```

`hazard_score` is a documented, deterministic formula (defect severity x confidence x proximity to vulnerable objects), not a learned value.

**API:**
- `POST /analyze` (image upload) → HazardRecord
- `POST /index` (image or batch) → adds to Qdrant
- `POST /query` `{text?, image?, filters?, top_k, llm: "llama|claude"}` → answer + retrieved records
- `GET /records/{id}`, `GET /health` (model and LLM status)

## 6. Workstreams (for parallel agents)

1. **Perception:** extract Faster R-CNN from notebook 02 into `perception/`, train on v1.0-mini, save checkpoint, `detect()` + tests. (Superseded by [ADR 0014](docs/adr/0014-object-detector.md): zero-shot COCO weights, no training unless a measured gap and licence approval justify it.)
2. **Defects:** download RDD2022, retrain EfficientNet-B4 (watch the 0.92 train / 0.73 val overfit gap in the original). Replace the fixed 8x8 grid with road-masked tiling: mask the drivable road (nuScenes map data or a road segmentation model), tile only that region with overlap at native resolution (focus on the bottom third, where defects are large enough), upscale tiles to classifier input size. Add non-road tiles from nuScenes frames as automatically labeled negatives (labeled by the road mask, no hand labeling) so the classifier learns sky, vehicles, and buildings are not defects. Then `find_defects()` + tests.
3. **Fusion:** schema, patch→box mapping, road mask, proximity logic, hazard score; unit tests with synthetic inputs. Depends only on the interfaces above, so it can start with stubs.
4. **RAG:** CLIP embeddings, Qdrant collections (image vector + text vector + metadata payload), filtered retrieval, switchable LLM wrapper, Ollama and Claude backends.
5. **API + Docker:** FastAPI, compose files (api, qdrant, ollama), GPU config, CUDA 12.8 base image for the 5080.
6. **Web UI:** upload and analyze view with overlays (boxes, severity colors), search/chat view, model toggle.
7. **Eval + data:** nuScenes loader, evaluation scripts.
8. **Deploy:** HF Space, Qdrant Cloud, secrets, CPU-only config, README and walkthrough video.

Streams 3-6 can begin immediately against stubs of the interfaces; 1-2 gate end-to-end integration.

## 7. Evaluation (no manual labeling)

1. **Per-model:** detector P/R/F1, COCO AP, condition slices (reuse nuscenes-stability metrics); defect classifier macro-F1 on RDD2022 validation.
2. **Retrieval:** auto-generate queries from nuScenes ground-truth annotations (for example "frames with a pedestrian at night"), score recall@k against annotation-derived answers.
3. **Fusion:** synthetic test pasting RDD2022 defect patches onto nuScenes road regions at known positions; verify detected defect box, `in_road`, and `near_objects` against known placement.

Known confounds to report honestly: in v1.0-mini, night and Singapore location are perfectly correlated and there are no rainy frames; evaluation on v1.0-full resolves this.

## 8. Milestones

- **Days 1-2:** repo, schema, interfaces and stubs, Docker skeleton, data downloads started.
- **Days 3-5:** both models retrained, fusion working on real outputs, indexing pipeline.
- **Days 6-8:** RAG + switchable LLM, API complete, UI upload and search views.
- **Days 9-10:** eval scripts and numbers, hosted deployment.
- **Days 11-14:** polish, README with metrics and limitations, walkthrough video, buffer.

## 9. Risks

- **Domain gap** (RDD2022 close-ups vs wide windshield frames): mitigated by road-masked tiling plus auto-labeled negatives; far-field defects (a few pixels) will still be missed, so state the effective detection range in the README. Report results either way.
- **Retraining time** for two models on the 5080 competes with the 1-2 week timeline; start training early, run in the background.
- **Label-name mismatch in source (verify before porting):** `defect-detector/src/datasets/rdd2022.py` documents D00=pothole, D10=longitudinal, D20=transverse, D40=alligator, but the official RDD convention is D00=longitudinal crack, D10=transverse crack, D20=alligator crack, D40=pothole. If the source is wrong, its per-class F1 names (Pothole 0.69, Longitudinal 0.65, Transverse 0.84, Alligator 0.73) are mislabeled, and anything keyed on "pothole" (severity, hazard score) would be wrong. Check against the official RDD2022 documentation first.
- **Over-firing classifier:** the source classifier flagged 31 of 64 patches on a sample frame at threshold 0.5; expect many false positives until negatives, regularization, and threshold tuning are in.
- **Licensing:** nuScenes stays local; RDD2022 license is unconfirmed (the figshare page names none), so verify before publishing its images.
- **HF Spaces CPU latency:** EfficientNet-B4 over 64 patches plus Faster R-CNN on CPU is slow; limit patch count, downscale, cache results, or precompute for sample images.
- **React plus Python stack** increases integration surface; keep the OpenAPI schema as the single contract and generate the TypeScript client from it.

## 10. Open items

- Confirm RDD2022 license terms.
- Hosted corpus source: RDD2022 samples (after license verification), the owner's own photos, or both ([OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md), row 10).

---

Related: [BUILD_PLAN.md](BUILD_PLAN.md), [ENGINEERING.md](ENGINEERING.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/EVALUATION.md](docs/EVALUATION.md), [docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md), [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md), [CONTRACT_CHANGELOG.md](CONTRACT_CHANGELOG.md), [docs/adr/](docs/adr/).
