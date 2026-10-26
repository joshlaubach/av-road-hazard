Purpose: architecture of av-road-hazard: data flow per stage, the HazardRecord schema, API contract, GPU/CPU placement, and local vs hosted topology.

# Architecture

Authoritative contracts are in [SPEC.md](../SPEC.md) section 5. This document explains them.

## Diagram

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

## Data flow per stage

| Stage | Package | Input | Output | Notes |
|---|---|---|---|---|
| 1. Object detection | `perception/` | RGB uint8 frame | `list[Object]` (`label`, `score`, `box`) | Faster R-CNN ResNet50 FPN v2, trained on nuScenes CAM_FRONT; batch and device-aware; CPU path for hosting |
| 2. Defect detection | `defects/` | RGB frame | `list[Defect]` (`cls`, `name`, `confidence`, `box`, `patch_row`, `patch_col`) | Road crop, overlapping tiles at native resolution on the near-field road (bottom third), upscaled to 380x380, EfficientNet-B4 multi-label sigmoid over 4 RDD2022 classes (D00, D10, D20, D40), threshold from config. Replaces the source's fixed 8x8 grid. |
| 3. Fusion | `fusion/` | objects, defects, road mask | `HazardRecord` | Pure functions, no model imports. Maps patch to pixel boxes, intersects with road mask (`in_road`), finds nearest objects (`near_objects`: IoU and pixel distance), computes `hazard_score` |
| 4. Index | `rag/` | `HazardRecord` + image | Qdrant point | Image embedding (CLIP-style), text embedding of the record and report, metadata payload, one collection |
| 5. Retrieve and answer | `rag/` | text and/or image query, filters | answer + retrieved records | Metadata filter, then vector search (image and text), then rerank; LLM grounded only in retrieved records, citing frame IDs |
| 6. Serve | `api/` | HTTP | JSON | FastAPI; a `Pipeline` service object loads models once, lazily |
| 7. UI | `web/` | API | pages | Next.js with a client generated from `openapi.json` |

## HazardRecord

One record per frame.

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

### Field semantics

| Field | Meaning |
|---|---|
| `frame_id` | Unique id of the frame; cited by RAG answers |
| `source` | Where the frame came from: `rdd2022`, `upload`, `nuscenes` |
| `image_size` | Image size in pixels; the example is `[1600, 900]` (nuScenes). Ordering (width, height) must be pinned in `fusion/schema.py`; image size ordering is a likely coordinate bug (BUILD_PLAN section 4) |
| `objects[]` | Detections: `label`, `score`, `box` as `[x1,y1,x2,y2]` pixels |
| `defects[].name` | Defect class name. Depends on the RDD2022 label mapping, which must be verified first (see [SOURCE_REPOS.md](SOURCE_REPOS.md), S-11) |
| `defects[].confidence` | Classifier sigmoid probability |
| `defects[].box` | Tile-to-frame mapped pixel box |
| `defects[].severity` | `high`, `medium`, `low`, or `unknown`; a heuristic, or optional LLM text (off by default) |
| `defects[].in_road` | `true` when at least a configured fraction of the defect box lies on the road mask, `false` when confirmed off the road, `null` when unknown (no usable road mask). Fusion scores unknown like `true`. Readers must not test truthiness (`null` is falsy): use `is not False` to keep unknown hazards. See CONTRACT_CHANGELOG v0.2.0 |
| `defects[].near_objects[]` | Nearby objects: `label`, `iou`, `dist_px` (pixel distance) |
| `hazard_score` | Documented deterministic formula (defect severity x confidence x proximity to vulnerable objects), not learned. The formula is documented in `fusion/README.md` once built |
| `meta.time_of_day` | `day`, `night`, `unknown` |
| `meta.location` | string or null |

The fused output is one JSON `HazardRecord` per frame with objects, defects with severity, `in_road`, and `near_objects`; `hazard_score` is deterministic, not learned. <!-- F:R-08 -->

The per-stage Python interfaces are `detect(frame) -> list[Object]` with `Object = {label, score, box}` and `find_defects(frame) -> list[Defect]` with `Defect = {cls, name, confidence, box, patch_row, patch_col}`.

## API contract

| Route | Purpose |
|---|---|
| `POST /analyze` | image upload, returns a `HazardRecord` |
| `POST /index` | image or batch, adds to Qdrant |
| `POST /query` | `{text?, image?, filters?, top_k, llm: "llama\|claude"}`, returns answer plus retrieved records |
| `GET /records/{id}` | one record |
| `GET /health` | model and LLM status, Qdrant connectivity, versions |

`openapi.json` is the single contract between the Python API and the TypeScript client; the client is generated from it. Changes need a `contract-change` PR.

## RAG design

Index per frame: an image embedding (CLIP-style), a text embedding of the structured record and report, and metadata (scene, time of day, location, labels, severity, hazard score), all in one Qdrant collection. Retrieval is metadata filter, then vector search (image and text), then rerank. Structured queries such as severity above X run as plain filters. Generation uses a vision-language model, grounded only in retrieved records, citing frame IDs. Evaluation uses queries auto-generated from nuScenes annotations. <!-- F:R-09 -->

## Domain-gap handling

Crop to the drivable-road region (nuScenes map data or a road segmentation model; the exact method is undecided), classify at higher zoom, measure first, fine-tune only if needed; negatives are auto-labeled by the road mask. <!-- F:R-11 -->

## GPU/CPU placement per stage

| Stage | Local (NVIDIA host) | Local (Mac) | Hosted (HF Spaces) |
|---|---|---|---|
| Detection | GPU | CPU or native MPS | CPU |
| Defect classification | GPU | CPU or native MPS | CPU (limit tiles, downscale, cache, precompute) |
| Embeddings | GPU | CPU (feasible for a few thousand frames) | CPU, precomputed for sample set |
| Vector search, filtering | CPU | CPU | CPU (Qdrant Cloud) |
| LLM | Ollama on GPU, or Claude API | Ollama native, or Claude API | Claude API (the hosted demo has no GPU for Ollama; Claude key is a Space secret) |

Details and sizes: [ENVIRONMENT.md](ENVIRONMENT.md). The Mac column is derived from the Docker GPU rule: containers on macOS are CPU-only, native runs can use MPS.

## Topology: what runs where

| Component | Local | Hosted |
|---|---|---|
| FastAPI service (both detectors) | Docker container | HF Space (CPU Docker) |
| Ollama (LLaMA 3.2 Vision) | Docker container (GPU on NVIDIA hosts) or host | not used by default |
| Qdrant | Docker container | Qdrant Cloud free tier |
| Indexing job | one-off Docker job | precomputed offline for the sample set |
| Training and full evaluation | owner's machine, not CI | n/a |
| Web UI | `web/` dev server or container | hosting location undecided (BUILD_PLAN section 5 names only the HF Space, Qdrant Cloud, and the Claude key as a Space secret) |
| Datasets and weights | mounted volumes, never in images | corpus is open: RDD2022 samples (after license verification), the owner's own photos, or both ([OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)) |

Compose files: `docker/docker-compose.yml` (api, qdrant, ollama) and `docker/compose.gpu.yml` for NVIDIA passthrough. CI never needs a GPU.
