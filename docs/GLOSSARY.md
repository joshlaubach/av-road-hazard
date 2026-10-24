Purpose: definitions of terms used across av-road-hazard docs.

# Glossary

| Term | Meaning |
|---|---|
| EVAV / AV | "EVAV" is the maintainer's shorthand for their autonomous-vehicle (AV) projects. In this repo: front-camera perception for autonomous vehicles. <!-- F:U-02 --> |
| RDD2022 | Road Damage Detection dataset 2022 (CRDDC 2022 on figshare). Close-up road photos with damage annotations. Training data for the defect classifier. License unverified |
| RDD class codes | D00, D10, D20, D40. **Label-name caveat:** the source repo maps D00 pothole, D10 longitudinal, D20 transverse, D40 alligator; the commonly documented convention is D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole. UNVERIFIED; workstream B verifies against official docs and annotation XML before porting |
| nuScenes | Autonomous driving dataset with 3D annotations. Non-commercial; images stay local |
| scene | In nuScenes, a roughly 20-second driving sequence. v1.0-mini has 10; v1.0-full has 700 |
| sample | In nuScenes, a keyframe within a scene (annotated frame) |
| CAM_FRONT | nuScenes front camera channel, used for detection |
| v1.0-mini | nuScenes mini split (about 400 keyframes, 10 scenes; UNVERIFIED). Pipeline validation only |
| HazardRecord | One JSON record per frame: objects, defects with severity, `in_road`, `near_objects`, `hazard_score`, meta |
| tile vs patch | Patch: a cell of the source's fixed 8x8 grid (64 per frame). Tile: a road-masked, overlapping crop at native resolution upscaled to 380x380, used by this repo |
| road mask | Binary mask of the drivable road region (from nuScenes map data or a road segmentation model; method undecided) |
| near-field | The road area close to the camera (bottom third of the frame) where defects are large enough to classify |
| contract | An interface frozen after Phase 0: `fusion/schema.py`, `openapi.json`, the `Object`/`Defect`/`HazardRecord` shapes, cross-package config keys |
| lane | Claude lane or Codex lane: the split of workstreams between the two agent tools |
| brief | A self-contained per-workstream instruction file in `docs/agents/` |
| sprint | A planned block of days with lane assignments (BUILD_PLAN section 1) |
| ADR | Architecture decision record in `docs/adr/` |
| hazard score | Deterministic formula: defect severity x confidence x proximity to vulnerable objects. Not learned |
| RAG | Retrieval-augmented generation: retrieve frame records, then answer grounded in them |
| Qdrant | Vector database used for the RAG store (Docker locally, Qdrant Cloud free tier hosted) |
| Ollama | Local LLM runtime serving LLaMA 3.2 Vision at `http://localhost:11434` |
| CARLA | Driving simulator that was considered as a frame generator; dropped on 2026-10-23 and not used ([ADR 0012](adr/0012-drop-carla.md)). |
| Stub | Fixed fake output of `detect()`/`find_defects()` behind `USE_STUBS=1` |
| Core-first gate | Proposed rule: RAG, React UI, and deploy stay at stub level until the core metrics exist |
