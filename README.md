Purpose: entry point for av-road-hazard, a planned system that fuses road-object detection and road-defect detection into per-frame hazard records with multimodal search.

# av-road-hazard

> **Status: core under evaluation.** The object detector, road mask, defect classifier, fusion and evaluation notebooks exist and run on real models; RAG and the web UI do not. See [docs/ROADMAP.md](docs/ROADMAP.md) for what is done and what to build next. Parts of this documentation were drafted on 2026-10-21 from the source repositories and have not all been re-checked. <!-- F:P-07 -->

## What it is

av-road-hazard takes a front-camera driving frame and answers two questions: what is in the scene (objects) and is the road damaged (defects). It fuses both answers into one hazard record per frame, indexes records and images for multimodal retrieval (RAG), and will expose everything through an API and a web UI. It combines two earlier AV ("EVAV") projects, `nuscenes-stability` (a CV model that identifies common road objects) and `defect-detector` (identifies road defects), into one end-to-end system instead of partially completed demos on Kaggle-style public data. <!-- F:U-02 --> <!-- F:U-03 --> One project answers what is in the scene, the other whether the road is damaged. A pothole in the lane a pedestrian is walking toward is more urgent than the same pothole on an empty shoulder. <!-- F:U-03 -->

Spec details: [SPEC.md](SPEC.md). Repo: `joshlaubach/av-road-hazard`, private.

## Goals and constraints

- Audience is portfolio and recruiters. Interface is a web UI plus an API. The demo is hosted and public. <!-- F:U-06 -->
- Timeline target is a demoable v1 in 1-2 weeks, using parallel agents. Front end: React/Next.js (chosen over Streamlit; timeline tradeoff accepted, mitigated by an OpenAPI-generated client). <!-- F:U-07 -->
- A multimodal RAG layer sits on top of the fused output, queried with mixed visual and text criteria, for example "show me frames with a pothole in the lane near a pedestrian". <!-- F:U-05 -->
- No hand labeling: all evaluation is automatic. <!-- F:U-10 -->
- Hosted demo budget is up to about $20/month. A Claude API key exists; API usage is billed separately from the Claude subscription. A spend cap with hard cutoff is required. <!-- F:U-09 -->
- The repo `av-road-hazard` is private under `joshlaubach`. <!-- F:U-11 -->
- Engineering practices (requirements, design, testing, debugging, CI/CD, operations, security, documentation), agentic practices, and the full build plan are written into files. The first commit required every planned fact to be present in at least one file (see [docs/TRACEABILITY.md](docs/TRACEABILITY.md)). <!-- F:U-12 -->
- Documentation style: concise, direct, minimal formatting, no filler. <!-- F:U-15 -->
- Decisions: Qdrant; road-masked tiling with auto-labeled negatives; the three automatic evaluations; switchable LLM; Hugging Face Spaces plus Qdrant Cloud free tier; hosted corpus source open (RDD2022 samples after license verification, the owner's own photos, or both); React/Next.js; nuScenes v1.0-mini first then full; upload-your-own-image hosted demo model. <!-- F:U-16 -->

## Evidence standards

The supporting stack (RAG, two LLM backends, React, Docker, HF Spaces) sits on two models that do not yet work well on the target data: the classifier overfits (train F1 0.92 vs validation 0.73), the detector is trained on a 10-scene mini set, there is a possible domain gap, and there is no combined ground truth. Results therefore come first. See [docs/QUALITY_BAR.md](docs/QUALITY_BAR.md). Every claim in this README must be backed by a number from `eval/results.json` or marked as unmeasured. <!-- F:Q-05 --> <!-- F:Q-01 --> <!-- F:U-13 -->

## Architecture

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

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Planned features

Planned, none implemented:

1. Object detection on front-camera frames (Faster R-CNN, `detect(frame)`).
2. Road-defect detection with road-masked tiling and an EfficientNet-B4 classifier (`find_defects(frame)`).
3. Fusion into one `HazardRecord` JSON per frame (`in_road`, `near_objects`, deterministic `hazard_score`). <!-- F:R-08 -->
4. Multimodal RAG over records and images in Qdrant, with a switchable LLM (LLaMA 3.2 Vision via Ollama, or the Claude API). <!-- F:R-09 -->
5. FastAPI service: `POST /analyze`, `POST /index`, `POST /query`, `GET /records/{id}`, `GET /health`.
6. Next.js web UI: analyze view with overlays, search/chat view, model toggle.
7. Three automatic evaluations with no hand labeling (per-model metrics, retrieval, synthetic fusion).
8. Hosted demo on Hugging Face Spaces (CPU) with Qdrant Cloud free tier, upload-your-own-image.

Per the proposed core-first gate, items 4, 6 and 8 stay at stub level until items 1, 2, 3 and the synthetic fusion evaluation produce reportable numbers ([ADR 0011](docs/adr/0011-core-first-gate.md), status proposed). <!-- F:Q-04 -->

## Documentation map

| Document | What it is |
|---|---|
| [SPEC.md](SPEC.md) | Project spec v1 |
| [BUILD_PLAN.md](BUILD_PLAN.md) | Agent build plan, sprints, workstream briefs |
| [ENGINEERING.md](ENGINEERING.md) | Engineering practices and Definition of Done |
| [AGENTS.md](AGENTS.md) | Operating manual for coding agents (Codex reads this) |
| [CLAUDE.md](CLAUDE.md) | Short pointer and rules for Claude Code |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Ground rules, branch/PR/commit conventions |
| [CONTRACT_CHANGELOG.md](CONTRACT_CHANGELOG.md) | Log of interface contract changes |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Data flow, `HazardRecord`, API, GPU/CPU placement, topology |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Machines, Docker GPU rules, versions, disk sizes, timings |
| [docs/SOURCE_REPOS.md](docs/SOURCE_REPOS.md) | Pinned source repos, what to port and drop, known bugs |
| [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md) | Datasets, license status, what may ship where |
| [docs/EVALUATION.md](docs/EVALUATION.md) | The three evaluations, metrics, confounds, reporting |
| [docs/QUALITY_BAR.md](docs/QUALITY_BAR.md) | Quality bar, evidence standards, and the core-first gate |
| [docs/AGENT_WORKFLOW.md](docs/AGENT_WORKFLOW.md) | Lanes, sprints, handoffs, review, token rules |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Terms |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Known failures and causes |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Where the project stands, what is next, and what production-ready means |
| [docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md) | Open questions, assumptions, blockers |
| [docs/TRACEABILITY.md](docs/TRACEABILITY.md) | Fact-to-file matrix |
| [docs/PROMPT_HISTORY.md](docs/PROMPT_HISTORY.md) | How this commit was produced |
| [docs/adr/](docs/adr/0000-template.md) | Architecture decision records |
| [docs/agents/](docs/agents/00-scaffold.md) | Per-workstream agent briefs |

## Limitations

These are known now, before any code exists.

- **Domain gap.** The defect classifier trains on RDD2022 close-ups of road surface. Windshield frames are wide views where a cell of a 1600x900 frame in an 8x8 grid is about 200x112 px and mostly sky, buildings and vehicles; RDD2022 has no non-road negatives; fixed grid cells are the wrong size at different distances. The planned fix is road-masked overlapping tiling plus auto-labeled negatives, to be measured, not assumed. <!-- F:S-12 -->
- **Effective detection range.** A pothole 30 m away is a few pixels. Far-field defects will be missed. The effective detection range must be stated in this README once measured; it is unmeasured today. <!-- F:Q-05 -->
- **v1.0-mini confounds.** In nuScenes v1.0-mini, `is_night=True` and `location=singapore-hollandvillage` are perfectly correlated and the `is_rain` slice is empty. v1.0-mini is for pipeline validation, not robust conclusions. v1.0-full (700 scenes) resolves this. <!-- F:S-06 -->
- **Classifier over-firing.** The source classifier flags about half the patches at threshold 0.5 (31 of 64 on a sample frame). Expect many false positives until negatives, regularization and threshold tuning are in. The classifier false-positive rate on non-road tiles must be reported once measured. <!-- F:S-10 -->
- **Label-name check.** **[CRITICAL]** The source `defect-detector` maps D00 to pothole, D10 to longitudinal, D20 to transverse, D40 to alligator. The commonly documented RDD convention is D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole (UNVERIFIED against the official RDD2022 docs; not fetched). If the source is wrong, its per-class metric names and every "pothole" semantic are mislabeled. Workstream B must verify before porting and pin the mapping with a test. Nothing keyed on "pothole" is trustworthy until then. <!-- F:S-11 -->
- **Training machine unconfirmed.** **[CRITICAL]** The source READMEs describe Windows 11 with an NVIDIA RTX 5080, but it is UNCONFIRMED whether that machine is a separate desktop or whether the owner now works on the MacBook. This decides where training runs and whether Docker GPU passthrough is possible. It blocks Sprint 1. See [docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md). <!-- F:E-02 -->
- **RDD2022 license unverified.** **[CRITICAL]** The figshare record (CRDDC 2022) names no license. UNVERIFIED. No RDD2022 images will be published (hosted corpus, README images, committed fixtures) until it is confirmed. <!-- F:L-02 -->
- **No ground truth for fusion.** nuScenes has no road-defect labels and RDD2022 has no object labels. The synthetic paste test is the substitute and is easier than real defects; pasted-patch results are not real-world accuracy. <!-- F:R-07 -->
- No usable weights exist for either source model. Nothing has been run.

## Quickstart

Stub API only (no real models yet):

```
pip install -r requirements-dev.lock && pip install --no-deps -e .
uvicorn api.main:app --port 8000      # then open http://localhost:8000/docs
python -m pytest
```

All commands: [AGENTS.md](AGENTS.md) section 4.

## License

MIT (a recommendation recorded as the default, not a final owner decision). Third-party datasets have their own terms: see [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md).
