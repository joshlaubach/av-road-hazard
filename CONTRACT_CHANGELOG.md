Purpose: versioned log of changes to av-road-hazard interface contracts.

# Contract changelog

## What counts as a contract

- `fusion/schema.py` (pydantic models).
- `openapi.json` (the API schema; the single contract between the Python API and the TypeScript client).
- The `Object`, `Defect`, and `HazardRecord` shapes (SPEC.md section 5).
- Config keys consumed across packages.

Contracts are frozen after Phase 0. Any change needs a PR labeled `contract-change`, an entry below, and approval from a reviewer in the other lane. <!-- F:A-08 -->

## Entry format

```
## vX.Y.Z: YYYY-MM-DD
- What changed (field, route, or key).
- Why.
- Compatibility: breaking or additive; who must update.
- PR: link.
```

## Entries

### v0.4.0: 2026-10-27
- `openapi.json`: new route `POST /analyze/trace`, which takes the same upload as `POST /analyze` and returns `{record, trace}`. `record` is exactly what `/analyze` returns for the same image. `trace` holds the classifier's tile grid (each tile's covers, scored or skipped and why, per-class probabilities and flagged classes), the thresholds, a small base64 PNG of the road mask, and one fusion step per defect (the factors that multiply into its score and how many tiles were merged into it). New schemas: `TracedAnalysis`, `Trace`, `TileTrace`, `RoadMaskPreview`, `FusionStep`.
- Why: the web UI's walkthrough and heatmap must show what the pipeline did to the user's own image, which the record alone does not say.
- Compatibility: additive. `HazardRecord` and every existing route are unchanged, so existing callers and the RAG index are unaffected. Only the TypeScript client needs regenerating.
- PR: web UI, labeled `contract-change`.

### v0.3.0: 2026-10-26
- `fusion/schema.py`: the `source` values lose `carla`; they are now `rdd2022`, `upload` and `nuscenes`. `openapi.json` regenerated; the `source` enum is the only difference.
- Why: nothing in the project produces frames from that source.
- Compatibility: breaking for any writer that sends `source: "carla"`, which now fails validation. No code in the repo did; the only use was one test, now changed to `nuscenes`.
- PR: #49, labeled `contract-change`.

### v0.2.0: 2026-10-21 (workstream C, fusion)
- `fusion/schema.py`: `DefectRecord.in_road` changes from `bool` to `bool | None`. `True` means on the road, `False` means confirmed off the road, and `null` means unknown (for example no usable road mask). `openapi.json` regenerated; that field is the only schema difference.
- Why: missing road information must not silently become `False`, because fusion would then score the defect 0 and hide a possibly urgent hazard. Fusion scores unknown like on-road and keeps the uncertainty visible.
- Compatibility: this is a behavior change for readers, not merely additive. Writers that always emit `true` (the current stub pipeline) remain valid. Any reader that tests truthiness (`if defect.in_road`) treats `null` as false and silently drops unknown-road hazards. Consumers must review: use `in_road is not False` to keep unknown hazards, and `in_road is True` only to filter for confirmed on-road defects. No consumer exists in the repo yet (`web/` is not built); `tools/overlay.py` and `api/` do not read `in_road`. `SPEC.md` is a verbatim appendix and is not edited; its example (`"in_road": true`) stays valid.
- PR: workstream C (fusion), labeled `contract-change`; cross-lane review by Codex.

### v0.1.0: 2026-10-21 (Sprint 0)
- `fusion/schema.py` implements SPEC section 5. Conventions pinned, which SPEC left implicit:
  - `image_size` is `(width, height)`.
  - Boxes are named `box` everywhere (SPEC's interface text said `box_xyxy`); pixel `(x1, y1, x2, y2)` with `x1<=x2`, `y1<=y2`.
  - Two defect shapes: `Defect` (raw detector output: `cls`, `name`, `confidence`, `box`, `patch_row`, `patch_col`) and `DefectRecord` (inside a `HazardRecord`: `name`, `confidence`, `box`, `severity`, `in_road`, `near_objects`).
  - Scores, confidences, IoU are in [0, 1]; `hazard_score >= 0` (upper bound left to workstream C's formula); `cls >= 1`.
  - All models reject unknown fields.
- `openapi.json` v0.1.0: routes `/analyze`, `/index`, `/query`, `/records/{record_id}`, `/health`. `POST /query` takes `{text?, image_b64?, filters?, top_k, llm}`; `POST /index` takes multipart `files`.
- Compatibility: additive over v0.0.0 (no prior code).
- PR: Sprint 0 scaffold.

### v0.0.0
contracts defined in SPEC.md section 5, no code yet
