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
