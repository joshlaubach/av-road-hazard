Purpose: self-contained brief for the C Fusion workstream of av-road-hazard; read it plus `fusion/schema.py` (once it exists) and nothing else.

# Brief: C Fusion

## Lane
Claude

## Sprint and dependencies
Sprint 1 (days 2-4), stub-driven. Depends only on the scaffold (schema and stubs).

## Goal
Implement fusion of objects and defects into a `HazardRecord`: road mask intersection (`in_road`), nearest-object computation (`near_objects`), and the deterministic `hazard_score`.

## Owned paths
- `fusion/` (except `schema.py`, which is a contract)
- `tests/fusion/`

Do not touch:
- Model code in `perception/`, `defects/`
- `fusion/schema.py` (contract-change PR only)

## Inputs
- `fusion/schema.py`, the stubs (`USE_STUBS=1`), SPEC section 5
- `patch_index_to_coords()` in defect-detector `src/utils/patches.py` is the basis for patch-to-pixel mapping: `joshlaubach/defect-detector` @ `9da52cdf7b7c3bedf70f32b3750b2834f3acf433` (optional; tile mapping is owned by B)

No other exploration of the source repos is allowed.

## Outputs
- `fusion/fuse.py`
- `fusion/README.md` documenting the `hazard_score` formula
- Tests in `tests/fusion/`

## Contract references
- `SPEC.md` section 5 (interfaces), section 4 (repo layout)
- `HazardRecord` JSON and `hazard_score` (documented deterministic formula: defect severity x confidence x proximity to vulnerable objects; not learned)

## Ordered task list
1. Define the `hazard_score` formula (severity x confidence x proximity to vulnerable objects) and document it in `fusion/README.md`; record it as a decision (ADR) if it is non-obvious.
2. Implement road-mask intersection to set `in_road`.
3. Implement `near_objects` with IoU and pixel distance (`dist_px`).
4. Implement `fuse(objects, defects, road_mask, meta) -> HazardRecord` as pure functions with no model imports.
5. Handle edge cases: no objects, no defects, overlapping boxes, boxes at the frame edge.
6. Property-style tests with hand-built boxes; fill every schema field.

## Tests required
Layers from ENGINEERING.md section 3:
- Unit (pure functions, property-style tests with hand-built boxes), contract (schema round trip)

## Done criteria
- 100% of schema fields populated
- Edge cases covered: no objects, no defects, overlapping boxes, boxes at frame edge
- `hazard_score` formula documented in `fusion/README.md`

## Known pitfalls
- Fusion has no ground truth: nuScenes has no road-defect labels and RDD2022 has no object labels; correctness is checked by the synthetic paste test (workstream G).
- `image_size` ordering and box coordinate order (width/height, x/y) are likely bug sources.
- Anything keyed on "pothole" depends on the unverified RDD2022 label mapping (S-11).
- `hazard_score` is deterministic, not learned; do not tune it on the synthetic test and report it as accuracy.

## Debugging hints
- Reproduce first: capture exact input image, config, seed, git SHA into `tests/fixtures/regressions/`.
- Use `tools/overlay.py` and `DEBUG_DUMP_DIR` before theorizing; most bugs are coordinate bugs (tile to frame, width/height order, resize scale).

## Token-saving notes
- Start a fresh session; read only this brief and `fusion/schema.py`.
- Do not explore beyond the files named in Inputs.
- Small, test-first PRs (under about 400 changed lines).
- Pure functions with no model imports keep tests cheap.

## Skills to load
Owner to fill in; none enumerated yet.

## PR checklist
- [ ] Branch `agent/fusion`; PR into `main`; no direct push
- [ ] Spec section implemented is named in the PR
- [ ] Tests added or updated and passing; ruff and mypy clean
- [ ] Config values in `configs/*.yaml` (no hardcoded paths, thresholds, seeds); seeds 42; `num_workers` from config
- [ ] No secrets, data, or weights committed; license boundaries respected
- [ ] Docs and changelog updated if behavior or contract changed (`contract-change` label if so)
- [ ] "How I verified this" note and handoff note in the PR description
- [ ] Owner explainer paragraph written

## Reviewer lane
Codex (the other tool), using the ENGINEERING.md section 2 checklist in one short pass.

## Owner explainer
Write this paragraph in your PR: what the component does, why, and what could be wrong. The owner must be able to explain it in an interview.
