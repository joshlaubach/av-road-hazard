Purpose: self-contained brief for the B Defects workstream of av-road-hazard; read it plus `fusion/schema.py` (once it exists) and nothing else.

# Brief: B Defects

## Lane
Codex

## Sprint and dependencies
Sprint 1 (days 2-4), after A in the Codex lane. Depends on the scaffold. Needs RDD2022 downloaded locally. Training runs on the owner's machine in the background. Training machine is UNCONFIRMED (blocks Sprint 1). Feeds Phase 2 integration (critical path).

## Goal
Retrain the EfficientNet-B4 multi-label defect classifier on RDD2022, replace the fixed 8x8 grid with road-masked tiling, add auto-labeled negatives, and implement `find_defects(frame) -> list[Defect]`.

## Owned paths
- `defects/`
- `tests/defects/`

Do not touch:
- `perception/`, `fusion/`, `api/`
- `fusion/schema.py` (contract-change PR only)
- MVTec and the Streamlit dashboard are not ported in v1

## Inputs
- `joshlaubach/defect-detector` @ `9da52cdf7b7c3bedf70f32b3750b2834f3acf433`: `src/pipeline.py`, `src/utils/patches.py` (`patch_index_to_coords()`), `src/datasets/rdd2022.py`, `report.py`, `config.py`
- RDD2022 locally (`RDD2022_ROOT`); nuScenes frames for negatives
- Official RDD2022 documentation and annotation XML names (for the label check)

No other exploration of the source repos is allowed.

## Outputs
- `defects/train.py`, `defects/infer.py`, `defects/patches.py`, `defects/road_crop.py`
- `checkpoints/` classifier and its manifest entry
- Label-mapping pinning test
- Macro-F1 and non-road false-positive rate numbers

## Contract references
- `SPEC.md` section 5 (interfaces), section 4 (repo layout)
- `find_defects(frame) -> list[Defect]`, `Defect = {cls, name, confidence, box, patch_row, patch_col}`; multi-label sigmoid over 4 RDD2022 classes (D00, D10, D20, D40), threshold from config

## Ordered task list
1. FIRST verify the RDD2022 label names against official documentation and annotation XML names (S-11); add a test that pins the label mapping; fix `CLASS_NAMES` accordingly.
2. Download RDD2022 with a script that verifies hashes and deletes archives; record in `data/MANIFEST.json`.
3. Port the EfficientNet-B4 multi-label model and training loop (timm, input 380, sigmoid, threshold from config); use the deterministic 15% hold-out of train for validation because the test split has no annotations.
4. Address the 0.92 train / 0.73 val gap: augmentation, stronger regularization, early stopping on val.
5. Implement `road_crop.py`: drivable-road mask (the comma10k U-Net per ADR 0013; weights come from `tools/fetch_road_model.py`) and `patches.py`: overlapping tiles on the near-field road (bottom third) at native resolution, upscaled to 380x380, with tile-to-frame box mapping.
6. Add non-road tiles from nuScenes frames as automatically labeled negatives (labeled by the road mask, no hand labeling).
7. Implement `find_defects(frame)`; severity heuristic or optional LLM text (off by default).
8. Measure macro-F1 on RDD2022 val and the false-positive rate on non-road tiles before and after negatives; report both.
9. Commit message for the port includes source repo and SHA.

## Tests required
Layers from ENGINEERING.md section 3:
- Unit (tiling, tile-to-frame mapping, label mapping pin), ML smoke (overfit 8 images, shapes, checkpoint load), data validation (label ranges, no leakage), metric regression

## Done criteria
- Macro-F1 on RDD2022 val reported
- Negatives reduce false positives on non-road tiles (rate measured)
- Tiling tested on fixtures
- Label mapping verified against official docs and pinned by a test

## Known pitfalls
- **[CRITICAL]** Probable label-name bug: source documents D00 pothole, D10 longitudinal, D20 transverse, D40 alligator; the commonly documented convention is D00 longitudinal, D10 transverse, D20 alligator, D40 pothole (UNVERIFIED). Anything keyed on "pothole" depends on this.
- Over-firing: the source flagged 31 of 64 patches at threshold 0.5.
- Overfit gap 0.92 train vs 0.73 val.
- Domain gap: RDD2022 close-ups vs wide frames; scale mismatch, no negatives, perspective.
- RDD2022 license unverified: no RDD2022 images in committed fixtures or docs.
- Name collision: do not create a top-level `src/` package; source has `src/utils/`.
- Coordinate bugs: tile to frame, width/height order, resize scale.

## Debugging hints
- Reproduce first: capture exact input image, config, seed, git SHA into `tests/fixtures/regressions/`.
- Use `tools/overlay.py` and `DEBUG_DUMP_DIR` before theorizing; most bugs are coordinate bugs (tile to frame, width/height order, resize scale).
- Draw tiles, boxes, and the road mask with `tools/overlay.py` on a failing frame before changing thresholds.

## Token-saving notes
- Start a fresh session; read only this brief and `fusion/schema.py`.
- Do not explore beyond the files named in Inputs.
- Small, test-first PRs (under about 400 changed lines).

## Skills to load
Owner to fill in; none enumerated yet. Suggested placeholder: ML/PyTorch.

## PR checklist
- [ ] Branch `agent/defects`; PR into `main`; no direct push
- [ ] Spec section implemented is named in the PR
- [ ] Tests added or updated and passing; ruff and mypy clean
- [ ] Config values in `configs/*.yaml` (no hardcoded paths, thresholds, seeds); seeds 42; `num_workers` from config
- [ ] No secrets, data, or weights committed; license boundaries respected
- [ ] Docs and changelog updated if behavior or contract changed (`contract-change` label if so)
- [ ] "How I verified this" note and handoff note in the PR description
- [ ] Maintainer explainer paragraph written

## Reviewer lane
Claude (the other tool), using the ENGINEERING.md section 2 checklist in one short pass.

## Maintainer explainer
Write this paragraph in your PR: what the component does, why, and what could be wrong. Maintainers must be able to explain it in an interview.
