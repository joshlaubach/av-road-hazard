Purpose: self-contained brief for the A Perception workstream of av-road-hazard; read it plus `fusion/schema.py` (once it exists) and nothing else.

# Brief: A Perception

> Update 2026-10-25: the detector was built differently from the plan below: zero-shot torchvision COCO models with pinned weights, no fine-tuning (see [ADR 0014](../adr/0014-object-detector.md)). The tasks below that port and fine-tune the notebook model are on hold until a measured gap and licence approval.

## Lane
Codex

## Sprint and dependencies
Sprint 1 (days 2-4), first in the lane, then B. Depends on the scaffold (schema, stubs, config). Needs nuScenes v1.0-mini locally (`NUSCENES_ROOT`). Training runs on the owner's machine in the background. Feeds Phase 2 integration (critical path).

## Goal
Port the Faster R-CNN detector from the nuscenes-stability notebook 02 into modules and implement `detect(frame) -> list[Object]`. Reproduce the original v1.0-mini metrics as a regression check.

## Owned paths
- `perception/`
- `tests/perception/`

Do not touch:
- `fusion/`, `defects/`, `api/` (go through the contract)
- `fusion/schema.py` (contract-change PR only)

## Inputs
- `joshlaubach/nuscenes-stability` @ `9acd844bf61c7a3a02b7210d3fe0d9a284e212f6`: notebook `02_detector_transfer_learning`; `src/data.py`, `src/metrics.py`, `src/utils_nuscenes.py`, `src/utils.py` (logic as modules, not notebooks)
- See docs/SOURCE_REPOS.md for the code map and config values
- nuScenes v1.0-mini local (`NUSCENES_ROOT`)

No other exploration of the source repos is allowed.

## Outputs
- `perception/train.py`, `perception/infer.py`
- `checkpoints/detector.pt` (gitignored) and its entry in `checkpoints/MANIFEST.json`
- Regression numbers recorded in `eval/baselines.json`

## Contract references
- `SPEC.md` section 5 (interfaces), section 4 (repo layout)
- `detect(frame: np.ndarray[RGB uint8]) -> list[Object]`, `Object = {label, score, box}`

## Ordered task list
1. Port the model (Faster R-CNN ResNet50 FPN v2, torchvision, torchmetrics AP) from notebook 02 into `train.py` and `infer.py`. The detector is built inside the notebook; there is no inference module to copy.
2. Reuse `src/data.py`, `src/metrics.py`, `src/utils_nuscenes.py` logic as modules: 2D box cache, deterministic 80/20 scene split, metrics.
3. Replace hardcoded `num_workers=0` and paths with config; seeds 42.
4. Build the 2D box cache from `NUSCENES_ROOT`; train; save to `checkpoints/detector.pt`.
5. Implement `detect(frame) -> list[Object]`: batch and device-aware, CPU path for the hosted demo, `USE_STUBS` aware.
6. Reproduce original v1.0-mini metrics (F1 about 0.534, AP@0.50 about 0.312) as a regression check; record the tolerance.
7. Make the dataset version (mini to full) a single config key.
8. Commit message for the port includes source repo and SHA.

## Tests required
Layers from ENGINEERING.md section 3:
- Unit (box projection, split, metrics), ML smoke (overfit 8 images, forward-pass shapes, checkpoint load), data validation (label ranges, box sanity, no scene leakage), metric regression

## Done criteria
- Checkpoint trains reproducibly
- Metrics within tolerance of original
- `detect()` unit-tested on a fixture image
- Full-dataset switch is one config key

## Known pitfalls
- `ROIAlign` is non-deterministic: results reproduce only on the same hardware and library versions; note it in the README.
- `num_workers=0` is hardcoded in the source for Windows; read from config.
- In v1.0-mini night and Singapore location are perfectly correlated and there is no rain; do not draw robust conclusions.
- Name collision: do not create a top-level `src/` package.
- Path to the cache: source writes `outputs/cache/boxes2d_v1.0-mini_CAM_FRONT.pkl`; paths must come from config.
- No nuScenes images in fixtures.

## Debugging hints
- Reproduce first: capture exact input image, config, seed, git SHA into `tests/fixtures/regressions/`.
- Use `tools/overlay.py` and `DEBUG_DUMP_DIR` before theorizing; most bugs are coordinate bugs (tile to frame, width/height order, resize scale).
- Suspect data before model: label ranges, class balance, leakage; compare with the last good run.

## Token-saving notes
- Start a fresh session; read only this brief and `fusion/schema.py`.
- Do not explore beyond the files named in Inputs.
- Small, test-first PRs (under about 400 changed lines).

## Skills to load
Owner to fill in; none enumerated yet. Suggested placeholder: ML/PyTorch, notebook-to-module refactor.

## PR checklist
- [ ] Branch `agent/perception`; PR into `main`; no direct push
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
