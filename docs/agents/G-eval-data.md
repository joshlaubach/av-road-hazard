Purpose: self-contained brief for the G Eval + Data workstream of av-road-hazard; read it plus `fusion/schema.py` (once it exists) and nothing else.

# Brief: G Eval + Data

## Lane
Claude

## Sprint and dependencies
Data scripts early (right after the scaffold, merge order: scaffold, then C and G data scripts); Sprint 3 (days 7-9) for the evaluations and the CARLA frame generator. Depends on A and B checkpoints for the real-model runs; evals first run on stubs.

## Goal
Provide data download scripts, the nuScenes loader, the CARLA frame generator, and the three automatic evaluations that write `eval/results.json` and a README table.

## Owned paths
- `eval/`
- `data/` (download scripts and manifests only; no data committed)

Do not touch:
- Model code in `perception/`, `defects/`
- `fusion/schema.py`

## Inputs
- `joshlaubach/nuscenes-stability` @ `9acd844bf61c7a3a02b7210d3fe0d9a284e212f6`: `src/data.py` (e.g. `compute_scene_split`), `src/metrics.py`, `src/utils_nuscenes.py` for metric reuse
- docs/EVALUATION.md for the protocol
- RDD2022 locally; nuScenes locally; CARLA 0.9.16 on the Windows host

No other exploration of the source repos is allowed.

## Outputs
- Download scripts (RDD2022; nuScenes instructions) with SHA-256 verification against `data/MANIFEST.json`
- nuScenes loader
- CARLA frame generator (host-side script)
- `eval/models.py`, `eval/retrieval.py`, `eval/fusion_synthetic.py`
- `eval/results.json` and a markdown table for the README

## Contract references
- `SPEC.md` section 5 (interfaces), section 4 (repo layout)
- `HazardRecord` for fusion checks

## Ordered task list
1. Write download scripts (RDD2022; nuScenes instructions) that verify hashes against `data/MANIFEST.json` and delete archives unless `--keep-archive`.
2. Write the nuScenes loader.
3. Write the CARLA frame generator as a host-side script; resolve the defect decals/textures decision with the owner (stock CARLA has no potholes or cracks).
4. Implement `eval/models.py`: detector P/R/F1, COCO AP, condition slices; classifier macro-F1 on RDD2022 val, non-road false-positive rate.
5. Implement `eval/retrieval.py`: queries auto-generated from nuScenes GT annotations (for example "frames with a pedestrian at night"), recall@k against annotation-derived answers.
6. Implement `eval/fusion_synthetic.py`: paste RDD2022 defect patches on road regions at known positions; verify detected box, `in_road`, `near_objects`; report detection rate versus distance.
7. Write a single `eval/results.json` and a README markdown table; state the limits of the synthetic test.
8. Fill `eval/baselines.json` with the reproduced baselines.

## Tests required
Layers from ENGINEERING.md section 3:
- Data validation (label ranges, box sanity, no scene leakage), metric regression, retrieval eval, synthetic fusion, e2e run on stubs and real models

## Done criteria
- All three evals run end-to-end on stubs and real models and write results
- `eval/results.json` plus README table generated

## Known pitfalls
- RDD2022 license unverified: do not publish pasted images or samples.
- Stock CARLA has no defects; without decals the CARLA corpus exercises only the object side.
- CARLA runs on the Windows host only, outside Docker; not on macOS.
- Pasted patches are easier than real defects: never present synthetic results as real-world accuracy.
- v1.0-mini confounds (night and location perfectly correlated; no rain).
- Split by scene, never by frame; touch the test split once per release.

## Debugging hints
- Reproduce first: capture exact input image, config, seed, git SHA into `tests/fixtures/regressions/`.
- Use `tools/overlay.py` and `DEBUG_DUMP_DIR` before theorizing; most bugs are coordinate bugs (tile to frame, width/height order, resize scale).

## Token-saving notes
- Start a fresh session; read only this brief and `fusion/schema.py`.
- Do not explore beyond the files named in Inputs.
- Small, test-first PRs (under about 400 changed lines).

## Skills to load
Owner to fill in; none enumerated yet. Suggested placeholder: a dataviz skill for charts.

## PR checklist
- [ ] Branch `agent/eval-data`; PR into `main`; no direct push
- [ ] Spec section implemented is named in the PR
- [ ] Tests added or updated and passing; ruff and mypy clean
- [ ] Config values in `configs/*.yaml` (no hardcoded paths, thresholds, seeds); seeds 42; `num_workers` from config
- [ ] No secrets, data, or weights committed; license boundaries respected
- [ ] Docs and changelog updated if behavior or contract changed (`contract-change` label if so)
- [ ] "How I verified this" note and handoff note in the PR description
- [ ] Maintainer explainer paragraph written

## Reviewer lane
Codex (the other tool), using the ENGINEERING.md section 2 checklist in one short pass.

## Maintainer explainer
Write this paragraph in your PR: what the component does, why, and what could be wrong. Maintainers must be able to explain it in an interview.
