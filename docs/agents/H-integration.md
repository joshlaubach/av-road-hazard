Purpose: self-contained brief for the H Integration (Phase 2) workstream of av-road-hazard; read it plus `fusion/schema.py` (once it exists) and nothing else.

# Brief: H Integration (Phase 2)

## Lane
Claude

## Sprint and dependencies
Sprint 4 (days 9-11). Depends on A and B checkpoints (critical path), C, D, E. One agent, with a reviewer.

## Goal
Swap stubs for models, make the real pipeline work end to end, run the evaluations, and finish with an independent review.

## Owned paths
- Cross-package glue and fixes; PRs touching other workstreams' paths need an explicit note

Do not touch:
- Contract changes (go through `contract-change`)

## Inputs
- `BUILD_PLAN.md` section 4
- All merged workstream code and `eval/`
- 20 fixture images (license-clean)

No other exploration of the source repos is allowed.

## Outputs
- Integration PR
- Recorded baselines in `eval/baselines.json`
- Review notes from a separate agent

## Contract references
- `SPEC.md` section 5 (interfaces), section 4 (repo layout)
- All contracts; contract changes need a `contract-change` PR

## Ordered task list
1. Set `USE_STUBS=0`; run `/analyze` on 20 fixture images; fix schema or box-coordinate mismatches (most likely: tile-to-frame coordinates, image size ordering).
2. Index nuScenes v1.0-mini locally; run the retrieval eval.
3. Run the synthetic fusion test; tune thresholds in config only.
4. Review pass by a separate agent that has not seen the build: contract adherence, secrets, hardcoded paths, license boundaries (no nuScenes images in the hosted bundle).
5. Record baselines; run full evals on the owner's machine.

## Tests required
Layers from ENGINEERING.md section 3:
- Integration (real pipeline on fixtures), e2e, metric regression, retrieval eval, synthetic fusion

## Done criteria
- `/analyze` works with `USE_STUBS=0` on 20 fixtures
- Retrieval eval and synthetic fusion run with real models
- Independent review completed and findings fixed

## Known pitfalls
- Most likely bugs: tile-to-frame coordinates and image size ordering (width/height).
- Tune thresholds in config only, never in code.
- Do not weaken tests to pass.
- License boundaries: no nuScenes images in any hosted bundle.

## Debugging hints
- Reproduce first: capture exact input image, config, seed, git SHA into `tests/fixtures/regressions/`.
- Use `tools/overlay.py` and `DEBUG_DUMP_DIR` before theorizing; most bugs are coordinate bugs (tile to frame, width/height order, resize scale).

## Token-saving notes
- Start a fresh session; read only this brief and `fusion/schema.py`.
- Do not explore beyond the files named in Inputs.
- Small, test-first PRs (under about 400 changed lines).
- The independent reviewer must be a fresh session that has not seen the build.

## Skills to load
Owner to fill in; none enumerated yet.

## PR checklist
- [ ] Branch `agent/integration`; PR into `main`; no direct push
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
