Purpose: self-contained brief for the I Deploy and polish (Phase 3) workstream of av-road-hazard; read it plus `fusion/schema.py` (once it exists) and nothing else.

# Brief: I Deploy and polish (Phase 3)

## Lane
Claude

## Sprint and dependencies
Sprint 5 (days 11-14). Depends on integration (H). Gated by the proposed core-first gate: beyond stubs only after A, B, C and the synthetic fusion evaluation produce reportable numbers, or the owner overrides.

## Goal
Deploy the CPU demo to a Hugging Face Space with Qdrant Cloud, finish the README with measured numbers and limitations, and record the walkthrough.

## Owned paths
- `docker/` CPU variant, hosting config, `README.md` results sections, walkthrough assets

Do not touch:
- Model and contract code

## Inputs
- `BUILD_PLAN.md` section 5
- `eval/results.json`
- docs/DATA_AND_LICENSES.md

No other exploration of the source repos is allowed.

## Outputs
- Live HF Space (CPU Docker)
- Qdrant Cloud collection
- README with architecture diagram, metrics table, limitations, run instructions
- 2-3 minute walkthrough video

## Contract references
- `SPEC.md` section 5 (interfaces), section 4 (repo layout)
- Hosted behavior of `/analyze`, `/query`, `/health`

## Ordered task list
1. Verify the RDD2022 license (owner) before including any RDD2022 sample image in the hosted corpus.
2. Build the HF Space (CPU Docker); Claude key as a Space secret; Qdrant Cloud free tier.
3. Hosted corpus = RDD2022 samples (after license verification), optionally the owner's own photos; precompute and cache outputs for the sample set so the demo stays fast on CPU.
4. Per-IP rate limit, upload limits, EXIF GPS stripping, Claude monthly spend cap with hard cutoff and fallback message.
5. README: architecture diagram, metrics table from `eval/results.json`, limitations (effective detection range, v1.0-mini confounds, domain gap), run instructions.
6. Record a 2-3 minute walkthrough video; on tag, deploy and smoke-test the deployed `/health`; keep the previous image tag for rollback.

## Tests required
Layers from ENGINEERING.md section 3:
- E2E against the deployed `/health`; smoke tests; LLM live check with cost cap

## Done criteria
- Demo runs on CPU and stays fast for the sample set
- No nuScenes images in the hosted bundle or corpus
- README claims all backed by `eval/results.json` or marked unmeasured
- Spend cap and fallback verified

## Known pitfalls
- Hosted compute is CPU only; EfficientNet-B4 over many tiles plus Faster R-CNN is slow.
- RDD2022 license unverified: do not publish its images until confirmed.
- nuScenes-derived records hosted only as text/metadata/charts if the owner confirms (default: do not show).
- Claude API is billed separately from the subscription; budget up to about $20/month.
- Rejected hosts: Fly.io/Render (use most of the budget), GPU instances ($50+/month).

## Debugging hints
- Reproduce first: capture exact input image, config, seed, git SHA into `tests/fixtures/regressions/`.
- Use `tools/overlay.py` and `DEBUG_DUMP_DIR` before theorizing; most bugs are coordinate bugs (tile to frame, width/height order, resize scale).

## Token-saving notes
- Start a fresh session; read only this brief and `fusion/schema.py`.
- Do not explore beyond the files named in Inputs.
- Small, test-first PRs (under about 400 changed lines).

## Skills to load
Owner to fill in; none enumerated yet.

## PR checklist
- [ ] Branch `agent/deploy`; PR into `main`; no direct push
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
