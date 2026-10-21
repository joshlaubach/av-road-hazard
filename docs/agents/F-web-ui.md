Purpose: self-contained brief for the F Web UI workstream of av-road-hazard; read it plus `fusion/schema.py` (once it exists) and nothing else.

# Brief: F Web UI

## Lane
Codex

## Sprint and dependencies
Sprint 3 (days 7-9). Depends on the scaffold and the stub API with its generated TypeScript client. Gated by the proposed core-first gate: beyond stubs only after A, B, C and the synthetic fusion evaluation produce reportable numbers, or the owner overrides. React polish is first in the scope-cut order; fallback if the timeline slips is Streamlit on the same API.

## Goal
Build the Next.js + TypeScript UI against the stub API with a typed client generated from OpenAPI.

## Owned paths
- `web/`

Do not touch:
- `api/`, `rag/`, model packages
- `openapi.json` (contract-change PR only)

## Inputs
- `openapi.json` and the generated TypeScript client
- SPEC section 5; the Streamlit dashboard `dashboard/app.py` in `joshlaubach/defect-detector` @ `9da52cdf7b7c3bedf70f32b3750b2834f3acf433` is not ported in v1 and is only relevant to the fallback

No other exploration of the source repos is allowed.

## Outputs
- Next.js app: Analyze view (upload, overlay boxes with severity colors, record JSON panel), Search view (text + image query, filters, results grid, answer with cited frames), Settings (LLM backend toggle)

## Contract references
- `SPEC.md` section 5 (interfaces), section 4 (repo layout)
- API routes and `openapi.json` as the single contract

## Ordered task list
1. Scaffold Next.js + TypeScript in `web/`; use the typed client generated from `openapi.json`.
2. Build the Analyze view against the stub API: upload, boxes with severity colors, `HazardRecord` JSON panel.
3. Build the Search view: text + image query, filters, results grid, answer with cited frame IDs.
4. Build the Settings view: LLM backend toggle (`llama` or `claude`).
5. Make it responsive at phone width; Lighthouse and accessibility basics.
6. Verify no nuScenes images are bundled.

## Tests required
Layers from ENGINEERING.md section 3:
- TypeScript typecheck, lint, build (CI); component tests as needed; e2e against the stub API

## Done criteria
- Runs against the stub API
- Lighthouse/a11y basics
- Responsive at phone width
- No nuScenes images bundled

## Known pitfalls
- React/Next.js was chosen over Streamlit; the timeline tradeoff is accepted and mitigated by the OpenAPI-generated client.
- OpenAPI is the single contract; do not hand-write API types.
- No nuScenes images; RDD2022 images only after license verification.
- Charts and styling: avoid templated defaults.

## Debugging hints
- Reproduce first: capture exact input image, config, seed, git SHA into `tests/fixtures/regressions/`.
- Use `tools/overlay.py` and `DEBUG_DUMP_DIR` before theorizing; most bugs are coordinate bugs (tile to frame, width/height order, resize scale).

## Token-saving notes
- Start a fresh session; read only this brief and `fusion/schema.py`.
- Do not explore beyond the files named in Inputs.
- Small, test-first PRs (under about 400 changed lines).

## Skills to load
Owner to fill in; none enumerated yet. Placeholders suggested earlier: a frontend-design skill (non-templated UI styling) and a dataviz skill (charts).

## PR checklist
- [ ] Branch `agent/web-ui`; PR into `main`; no direct push
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
