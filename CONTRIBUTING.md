Purpose: contribution rules for humans and agents on av-road-hazard: ground rules, branch/PR/commit conventions, Definition of Done, contract-change labeling.

# Contributing

Applies to every contributor, human or agent. See also [AGENTS.md](AGENTS.md), [BUILD_PLAN.md](BUILD_PLAN.md), [ENGINEERING.md](ENGINEERING.md).

## Ground rules (every agent)

Source: BUILD_PLAN.md section 0.

- One branch per agent: `agent/<workstream>`. Open a PR into `main`; never push to `main` directly.
- Own only your listed paths. Anything outside them goes through the interface contract in SPEC.md section 5 or a PR comment.
- Interfaces are frozen after Phase 0. Changes to `fusion/schema.py` or the API OpenAPI spec need a PR labeled `contract-change` and a note in `CONTRACT_CHANGELOG.md`.
- No data, weights, or secrets in git. Weights go in `checkpoints/` (gitignored); secrets via `.env` (`.env.example` committed).
- Every PR follows the Definition of Done in ENGINEERING.md section 10: tests pass, `ruff` and `mypy` clean, config values in `configs/*.yaml` (no hardcoded paths, thresholds, or seeds), and a short "how I verified this" note.
- Seeds fixed (42); `num_workers` read from config.
- Commit trailers per session attribution settings.

## Branches, PRs, commits

1. Branch per agent or feature: `agent/<workstream>`. Never push to `main`; PRs only. `main` is always deployable. Tag releases (`v0.1.0`). <!-- F:A-07 -->
2. Keep PRs small (aim under about 400 changed lines) and test-first.
3. Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`.
4. Every PR states which spec section it implements.
5. A PR needs CI green and one review, from an agent or person that did not write it, checked against contract adherence, tests present, no hardcoded paths or secrets, and license boundaries. The other tool reviews (Claude PRs by Codex, Codex PRs by Claude). <!-- F:A-03 -->
6. Branch protection (PRs required, CI required once CI exists) must be enabled before any agent starts, after the remote exists. <!-- F:A-07 -->
7. Commit trailers follow each tool's attribution requirement: the Claude session adds `Co-Authored-By` and a session link; Codex has its own. <!-- F:A-11 -->
8. Copying code from a source repo happens in separate commits, one per source repo, with the source repo and commit SHA in each message. See [ADR 0010](docs/adr/0010-source-ports-not-forks.md).

## Contract-change PRs

Contracts are defined in [CONTRACT_CHANGELOG.md](CONTRACT_CHANGELOG.md). A change to any contract:

1. Label the PR `contract-change`.
2. Add an entry to `CONTRACT_CHANGELOG.md` in the same PR.
3. Get approval from a reviewer in the other lane.
4. Regenerate `openapi.json` and the TypeScript client if the API changed. <!-- F:A-08 -->

## Definition of Done (per PR)

Source: ENGINEERING.md section 10.

Tests added or updated and passing; lint/type checks clean; config not code for tunables; docs and changelog updated if behavior or contract changed; `docs/ROADMAP.md` updated for the task (move it from Now or Next to Done, record the result, and set the next Now); no secrets, data, or weights committed; reviewer from outside the author checked it.
