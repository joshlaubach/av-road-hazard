# REPORT-OWNER: decisions and work left for the owner

These came out of the 2026-11-16 publish-readiness review. Updated 2026-11-17 after PRs #142 to #150 merged. Each open item needs an owner decision, an owner-run step, another person, or a history rewrite.

## Still open

| # | Item | Why it is yours | Where |
|---|---|---|---|
| 1 | **Re-run the older evals from `main`.** 7 of the 8 `git_commit` values in `eval/results.json` are not in this repository's history (6c4600e, 7878e0c, 816dc3a, 93ecf93, b0055fd, b2a4186, dc5046a); only the sealed test-split run (0ef128a) is. The retrieval run was also `git_dirty: true`. Re-run from a commit on `main`, or record what each old commit corresponds to. | Changes `eval/results.json` | `eval/results.json` |
| 2 | **Demo photos that are training images.** 18 of the 40 RDD2022 demo photos are in fold-0 train (2 validation, 11 outer fold, 9 test). The README now discloses this. Replace them with test-split photos, or keep the disclosure. | Which photos to publish | `data/demo_corpus/`, `configs/corpus_manifest.json` |
| 3 | **Faces and license plates.** Demo and test photos (RDD2022 demo corpus, 20 Stellenbosch photos, `tests/fixtures/stellenbosch/`) have not been checked for faces or plates. | Which photos to publish | `tests/fixtures/stellenbosch/`, `data/demo_corpus/` |
| 4 | **Email addresses in commit metadata.** A university address (17 commits) and a personal Gmail address (105 commits) are in the public history. Removing them needs a history rewrite and force-push. | History rewrite | git history |
| 5 | **Pin downloads.** OpenCLIP weights are not hash-checked (`docs/SECURITY_RAG.md:24`); the classifier is fetched at `"revision": "main"` (`configs/hosted_classifier.json:2`) and needs a commit hash. | Needs the owner's HF repo commit hash | `rag/embed.py`, `configs/hosted_classifier.json:2` |
| 6 | **Release review.** Every box in `docs/RELEASE_REVIEW.md` is unchecked; 6 of 8 "Production-ready" boxes in `docs/ROADMAP.md` are open. Needs a reviewer who has not seen the build. | Needs another person | `docs/RELEASE_REVIEW.md`, `docs/ROADMAP.md` |
| 7 | **Hero image wording.** The hero tagline and the 50-vs-51 hero score, and the "nobody nearby" wording (REPORT-E). | Wording decision | `web/` hero section |
| 8 | **Motional disagreement.** ADR 0015 and OPEN_QUESTIONS D2 disagree on Motional approval, and `SOURCE.json` still says "approved" (REPORT-B). | License fact only the owner knows | `docs/adr/0015-*`, `docs/OPEN_QUESTIONS.md`, `SOURCE.json` |
| 9 | **`splits.json` and the pre-commit hooks.** `splits.json` (7.9 MB, IDs only) is not committed because of the 1 MB large-file hook; options are a hook exclusion, a gzipped copy with the raw SHA-256 recorded, or Git LFS. Separately, `end-of-file-fixer` and `trailing-whitespace` would rewrite the `eval/ids/*.json` files and break their hashes under `pre-commit run --all-files`; an `exclude: ^eval/ids/.*\.json$` on both hooks fixes that (REPORT-F). | Repo policy choice | `.pre-commit-config.yaml`, `eval/ids/` |

## Small follow-ups (no decision needed, not done yet)

- `requirements-dev.lock` still pins `opentelemetry-api==1.45.1` (the version with the known CVE); `requirements.lock` is on 1.45.2. It also lacks `pytest-cov`, which CI now installs separately.
- vitest / tinypool major-version bump in `web/`.
- The `WITH_ML=1` Docker image has not been rebuilt since #150 put it back on `python:3.12-slim`; CI builds only the stub image. Check with `docker compose -f docker/docker-compose.yml -f docker/compose.real.yml build api`.

## Done since the review

- **Public or private:** README says "(public)" (#143 to #147) and the GitHub description now describes the project.
- **Sealed test-split evaluation:** run once on 2026-11-17 at 0ef128a (#149); results in `eval/test_split_result.json`, `eval/results.json` and the README.
- **Stale refs and branches:** the local `refs/codex/...` ref is deleted and 122 merged remote `agent/*` branches are pruned (restore list kept locally in `handoff/`).
- **Python 3.15 Docker image (#142):** reverted in #150; Dependabot now ignores major and minor bumps of the `python` image.

Process note: CLAUDE.md asks for `agent/<workstream>` branch names. This branch is named `fix/owner-report` because the review task used `fix/<task>` names.
