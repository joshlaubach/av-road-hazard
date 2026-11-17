# REPORT-A: git hygiene

## What changed
- The uncommitted working-tree edit to `.gitignore` (which deleted the `notebooks/*` rule and its comment, 4 lines) was reverted with `git restore .gitignore` in the main checkout. `main` itself already had the rule, so this branch has no `.gitignore` diff.
- `git check-ignore -v` now reports both notebooks as ignored:
  - `.gitignore:68:notebooks/*  notebooks/01_nuscenes_eda.ipynb`
  - `.gitignore:68:notebooks/*  notebooks/02_rdd2022_eda.ipynb`

## For the owner to decide (nothing deleted)

### refs/codex/turn-diffs/
- `refs/codex/turn-diffs/checkpoints/c1bb4be2577dfdce2459ab21d71d5293/f6914212fcfcbaf56ec82aab9bbc994c/1793671818433/57368458-7a08-4cfb-bddd-8a2e0f9ca2c4` -> `c808a7a`

This local-only ref keeps `notebooks/01_nuscenes_eda.ipynb` (5 MB, 8 embedded nuScenes images) reachable in history. It is not on GitHub (`git ls-remote origin` lists 0 codex refs). A mirror push would publish it.

### agent/* branches merged into origin/main (by `git branch --merged`)
Local (41):
- `agent/api-hardening`
- `agent/corpus-index`
- `agent/corpus-manifest`
- `agent/deploy-prep`
- `agent/deploy-space`
- `agent/docker-gpu`
- `agent/docker-ollama`
- `agent/docker-search`
- `agent/docker-verify`
- `agent/docs-claude-md-rdd2022`
- `agent/docs-close-docker`
- `agent/docs-corpus-decision`
- `agent/docs-license-and-corpus`
- `agent/docs-license-cleanup`
- `agent/docs-no-cached-records`
- `agent/docs-tick-docker-box`
- `agent/fetch-classifier`
- `agent/fix-claims-and-spend`
- `agent/fix-corpus-integrity`
- `agent/fix-parallel-jobs`
- `agent/fix-proxy-address`
- `agent/fix-quota-race`
- `agent/fix-retention`
- `agent/fix-retention-durable`
- `agent/fix-session-eviction`
- `agent/fix-space-node`
- `agent/health-ready`
- `agent/llm-log-level`
- `agent/pipeline-graphic`
- `agent/readme-final`
- `agent/record-index-run`
- `agent/release-bundle`
- `agent/review-cover-note`
- `agent/reviewer-instructions-v2`
- `agent/search-notice`
- `agent/smoke-check-notes`
- `agent/space-assembly`
- `agent/stellenbosch-corpus`
- `agent/stub-answers`
- `agent/stub-defect-label`
- `agent/web-e2e`

Remote (122):
- `origin/agent/adr-0011-accepted`
- `origin/agent/adr-0014-detector`
- `origin/agent/adr-0015-detector-finetune`
- `origin/agent/alert-review-result`
- `origin/agent/api-hardening`
- `origin/agent/api-real-pipeline`
- `origin/agent/ci-fixtures`
- `origin/agent/ci-real-fixtures`
- `origin/agent/construction-gap`
- `origin/agent/construction-labels`
- `origin/agent/core-report`
- `origin/agent/corpus-index`
- `origin/agent/corpus-manifest`
- `origin/agent/corpus-retrieval`
- `origin/agent/d10-decision`
- `origin/agent/defects`
- `origin/agent/defects-cleanup`
- `origin/agent/defects-country-eval`
- `origin/agent/defects-find-defects`
- `origin/agent/defects-groups`
- `origin/agent/defects-hard-negatives`
- `origin/agent/defects-label-names`
- `origin/agent/defects-labels`
- `origin/agent/defects-nuscenes-review`
- `origin/agent/defects-patches`
- `origin/agent/defects-provenance`
- `origin/agent/defects-road-filter`
- `origin/agent/defects-road-mask`
- `origin/agent/defects-ship-checkpoint`
- `origin/agent/defects-tile-cache`
- `origin/agent/defects-tile-index`
- `origin/agent/defects-tile-labels`
- `origin/agent/defects-tile-spacing`
- `origin/agent/defects-tile-training`
- `origin/agent/defects-train`
- `origin/agent/defects-validation-fix`
- `origin/agent/demo-corpus`
- `origin/agent/deploy-prep`
- `origin/agent/deploy-space`
- `origin/agent/detector-final`
- `origin/agent/detector-finetune`
- `origin/agent/detector-sanity-check`
- `origin/agent/detector-threshold-sweep`
- `origin/agent/docker-gpu`
- `origin/agent/docker-ollama`
- `origin/agent/docker-search`
- `origin/agent/docker-verify`
- `origin/agent/docs-claude-md-rdd2022`
- `origin/agent/docs-close-docker`
- `origin/agent/docs-drop-carla`
- `origin/agent/docs-frontend-explanations`
- `origin/agent/docs-license-and-corpus`
- `origin/agent/docs-license-cleanup`
- `origin/agent/docs-no-cached-records`
- `origin/agent/docs-nuscenes-terms`
- `origin/agent/docs-own-photos`
- `origin/agent/docs-rdd2022-licence`
- `origin/agent/docs-real-photo-check`
- `origin/agent/docs-roadmap`
- `origin/agent/docs-roadmap-rule`
- `origin/agent/eval-metrics`
- `origin/agent/eval-notebooks`
- `origin/agent/eval-per-tile`
- `origin/agent/eval-synthetic-fusion`
- `origin/agent/fetch-classifier`
- `origin/agent/find-defects-region`
- `origin/agent/finetune-results`
- `origin/agent/fix-claims-and-spend`
- `origin/agent/fix-corpus-integrity`
- `origin/agent/fix-parallel-jobs`
- `origin/agent/fix-proxy-address`
- `origin/agent/fix-quota-race`
- `origin/agent/fix-retention`
- `origin/agent/fix-retention-durable`
- `origin/agent/fix-session-eviction`
- `origin/agent/fix-space-node`
- `origin/agent/fusion`
- `origin/agent/fusion-severity`
- `origin/agent/health-ready`
- `origin/agent/integration-fuse`
- `origin/agent/license-approval`
- `origin/agent/noise-test`
- `origin/agent/perception-detect`
- `origin/agent/perception-detector`
- `origin/agent/perception-eval`
- `origin/agent/perception-labels`
- `origin/agent/pipeline-graphic`
- `origin/agent/pipeline-road-mask`
- `origin/agent/rag`
- `origin/agent/rag-answers`
- `origin/agent/rag-api`
- `origin/agent/rag-claude`
- `origin/agent/rag-eval`
- `origin/agent/rag-security`
- `origin/agent/readme-eli16`
- `origin/agent/readme-final`
- `origin/agent/record-index-run`
- `origin/agent/refine-docs-paths`
- `origin/agent/refine-eval-logic`
- `origin/agent/refine-load-checkpoint`
- `origin/agent/refine-python-layout`
- `origin/agent/refine-shared-helpers`
- `origin/agent/refine-test-layout`
- `origin/agent/release-bundle`
- `origin/agent/review-cover-note`
- `origin/agent/review-fixes`
- `origin/agent/reviewer-instructions-v2`
- `origin/agent/roadmap-pointer`
- `origin/agent/safety-net-light-imports`
- `origin/agent/scaffold`
- `origin/agent/search-notice`
- `origin/agent/smoke-check-notes`
- `origin/agent/space-assembly`
- `origin/agent/stellenbosch-corpus`
- `origin/agent/stub-answers`
- `origin/agent/stub-defect-label`
- `origin/agent/web-e2e`
- `origin/agent/web-font`
- `origin/agent/web-search`
- `origin/agent/web-settings`
- `origin/agent/web-ui`
- `origin/agent/weighted-search`

### agent/* branches git does NOT see as merged (may be squash-merged or abandoned; check before deleting)
- `agent/docs-license-decision`
- `agent/reviewer-instructions`
- `agent/score-table`
- `origin/agent/docs-corpus-decision`
- `origin/agent/docs-license-decision`
- `origin/agent/docs-tick-docker-box`
- `origin/agent/score-table`

## Skipped
Nothing deleted, no history rewritten (per ground rules).

## Tests
No code changed; tests not affected. Full suite earlier today on the same commit: 1200 passed, 4 skipped.
