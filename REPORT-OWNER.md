# REPORT-OWNER: decisions and work left for the owner

These came out of the 2026-11-16 publish-readiness review. None of the fix/* agent branches touch them, because each one needs an owner decision, an owner-run step, or a history rewrite.

| # | Item | Why it is yours | Where |
|---|---|---|---|
| 1 | **Public or private.** GitHub reports the repo as PUBLIC; README line 242 says "(private)"; the GitHub description still says "Documentation only so far... Nothing built or tested." | Visibility decision | `README.md:242`, GitHub repo settings |
| 2 | **Test-split evaluation.** The headline classifier numbers are on fold-0 validation tiles (`tools/eval_classifier.py:119`), the same split that picked the checkpoint (`defects/train.py:450`). The 15% test split (4,745 images) has never been scored. Run it once, with checkpoint, thresholds and config frozen, and never tune on it afterwards. | Must be run once, by the owner | `docs/EVALUATION.md:18`, `eval/results.json` |
| 3 | **Re-run evals from `main`.** Every `git_commit` recorded in `eval/results.json` (e.g. line 4, 1185, 1210) is not in this repository's history; the retrieval run was `git_dirty: true`. Re-run from a commit on `main`, or record the mapping. | Changes `eval/results.json` | `eval/results.json` |
| 4 | **Demo photos that are training images.** 18 of the 40 RDD2022 demo photos are in fold-0 train (2 validation, 11 outer fold, 9 test). Replace with test-split photos or keep and disclose. | Which photos to publish | `data/demo_corpus/`, `configs/corpus_manifest.json` |
| 5 | **Faces and license plates.** Demo and test photos (RDD2022 demo corpus, 20 Stellenbosch photos, `tests/fixtures/stellenbosch/`) have not been checked for faces or plates. | Which photos to publish | `tests/fixtures/stellenbosch/`, `data/demo_corpus/` |
| 6 | **Email addresses in commit metadata.** A university address (17 commits) and a personal Gmail address (105 commits) are in the public history. Removing them needs a history rewrite and force-push. | History rewrite | git history |
| 7 | **Pin downloads.** OpenCLIP weights are not hash-checked (`docs/SECURITY_RAG.md:24`); the classifier is fetched at `"revision": "main"` (`configs/hosted_classifier.json:2`) and needs a commit hash. | Needs the owner's HF repo commit hash | `rag/embed.py`, `configs/hosted_classifier.json:2` |
| 8 | **Release review.** Every box in `docs/RELEASE_REVIEW.md` is unchecked; 6 of 8 "Production-ready" boxes in `docs/ROADMAP.md` are open. Needs a reviewer who has not seen the build. | Needs another person | `docs/RELEASE_REVIEW.md`, `docs/ROADMAP.md:93` |

Also waiting on owner input from the agent reports: the hero tagline and the 50-vs-51 hero score (REPORT-E), the local `refs/codex/turn-diffs/...` ref and stale `agent/*` branches (REPORT-A), the ADR 0015 vs OPEN_QUESTIONS D2 Motional disagreement (REPORT-B).

Process note: CLAUDE.md asks every PR to update `docs/ROADMAP.md` and use `agent/<workstream>` branch names. These branches are named `fix/<task>` and do not touch the roadmap, because the task brief restricted each branch to specific files. Add a roadmap row when merging.
