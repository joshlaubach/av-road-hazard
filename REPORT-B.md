# REPORT-B: license docs (branch fix/license-docs, commit 8e7743c)

## What changed (docs/DATA_AND_LICENSES.md only)
1. Approval reworded to "not confirmed":
   - RDD2022 table row: "Owner's statement 2026-10-27: the authors approve non-commercial use" became "Author approval not confirmed; owner assumption (2026-11-03). No author replies are stored in the repo". Hosted column: "Allowed on the owner's statement" became "Used on the owner's assumption (not a verified licence)".
   - RDD2022 bullet: the 2026-10-27 "have approved" sentence now says approval is NOT confirmed and names the 2026-11-03 update as the accurate record. The 2026-11-03 text is unchanged.
   - Hosted corpus: "approved per the owner 2026-10-27" became "author approval not confirmed; owner assumption (2026-11-03)".
   - Summary table, RDD2022 row: reworded the same way.
2. Stellenbosch rows added to the dataset table and the summary table. License status: "CC0 1.0 per the Kaggle and Dataset Ninja listings. NOT verified at the authors' own page. Author approval not confirmed; owner assumption (2026-11-03)". Use, local, hosted and repo facts were checked against SOURCE.json, `git ls-files` (20 fixtures), configs/corpus_manifest.json (20 entries), docs/DEPLOY.md and web/public/examples/.

## Flag for the owner: Motional approval (no files changed)
- docs/adr/0015-detector-finetune.md:8 says "Motional confirmed that a portfolio project counts as non-commercial and that a public demo may serve a model fine-tuned on nuScenes"; line 59 says "the demo may serve them (Motional's reply)".
- docs/OPEN_QUESTIONS.md:81 (D2) and :36 say this is still open ("Email nuScenes@motional.com").
- DATA_AND_LICENSES.md (summary table and ~line 38) sides with OPEN_QUESTIONS ("Motional not yet asked").
- No Motional reply is stored anywhere in the repo. Status: UNKNOWN. The owner decides which is true.

## Skipped (outside the allowed file)
- tests/fixtures/stellenbosch/SOURCE.json:5 still says "the owner states the authors approved non-commercial use". It now disagrees with the doc.
- docs/OPEN_QUESTIONS.md item 3 (line 13) still says hosting stays blocked until the authors confirm. That conflicts with item 10 and with DATA_AND_LICENSES.
- DATA_AND_LICENSES.md line 42 still calls the corpus "open" (it was decided 2026-11-03), and the NOTICE (line 7) does not list Stellenbosch. Not in the brief.
- docs/ROADMAP.md not updated (the brief limited edits).

## Tests
pytest: 1200 passed, 4 skipped, 1 deselected (baseline). All 8 F:L markers intact.
