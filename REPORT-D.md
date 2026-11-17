# REPORT-D: code cleanups (branch fix/code-cleanups, commit d4cb702)

## What changed
1. api/main.py startup comment: the startup purge is best-effort cleanup, and expiry is enforced by
   expires_at (add_upload sets it; build_filter and get check it; RagService.touch sweeps with
   delete_expired). The multi-instance warning was kept because it is still true:
   delete_visitor_points() deletes all non-corpus points, live ones included, so a second instance
   restarting on a shared index would wipe the first one's live visitors. delete_expired() would
   not. The call was not changed (owner's decision).
2. web/e2e/lighthouse.mjs: removed the unused MIN_SCORE. lighthouserc.json enforces minScore 0.9
   through lhci assert.
3. defects/road_crop.py:145: torch.from_numpy(frame.copy()). Checked that a read-only array warns
   and its copy does not. Existing tests never pass a read-only frame.
4. README.md line 9: status date 2026-11-03 changed to 2026-11-05.
5. pyproject.toml: pytest-cov added to the dev extras.
6. ci.yml python job: new `pip install pytest-cov` step; pytest now runs once as
   `python -m pytest --cov --cov-report=term`; no threshold; contract-test comment kept.

## Skipped / follow-ups
- Regenerate requirements-dev.lock to include pytest-cov, then drop the extra CI install step.
- Recommended test (tests/ was not in the allowed files): a read-only frame through road_crop
  should raise no "not writable" warning.
- No [tool.coverage] config was added; bare --cov also measures tests/.
- Whether startup should call delete_expired() instead is an owner decision.

## Results
ruff check and ruff format --check: clean; mypy: clean.
pytest --cov: 1200 passed, 4 skipped, 88% total coverage (pytest-cov in a throwaway --target
folder; nothing installed into the .venv).
web: npm ci OK, lint clean, 88 tests passed.
