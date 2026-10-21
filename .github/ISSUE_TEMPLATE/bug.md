---
name: Bug
about: Reproduce-first bug report
labels: bug
---
<!-- Purpose: bug template following the reproduce-first playbook (ENGINEERING.md section 4). -->
## Reproduce
- Exact input image (add to `tests/fixtures/regressions/`):
- Config:
- Seed:
- Git SHA:

## Localize by stage
- `request_id`:
- `DEBUG_DUMP_DIR` intermediates (road mask, tiles, tile boxes, detections, fused record):
- Overlay from `tools/overlay.py` (tiles, boxes, masks on the source frame):

## Expected vs actual

## Failing test
<!-- A bug fix starts with a failing test. -->

## Notes
- ML-specific: suspect data before model (label ranges, class balance, leakage), then thresholds, then training config; compare against the last good run.
