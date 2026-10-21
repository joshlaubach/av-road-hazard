Purpose: versioned log of changes to av-road-hazard interface contracts.

# Contract changelog

## What counts as a contract

- `fusion/schema.py` (pydantic models).
- `openapi.json` (the API schema; the single contract between the Python API and the TypeScript client).
- The `Object`, `Defect`, and `HazardRecord` shapes (SPEC.md section 5).
- Config keys consumed across packages.

Contracts are frozen after Phase 0. Any change needs a PR labeled `contract-change`, an entry below, and approval from a reviewer in the other lane. <!-- F:A-08 -->

## Entry format

```
## vX.Y.Z: YYYY-MM-DD
- What changed (field, route, or key).
- Why.
- Compatibility: breaking or additive; who must update.
- PR: link.
```

## Entries

### v0.0.0
contracts defined in SPEC.md section 5, no code yet
