Purpose: ADR 0010: Port source code into new packages, not forks.

# ADR 0010: Port source code into new packages, not forks

Status: accepted

## Context
Both source repos use a top-level `src/` package, and nuscenes-stability has `src/utils.py` while defect-detector has `src/utils/`. The originals are read-only inputs.

## Decision
Port code into new packages `perception/`, `defects/`, `fusion/`, `rag/`, `api/`, `eval/`. Originals stay untouched. Copy in separate commits, one per source repo, with the source repo and pinned SHA in each message. Pinned SHAs are in docs/SOURCE_REPOS.md. <!-- F:S-14 -->

## Alternatives considered
- Fork or extend a source repo in place: the top-level `src/` names collide and the originals must stay untouched. (Other alternatives were not recorded in the source plan.)

## Consequences
Behavior must be re-verified by regression tests against the source baselines (F1 0.534, AP@0.50 0.312 for the detector).
