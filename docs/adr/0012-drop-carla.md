Purpose: ADR 0012: CARLA is not used in this project.

# ADR 0012: Drop CARLA

Status: accepted (owner decision, 2026-10-23)

## Context
The first plan used the CARLA simulator for three things: generating frames for the hosted corpus, supplying non-road frames as automatically labelled negatives, and providing road regions for the synthetic fusion test. It also carried an open question about how to add potholes and cracks to CARLA (decals or textures), because stock CARLA has none, and a host-only runtime outside Docker that does not run on macOS. <!-- F:L-05 --> <!-- F:L-06 -->

## Decision
CARLA is not used at all: no frame generator, no simulator corpus, no CARLA host, no defect decals, and no reuse of the `drivenet-carla-av` driving policy. The `source` value `carla` stays in the frozen contract (`fusion/schema.py`, `openapi.json`) and in one fusion test until a contract-change PR removes it; nothing produces it.

- Non-road negatives and the synthetic fusion test use nuScenes road regions only (nuScenes stays local-only, ADR 0005).
- The hosted corpus is RDD2022 samples, only after the RDD2022 license is verified, and optionally the owner's own photos (no license risk). Which of these the hosted corpus uses is an open question (OPEN_QUESTIONS.md).
- The scope-cut order becomes: React polish, Claude/LLaMA toggle UI (keep the config switch), v1.0-full evaluation.

## Alternatives considered
- Keep CARLA for object-side frames only: still needs the Windows host, a CARLA license check, and a frame generator, for frames that exercise only the object side.
- Add defect decals to CARLA: more work than the evaluation value justifies, and the synthetic fusion test already covers the case with real RDD2022 patches.

## Consequences
The hosted corpus has fewer sources and depends on RDD2022 license verification or the owner's own photos. The CARLA host, runbook entry, frame-generator task and CARLA license check are removed from the plan. Older ADRs (0002, 0005, 0007, 0008) and the traceability rows keep their original wording with a note that CARLA was dropped.
