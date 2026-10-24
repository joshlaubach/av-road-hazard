Purpose: ADR 0007: Evaluate without hand labeling.

# ADR 0007: Evaluate without hand labeling

Status: accepted

## Context
Hand labeling is out of scope; all evaluation must be automatic. There is no combined ground truth: nuScenes has no road-defect labels, RDD2022 has no object labels.

## Decision
Three automatic evaluations: per-model metrics, retrieval tests auto-generated from nuScenes annotations, and a synthetic fusion test pasting RDD2022 defect patches onto nuScenes/CARLA road regions at known positions. See docs/EVALUATION.md. <!-- F:R-06 -->

## Alternatives considered
- Hand-label about 100 frames: rejected.
- Separate evaluation only, no fusion check: rejected.

## Consequences
The synthetic test is easier than real defects; its limits must be stated and its results never presented as real-world accuracy. <!-- F:R-07 -->
Update 2026-10-23: CARLA was dropped from the project (ADR 0012); read "nuScenes/CARLA" as "nuScenes" and ignore the CARLA corpus.
