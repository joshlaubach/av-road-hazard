Purpose: ADR 0003: Drop the DQN patch gate in v1.

# ADR 0003: Drop the DQN patch gate in v1

Status: accepted (owner approved)

## Context
The source pipeline has a DQN that selects which patches to classify. It has no usable checkpoint (the committed `dqn.pth` predates the stop action and will not load), RL training adds cost, and its reward oracle is the same EfficientNet classifier it feeds, so it learns where that classifier fires rather than where defects are. Token and time budget are tight (two $20 plans, 1-2 weeks).

## Decision
Scan all tiles in v1; do not port the DQN. Revisit later. <!-- F:R-03 -->

## Alternatives considered
- Retrain the DQN: training cost, circular reward. Rejected for v1.

## Consequences
More tiles per frame are classified (mitigated by road masking, fewer and larger tiles). Hosted CPU latency needs limiting tile count, downscaling, caching or precompute.
