Purpose: ADR 0008: Two agents, sequenced sprints, cross-review.

# ADR 0008: Two agents, sequenced sprints, cross-review

Status: accepted

## Context
The owner has one Claude plan and one Codex plan, $20/month each, so usage limits are the bottleneck and at most two agents run at once (one per tool). The owner also said they have several agents working side by side; both statements are recorded. <!-- F:U-08 -->

## Decision
Sequence work in sprints (BUILD_PLAN section 1): a Claude lane and a Codex lane, one workstream per fresh session with only its brief and `fusion/schema.py`. Every PR is reviewed by the other tool using the ENGINEERING.md section 2 checklist in one short pass. GPU training and downloads run on the owner's machine in the background. <!-- F:A-01 --> <!-- F:A-03 -->

## Alternatives considered
- Many agents in parallel: not supported by two plans' usage limits.

## Consequences
Lane boundaries must be respected (owned paths). Stubs let streams proceed in parallel. Scope-cut order if sprints overrun: React polish, CARLA corpus, LLM toggle UI, v1.0-full evaluation.
