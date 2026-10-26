Purpose: quality bar and evidence standards for av-road-hazard: criteria for a credible result, the core-first gate, and what "done" means.

# Quality bar and evidence standards

## Why these standards exist

Quality is a requirement, not a polish step. <!-- F:U-13 --> The plan is partly stack-driven (RAG, two LLM backends, React, Docker, HF Spaces) and is attached to two models that do not yet work well on the target data: classifier overfit (train F1 0.92 vs validation 0.73), a detector trained on a 10-scene mini set, a possible domain gap, no combined ground truth, and no real user question beyond demo queries. Without standards, supporting features can outrun the evidence behind the core. <!-- F:Q-01 -->

## Criteria for a credible result

1. **A measured question with an answer**, including honest negative results. Example: can a classifier trained on close-up road damage find damage in windshield frames, and how far away?
2. **Depth in the core.** Road-masked tiling plus synthetic evaluation is the core. RAG, LLM switching, and React are supporting features and do not count as results unless the core works.
3. **Evidence it works:** baseline, ablation, failure cases, stated limits.
4. **Explainability.** Maintainers can explain each core component, tiling and fusion in particular, without reading the code.

<!-- F:Q-02 -->

## Core scope (see ADR 0011)

Keep detection, road-masked tiling, fusion, and the synthetic evaluation, with a simple API and a small demo. Add RAG only if the core results justify it. [ADR 0011](adr/0011-core-first-gate.md) records this; the maintainer accepted it on 2026-10-26. <!-- F:Q-03 -->

## Core-first gate (accepted, ADR 0011)

Do not start D (RAG), F (React UI), or I (deploy) beyond stubs until A, B, C and the synthetic fusion evaluation produce reportable numbers, or the maintainer overrides. <!-- F:Q-04 -->

## Evidence rules for claims

- Every README claim must be backed by a number from `eval/results.json` or marked as unmeasured.
- State the effective detection range.
- State the classifier false-positive rate on non-road tiles.
- Do not present pasted-patch (synthetic) results as real-world accuracy.

<!-- F:Q-05 -->

## Maintainer explainer

Each agent brief ends with a "Maintainer explainer" paragraph that the agent writes in its PR: what the component does, why, and what could be wrong. <!-- F:Q-06 -->

## What "done" means

The Definition of Done in ENGINEERING.md section 10 is met for every PR, plus: the core components have reportable numbers in `eval/results.json`, limitations in the README are measured rather than guessed, and maintainers can explain tiling and fusion unaided.

## Existing repos

A candid review of which source repos are worth keeping is a possible follow-up. <!-- F:Q-07 -->
