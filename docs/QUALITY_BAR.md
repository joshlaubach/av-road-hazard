Purpose: the quality bar for av-road-hazard, written in response to the owner's "AI slop" concern: criteria, the core-first gate, and what "done and credible" means.

# Quality bar

## The concern

**[CRITICAL]** The owner said "I think my current project is ai slop". Treat this as a quality requirement. <!-- F:U-13 -->

The assistant agreed the risk is real. The plan is partly stack-driven (RAG, two LLM backends, React, Docker, HF Spaces) attached to two models that do not yet work well on the target data: classifier overfit 0.92 vs 0.73, detector trained on a 10-scene mini set, possible domain gap, no combined ground truth, and no real user question beyond demo queries. <!-- F:Q-01 -->

## Criteria for a credible project

1. **A real question with a measured answer**, including honest negative results. Example: can a classifier trained on close-up road damage find damage in windshield frames, and how far away?
2. **Depth in one place.** Road-masked tiling plus synthetic evaluation is the interesting core. RAG, LLM switching, and React are decoration unless the core works.
3. **Evidence it works:** baseline, ablation, failure cases, stated limits.
4. **Everything understood by the owner.** If the owner cannot explain tiling and fusion in an interview, it reads as slop even if it runs.

<!-- F:Q-02 -->

## Recommended scope cut (a recommendation, not an owner decision)

Keep detection, road-masked tiling, fusion, and the synthetic evaluation, with a simple API and a small demo. Add RAG only if the core results justify it. ADR 0011 records this with status "proposed". The owner did not yet answer whether this applies to "current project" meaning the plan or the existing repos; logged as an open question. <!-- F:Q-03 -->

## Core-first gate (proposed)

Do not start D (RAG), F (React UI), or I (deploy) beyond stubs until A, B, C and the synthetic fusion evaluation produce reportable numbers, or the owner overrides. See [adr/0011-core-first-gate.md](adr/0011-core-first-gate.md). <!-- F:Q-04 -->

## Anti-slop rules for claims

- Every README claim must be backed by a number from `eval/results.json` or marked as unmeasured.
- State the effective detection range.
- State the classifier false-positive rate on non-road tiles.
- Do not present pasted-patch results as real-world accuracy.

<!-- F:Q-05 -->

## Interview-explainability rule

The owner must be able to explain each core component. Each agent brief ends with an "Owner explainer" paragraph the agent writes in its PR: what it does, why, and what could be wrong. <!-- F:Q-06 -->

## What "done and credible" means

The Definition of Done in ENGINEERING.md section 10 is met for every PR, plus: the core components have reportable numbers in `eval/results.json`, limitations in the README are measured rather than guessed, and the owner can explain tiling and fusion unaided.

## Pending offer

The assistant also offered to review the existing repos honestly and say which are worth keeping. Pending the owner's answer. <!-- F:Q-07 -->
