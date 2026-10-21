Purpose: ADR 0011: core-first gating of RAG, the React UI, and deployment.

# ADR 0011: Core-first gating

Status: proposed (until accepted)

## Context
RAG, the React UI, and deployment depend on a validated core (detection, road-masked tiling, fusion, synthetic evaluation); building them first risks polishing results that do not hold up. The plan is partly stack-driven (RAG, two LLM backends, React, Docker, HF Spaces) on models that do not yet work well on target data. See docs/QUALITY_BAR.md.

## Decision
Recommended: keep detection, road-masked tiling, fusion and the synthetic evaluation, with a simple API and a small demo; add RAG only if the core results justify it. Gate: do not start D (RAG), F (React UI), or I (deploy) beyond stubs until A, B, C and the synthetic fusion evaluation produce reportable numbers, or the maintainer overrides. <!-- F:Q-03 --> <!-- F:Q-04 -->

## Alternatives considered
- No gate: faster demo, but supporting features may be built on results that do not hold up.
- Cut RAG and React from v1 entirely: stronger on evidence, but removes planned scope.

## Consequences
Delays D, F and I if the core is slow. Acceptance of this gate is an open question (docs/OPEN_QUESTIONS.md).
