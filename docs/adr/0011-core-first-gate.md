Purpose: ADR 0011: Core-first gate.

# ADR 0011: Core-first gate

Status: proposed (until the owner accepts)

## Context
The owner worried the project is "ai slop". The plan is partly stack-driven (RAG, two LLM backends, React, Docker, HF Spaces) on models that do not yet work well on target data. See docs/QUALITY_BAR.md.

## Decision
Recommended (a recommendation, not an owner decision): keep detection, road-masked tiling, fusion and the synthetic evaluation, with a simple API and a small demo; add RAG only if the core results justify it. Gate: do not start D (RAG), F (React UI), or I (deploy) beyond stubs until A, B, C and the synthetic fusion evaluation produce reportable numbers, or the owner overrides. <!-- F:Q-03 --> <!-- F:Q-04 -->

## Alternatives considered
- No gate: faster demo, higher slop risk.
- Cut RAG and React from v1 entirely: stronger, but not what the owner approved.

## Consequences
Delays D, F and I if the core is slow. The owner has not yet said whether this applies to the plan or the existing repos (open question).
