Purpose: ADR 0006: React/Next.js front end.

# ADR 0006: React/Next.js front end

Status: accepted

## Context
Streamlit is faster to build (defect-detector already has a dashboard to adapt). React/Next.js was chosen anyway; the tension with the 1-2 week timeline is accepted.

## Decision
Next.js + TypeScript backed by FastAPI. The TypeScript client is generated from `openapi.json`, which is the single contract. Fallback if the timeline slips: Streamlit on the same API. Rejected for v1: Streamlit and Gradio. <!-- F:R-05 --> <!-- F:U-07 -->

## Alternatives considered
- Streamlit: faster; rejected for v1 (kept as the fallback).
- Gradio: same.

## Consequences
Larger integration surface (Python + TypeScript). React polish is first in the scope-cut order. Under the proposed core-first gate (ADR 0011) the UI stays at stub level until core metrics exist.
