Purpose: ADR 0006: React/Next.js front end.

# ADR 0006: React/Next.js front end

Status: accepted (owner approved)

## Context
The assistant recommended Streamlit for speed (defect-detector already has a dashboard to adapt). The owner chose React/Next.js, accepting the tension with the 1-2 week timeline.

## Decision
Next.js + TypeScript backed by FastAPI. The TypeScript client is generated from `openapi.json`, which is the single contract. Fallback if the timeline slips: Streamlit on the same API. Rejected for v1: Streamlit and Gradio. <!-- F:R-05 --> <!-- F:U-07 -->

## Alternatives considered
- Streamlit: faster, recommended, declined by the owner for v1.
- Gradio: same.

## Consequences
Larger integration surface (Python + TypeScript). React polish is first in the scope-cut order. Under the proposed core-first gate (ADR 0011) the UI stays at stub level until core metrics exist.
