Purpose: ADR 0009: Hosting on Hugging Face Spaces and Qdrant Cloud.

# ADR 0009: Hosting on Hugging Face Spaces and Qdrant Cloud

Status: accepted

## Context
Budget is up to about $20/month. Hosted compute is CPU only. EfficientNet-B4 over 64 patches plus Faster R-CNN on CPU is slow.

## Decision
Hugging Face Spaces (CPU) plus Qdrant Cloud free tier. Claude key as a Space secret. Limit tile count, downscale, cache, and precompute outputs for the sample set. <!-- F:R-04 --> <!-- F:E-08 -->

## Alternatives considered
- Fly.io or Render: more control but uses most of the budget. Rejected.
- GPU instance: $50+/month. Rejected.

## Consequences
Live analysis of uploads will be slow; the demo relies on precomputed samples. Rate limits, upload limits, and the Claude spend cap are mandatory.
