Purpose: ADR 0005: nuScenes stays local only.

# ADR 0005: nuScenes stays local only

Status: accepted

## Context
nuScenes is non-commercial and restricts redistribution of images (UNVERIFIED; to be verified). The demo is public.

## Decision
nuScenes images stay on the local machine: never in the repo, Docker images, hosted bundle, hosted corpus, published screenshots, or demo video frames. nuScenes-derived text/metadata/charts may be shown hosted only if confirmed permitted (default: do not show). The hosted corpus is CARLA frames plus RDD2022 samples (after license verification). <!-- F:L-01 -->

## Alternatives considered
- Host nuScenes samples: license risk. Rejected.
- Find another dataset: no alternative had a verifiable license. Options left: CARLA frames, own photos, RDD2022 samples after verification.

## Consequences
Retrieval quality on nuScenes can only be shown through numbers, not images. Hosted RAG searches a different corpus than local RAG. A review step (Phase 2) checks no nuScenes images are in the hosted bundle.
