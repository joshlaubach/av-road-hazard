Purpose: datasets used or considered for av-road-hazard, their license status, and what may ship where.

# Data and licenses

## NOTICE

Third-party datasets (nuScenes, RDD2022, MVTec AD, Waymo Open Dataset, CARLA assets) have their own terms and are not covered by this repo's MIT license. Datasets and weights are never committed.

## Dataset table

| Dataset | Use | License status | Local | Hosted | Repo |
|---|---|---|---|---|---|
| nuScenes (v1.0-mini first, then v1.0-full) | Detector training, evaluation, retrieval queries | Non-commercial; restricts redistribution of images. UNVERIFIED: a web search did not surface the terms; the owner must verify | Yes | No images; text/metadata/charts only if the owner confirms (default: do not show) | No |
| RDD2022 | Defect classifier training; synthetic paste patches; hosted corpus samples | figshare record (CRDDC 2022) names no license. UNVERIFIED | Yes | Only after license verification | No images until verified |
| MVTec AD | Dropped from v1 | Not assessed; about 5 GB | No | No | No |
| Waymo Open Dataset | Optional | Non-commercial; needs separate application/approval (UNVERIFIED terms) | Optional, local-only | No | No |
| CARLA-generated frames | Hosted corpus | The owner's own data; CARLA's own license terms apply (UNVERIFIED) | Yes | Yes | No |

## Per-dataset facts and policy

- **nuScenes.** License is non-commercial and restricts redistribution of images (UNVERIFIED; the owner must verify). Policy: nuScenes images stay local; never in the repo, Docker images, hosted bundle, hosted corpus, screenshots published in docs, or demo video frames. <!-- F:L-01 -->
- **RDD2022.** **[CRITICAL]** The figshare record (CRDDC 2022) page named no license when checked. UNVERIFIED. Policy: do not publish any RDD2022 images (hosted corpus samples, README images, fixtures committed to the repo) until the license is confirmed. <!-- F:L-02 -->
- **Waymo Open Dataset.** Non-commercial and requires separate application/approval (UNVERIFIED terms); optional, local-only, excluded from v1 by default. <!-- F:L-03 -->
- **MVTec AD.** Dropped from v1. It was the only reason for Kaggle in defect-detector (about 5 GB). <!-- F:L-04 -->
- **CARLA.** Frames the owner generates are the owner's own data (no third-party dataset license; CARLA's own license terms apply, UNVERIFIED). Stock CARLA has no potholes or cracks; defects require decal/texture overlays, otherwise the CARLA corpus exercises only the object side. Decision pending (open question). <!-- F:L-05 -->

## Hosted corpus

Alternative datasets for the hosted corpus were searched and none had a verifiable license. Options left: CARLA frames plus the maintainer's own photos (no license risk), plus RDD2022 samples only after verification. Decision: CARLA frames + RDD2022 samples. <!-- F:L-06 -->

## Hosted demo data model

Upload-your-own-image analysis. The hosted RAG searches the CARLA + RDD2022-sample corpus and session uploads. nuScenes-derived records may be shown as text/metadata/charts only if the owner confirms that is permitted (open question; default: do not show). <!-- F:L-07 -->

## Manifests

`data/MANIFEST.json` and `checkpoints/MANIFEST.json` record dataset versions and checkpoint hashes (SHA-256). They are tracked in git (via negation rules in `.gitignore`) while the files themselves are not. Download scripts verify hashes. The exact JSON shape is defined in Sprint 0. <!-- F:L-08 -->

## Uploaded images (hosted demo)

Validate type and size, decode safely, strip EXIF GPS, rate limit. <!-- F:L-09 -->

## Disk sizes

See [ENVIRONMENT.md](ENVIRONMENT.md): RDD2022 about 13.3 GB download / about 15 GB extracted; MVTec AD about 5 GB; nuScenes full hundreds of GB (UNVERIFIED).

## Verification status summary

| License | Status |
|---|---|
| nuScenes | Unverified; owner to verify |
| RDD2022 | Unverified; blocks publishing any RDD2022 image |
| Waymo | Unverified; excluded from v1 by default |
| CARLA | Unverified |
| Repo license (MIT) | Recommendation, not a decision |
