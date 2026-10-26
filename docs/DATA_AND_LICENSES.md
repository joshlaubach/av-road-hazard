Purpose: datasets used or considered for av-road-hazard, their license status, and what may ship where.

# Data and licenses

## NOTICE

Third-party datasets (nuScenes, RDD2022, MVTec AD, Waymo Open Dataset) have their own terms and are not covered by this repo's MIT license. Datasets and weights are never committed.

## Dataset table

| Dataset | Use | License status | Local | Hosted | Repo |
|---|---|---|---|---|---|
| nuScenes (v1.0-mini first, then v1.0-full) | Detector training, evaluation, retrieval queries | Non-commercial; restricts redistribution of images. UNVERIFIED: a web search did not surface the terms; the owner must verify | Yes | No images; text/metadata/charts only if the owner confirms (default: do not show) | No |
| RDD2022 | Defect classifier training; synthetic paste patches; hosted corpus samples | figshare record (CRDDC 2022) names no license. UNVERIFIED | Yes | Only after license verification | No images until verified |
| MVTec AD | Dropped from v1 | Not assessed; about 5 GB | No | No | No |
| Waymo Open Dataset | Optional | Non-commercial; needs separate application/approval (UNVERIFIED terms) | Optional, local-only | No | No |
| comma10k U-Net weights and comma10k training data | Road mask model (ADR 0013) | Verified: smp code MIT; weights MIT (HF model card); comma10k data MIT (GitHub README). UNVERIFIED: encoder ImageNet pre-training provenance; timm licence (Apache-2.0, from memory); third-party image rights beyond the MIT statement. Counsel review open | Weights fetched by `tools/fetch_road_model.py` into `<checkpoint dir>/road_mask/comma10k_unet`; no images | No images or weights hosted by this repo | No: weights and images never committed; SHA-256 pinned in `defects/road_crop.py` |
| torchvision COCO detector weights (Faster R-CNN ResNet50 FPN v2, MobileNetV3 FPN) | Object detector (ADR 0014) | BSD-3 torchvision code; weights trained on COCO. Terms of the COCO training data not verified by counsel | Weights fetched by `tools/fetch_detector.py` into `<checkpoint dir>/detector`; no images | No images or weights hosted by this repo | No: weights and images never committed; SHA-256 pinned in `perception/detector.py` |

## Per-dataset facts and policy

- **nuScenes.** License is non-commercial and restricts redistribution of images (UNVERIFIED; the owner must verify). Policy: nuScenes images stay local; never in the repo, Docker images, hosted bundle, hosted corpus, screenshots published in docs, or demo video frames. <!-- F:L-01 -->
- **RDD2022.** **[CRITICAL]** The figshare record (CRDDC 2022) page named no license when checked. UNVERIFIED. Policy: do not publish any RDD2022 images (hosted corpus samples, README images, fixtures committed to the repo) until the license is confirmed. <!-- F:L-02 -->
- **Waymo Open Dataset.** Non-commercial and requires separate application/approval (UNVERIFIED terms); optional, local-only, excluded from v1 by default. <!-- F:L-03 -->
- **MVTec AD.** Dropped from v1. It was the only reason for Kaggle in defect-detector (about 5 GB). <!-- F:L-04 -->
- **comma10k U-Net weights and comma10k training data.** Used for the road mask ([ADR 0013](adr/0013-road-mask-model.md)). Source: Hugging Face `commaai/comma10k-segnet` at revision `b642c614eb0737fe7befa229f47fbd0245033038` (files `config.json` and `model.safetensors`, 38,502,740 bytes), SHA-256 pinned in `defects/road_crop.py`. Loaded with safetensors only, never pickle.
- **torchvision COCO detector weights.** Used for the object detector ([ADR 0014](adr/0014-object-detector.md)), zero-shot. Source: `download.pytorch.org`, SHA-256 of the published `.pth` and of the converted safetensors file pinned in `perception/detector.py`. Loaded from safetensors only.
  - Verified: smp code MIT (package metadata); weights MIT (Hugging Face model card header `license: mit`); comma10k training data MIT (its GitHub README: "MIT license, no academic only restrictions").
  - UNVERIFIED: provenance and licence of the encoder's ImageNet pre-training (the model card does not say); the timm licence (Apache-2.0, from memory); whether comma10k images carry third-party rights beyond the MIT statement.
  - This repo hosts no images and no weights. Weights are fetched locally by `tools/fetch_road_model.py` and never committed.
  - Legal review of this licence chain is open (see [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)).

## Hosted corpus

Alternative datasets for the hosted corpus were searched and none had a verifiable license. Options left: the maintainer's own photos (no license risk), plus RDD2022 samples only after verification. CARLA is not used ([ADR 0012](adr/0012-drop-carla.md)). Decision: open (RDD2022 samples after license verification, the owner's own photos, or both; see [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)). <!-- F:L-06 -->

## Hosted demo data model

Upload-your-own-image analysis. The hosted RAG searches the hosted corpus (open: RDD2022 samples after license verification, the owner's own photos, or both) and session uploads. nuScenes-derived records may be shown as text/metadata/charts only if the owner confirms that is permitted (open question; default: do not show). <!-- F:L-07 -->

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
| comma10k U-Net weights and data | Code, weights and data MIT (verified); encoder pre-training and timm unverified; counsel review open |
| torchvision COCO detector weights | BSD-3 code; COCO training-data terms not verified by counsel (OPEN_QUESTIONS D1) |
| Repo license (MIT) | Recommendation, not a decision |

## Manifest entry format (L-08, format only; code arrives with workstream G)

`data/MANIFEST.json` and `checkpoints/MANIFEST.json` are `{"schema_version": 1, "entries": [...]}`. Each entry: `path` (relative to the manifest's directory), `sha256`, `size_bytes`, `source` (dataset name, URL, or training run id), `version` (dataset version or git SHA of the producing code), `license` (or the literal `unverified`), and optional `retrieved` (ISO date). Download scripts write entries and verify hashes. <!-- F:L-08 -->
