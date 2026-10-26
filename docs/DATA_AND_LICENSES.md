Purpose: datasets used or considered for av-road-hazard, their license status, and what may ship where.

# Data and licenses

## NOTICE

Third-party datasets (nuScenes, RDD2022, MVTec AD) have their own terms and are not covered by this repo's MIT license. Datasets and weights are never committed.

## Dataset table

| Dataset | Use | License status | Local | Hosted | Repo |
|---|---|---|---|---|---|
| nuScenes (v1.0-mini first, then v1.0-full) | Detector training, evaluation, retrieval queries | Terms of Use read by the owner on 2026-10-26: CC BY-NC-SA 4.0 plus extra terms (non-commercial, attribution, share-alike). Data derived from the datasets, including tables and charts, is under the same terms. Whether a portfolio project counts as non-commercial, and how the terms apply to a model trained on the data, are not settled | Yes | No images (project policy). Derived numbers and charts are allowed under the same terms, with attribution and no suggestion that Motional endorses the project | No |
| RDD2022 | Defect classifier training; synthetic paste patches; hosted corpus samples | Sources disagree (checked 2026-10-26): the figshare page names no licence, the GitHub README says CC BY-SA 4.0, the paper names none. Working rule is the most restrictive reading: non-commercial, attribution, share-alike for derivatives. The authors have not confirmed it | Yes | Blocked until the authors confirm the licence. After that: non-US samples only, with attribution, non-commercial. The owner's own photos are the default | No images (the repo is MIT; share-alike would not fit) |
| MVTec AD | Dropped from v1 | Not assessed; about 5 GB | No | No | No |
| comma10k U-Net weights and comma10k training data | Road mask model (ADR 0013) | Verified: smp code MIT; weights MIT (HF model card); comma10k data MIT (GitHub README). UNVERIFIED: encoder ImageNet pre-training provenance; timm licence (Apache-2.0, from memory); third-party image rights beyond the MIT statement. Counsel review open | Weights fetched by `tools/fetch_road_model.py` into `<checkpoint dir>/road_mask/comma10k_unet`; no images | No images or weights hosted by this repo | No: weights and images never committed; SHA-256 pinned in `defects/road_crop.py` |
| torchvision COCO detector weights (Faster R-CNN ResNet50 FPN v2, MobileNetV3 FPN) | Object detector (ADR 0014) | BSD-3 torchvision code; weights trained on COCO. Terms of the COCO training data not verified by counsel | Weights fetched by `tools/fetch_detector.py` into `<checkpoint dir>/detector`; no images | No images or weights hosted by this repo | No: weights and images never committed; SHA-256 pinned in `perception/detector.py` |

## Per-dataset facts and policy

- **nuScenes.** Terms of Use (last updated 2021-11-16) read by the owner on 2026-10-26. The datasets are CC BY-NC-SA 4.0 plus the Dataset Terms, which win on conflict. In short: non-commercial only; attribution; share-alike for adapted material; "any data derived from the Datasets, which may appear in any format such as tables and charts" is subject to the same terms; no implying that Motional sponsors or endorses the project, and no use of its name or logo; the user is responsible for privacy and other law when publishing; an indemnification clause. "Commercial" covers any activity with the expectation of generating revenue now or in the future, in academia or industry. The terms say nothing about models trained on the data; "derived data" is broad, so weights fine-tuned on nuScenes should be treated as under the same terms. Policy: nuScenes images stay local (a project choice, stricter than the terms; also privacy); no weights or derived datasets are committed to the repo; the README states that the project uses nuScenes and is not affiliated with Motional. Still UNVERIFIED: whether using the project in a job search is non-commercial, and whether a public demo may serve a model fine-tuned on nuScenes. The terms invite questions at nuScenes@motional.com. <!-- F:L-01 -->
- **RDD2022.** Checked 2026-10-26: the [figshare record](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) page names no licence (an earlier API lookup showed CC BY 4.0, which the page does not confirm); the [GitHub README](https://github.com/sekilab/RoadDamageDetector) says CC BY-SA 4.0 in its general licence section; the [paper](https://arxiv.org/pdf/2209.08538) names none and points to GitHub. The Mendeley record that says CC BY-NC 3.0 is for RDD2020, not RDD2022. Policy: use the most restrictive reading (non-commercial, attribution, share-alike for derivatives). Publishing stays blocked until the authors confirm the licence and that is recorded here; the rule below is what a confirmed licence would still require. The demo stays non-commercial and credits the authors. The US subset is Google Street View imagery and stays out of anything public (Google's terms not checked). No RDD2022 images are committed to this repo. Still UNVERIFIED: the authors have not confirmed this reading, so asking them is the way to close it. <!-- F:L-02 -->
- **MVTec AD.** Dropped from v1. It was the only reason for Kaggle in defect-detector (about 5 GB). <!-- F:L-04 -->
- **comma10k U-Net weights and comma10k training data.** Used for the road mask ([ADR 0013](adr/0013-road-mask-model.md)). Source: Hugging Face `commaai/comma10k-segnet` at revision `b642c614eb0737fe7befa229f47fbd0245033038` (files `config.json` and `model.safetensors`, 38,502,740 bytes), SHA-256 pinned in `defects/road_crop.py`. Loaded with safetensors only, never pickle.
- **torchvision COCO detector weights.** Used for the object detector ([ADR 0014](adr/0014-object-detector.md)), zero-shot. Source: `download.pytorch.org`, SHA-256 of the published `.pth` and of the converted safetensors file pinned in `perception/detector.py`. Loaded from safetensors only.
  - Verified: smp code MIT (package metadata); weights MIT (Hugging Face model card header `license: mit`); comma10k training data MIT (its GitHub README: "MIT license, no academic only restrictions").
  - UNVERIFIED: provenance and licence of the encoder's ImageNet pre-training (the model card does not say); the timm licence (Apache-2.0, from memory); whether comma10k images carry third-party rights beyond the MIT statement.
  - This repo hosts no images and no weights. Weights are fetched locally by `tools/fetch_road_model.py` and never committed.
  - Legal review of this licence chain is open (see [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)).

## Hosted corpus

Alternative datasets for the hosted corpus were searched and none had a verifiable license. Options left: the maintainer's own photos (no license risk), plus RDD2022 samples only after verification. Decision: open (RDD2022 samples after license verification, the owner's own photos, or both; see [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)). <!-- F:L-06 -->

## Hosted demo data model

Upload-your-own-image analysis. The hosted RAG searches the hosted corpus (open: RDD2022 samples after license verification, the owner's own photos, or both) and session uploads. nuScenes-derived numbers, text and charts may be shown under the nuScenes terms (non-commercial, attribution); nuScenes images are not shown. <!-- F:L-07 -->

## Manifests

`data/MANIFEST.json` and `checkpoints/MANIFEST.json` record dataset versions and checkpoint hashes (SHA-256). They are tracked in git (via negation rules in `.gitignore`) while the files themselves are not. Download scripts verify hashes. The exact JSON shape is defined in Sprint 0. <!-- F:L-08 -->

## Uploaded images (hosted demo)

Validate type and size, decode safely, strip EXIF GPS, rate limit. <!-- F:L-09 -->

## Disk sizes

See [ENVIRONMENT.md](ENVIRONMENT.md): RDD2022 about 13.3 GB download / about 15 GB extracted; MVTec AD about 5 GB; nuScenes full hundreds of GB (UNVERIFIED).

## Verification status summary

| License | Status |
|---|---|
| nuScenes | Terms read 2026-10-26 (CC BY-NC-SA 4.0 plus extra terms); Motional (nuScenes@motional.com) confirmed by email, recorded 2026-10-26 from the owner's report (not verbatim; paste the reply here): a portfolio project counts as non-commercial as long as it is not used in actual production, and a public demo may serve a model fine-tuned on nuScenes. This holds only while the project stays non-commercial. Weights stay out of the repo and carry the nuScenes terms. |
| RDD2022 | Partly verified: sources disagree; working rule is non-commercial, attribution, share-alike; US subset excluded; authors not yet asked |
| comma10k U-Net weights and data | Code, weights and data MIT (verified); encoder pre-training and timm unverified; counsel review open |
| torchvision COCO detector weights | BSD-3 code; COCO training-data terms not verified by counsel (OPEN_QUESTIONS D1) |
| Repo license (MIT) | Recommendation, not a decision |

## Manifest entry format (L-08, format only; code arrives with workstream G)

`data/MANIFEST.json` and `checkpoints/MANIFEST.json` are `{"schema_version": 1, "entries": [...]}`. Each entry: `path` (relative to the manifest's directory), `sha256`, `size_bytes`, `source` (dataset name, URL, or training run id), `version` (dataset version or git SHA of the producing code), `license` (or the literal `unverified`), and optional `retrieved` (ISO date). Download scripts write entries and verify hashes. <!-- F:L-08 -->
