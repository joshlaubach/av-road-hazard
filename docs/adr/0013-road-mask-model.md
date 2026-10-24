Purpose: ADR 0013: road masks come from the comma10k U-Net, with weights fetched and hash-checked, not committed.

# ADR 0013: Road-mask model (comma10k U-Net)

Status: accepted (owner request, 2026-10-23). Counsel review of the licence chain is still open (OPEN_QUESTIONS.md).

## Context
Road-masked tiling (ADR 0002) needs a drivable-road mask for every frame. The exact method was undecided (OPEN_QUESTIONS.md, "Road mask method"). Candidates were measured on 148 human-annotated tiles (51 crack, 38 no-crack, 31 not-road, 28 ambiguous). The measure is AUC for separating not-road tiles by road share.

| Method | AUC | Weights |
|---|---|---|
| comma10k U-Net (chosen) | 0.98 | Yes |
| UperNet, ADE20K | 0.94 | Yes |
| SegFormer, Cityscapes | 0.93 | Yes |
| Geometric/colour method | 0.63 | No |
| Fixed geometric trapezoid | 0.58 | No |

Speed for the chosen model: about 0.024 s per frame on an RTX 5080 laptop GPU. CPU speed was not measured.

Preprocessing finding: the model's shipped eval transform is a bilinear resize with no normalisation. Raw 0-255 input gave AUC 0.98; ImageNet normalisation gave 0.81. OpenCV is not needed; a torch-only resize gave the same AUC.

## Decision
Use the comma10k U-Net:

- Architecture: segmentation-models-pytorch (smp) `Unet`, EfficientNet-B2 encoder, 5 classes, 512x384 input.
- Road = class 0 (road) + class 1 (lane markings). Class order is taken from the comma10k README and checked visually; beyond that it is UNVERIFIED.
- Source: Hugging Face repo `commaai/comma10k-segnet`, pinned to revision `b642c614eb0737fe7befa229f47fbd0245033038`. Files: `config.json` and `model.safetensors` (38,502,740 bytes).
- Both files are pinned by SHA-256 in `defects/road_crop.py`.
- Weights are never committed. `tools/fetch_road_model.py` fetches them into `<checkpoint dir>/road_mask/comma10k_unet`.
- Loaded with safetensors only. Pickle is never used.
- Threshold: `DEFAULT_MIN_ROAD = 0.4`. This value is provisional.

Measured effect of the threshold on the 148 tiles:

| min_road | Not-road tiles dropped | Crack tiles lost (of 51) |
|---|---|---|
| 0.3 | 74% | 0 |
| 0.4 | 87% | 1 |
| 0.5 | 90% | 3 |

Effect on the classifier (estimated, not a new evaluation):

- Precision at classifier score cut 0.9: 0.70 to 0.76 (strict truth). 90% bootstrap intervals are about 0.56-0.80 and 0.63-0.84, and they overlap.
- Precision at cut 0.5: 0.32 to 0.38. The intervals overlap.
- Recall is unchanged at min_road 0.4.

The filter helps modestly. It does not fix over-firing. Most remaining false alarms are model errors on clean pavement, which needs hard-negative training.

Licences (see [DATA_AND_LICENSES.md](../DATA_AND_LICENSES.md)):

- smp code: MIT (verified from package metadata).
- Weights: MIT (verified from the Hugging Face model card header, `license: mit`).
- Training data comma10k: MIT (verified from the GitHub README, "MIT license, no academic only restrictions").
- UNVERIFIED: provenance and licence of the encoder's ImageNet pre-training (the model card does not say); the timm licence (Apache-2.0, from memory); whether comma10k images carry third-party rights beyond the MIT statement.

## Alternatives considered
- UperNet, ADE20K: second-best separation (AUC 0.94). Rejected because the ADE20K data terms are UNVERIFIED (believed non-commercial), even though the weights are MIT (verified).
- SegFormer, Cityscapes: AUC 0.93. Rejected because the weights card gives licence "other" with no licence text, and Cityscapes is believed non-commercial (UNVERIFIED).
- Geometric/colour method with no weights: AUC 0.63. Rejected on quality.
- Fixed geometric trapezoid: AUC 0.58. Rejected on quality.

## Consequences
- Dependencies added to the `ml` extra: segmentation-models-pytorch, safetensors and huggingface_hub (see `pyproject.toml`). The lock files are unchanged because the `ml` extra is not locked.
- Setup step added: `python -m tools.fetch_road_model` must run before the road mask works. A missing or mismatched file is handled in [RUNBOOK.md](../RUNBOOK.md).
- The threshold `DEFAULT_MIN_ROAD = 0.4` is provisional. The test set is small, so the thresholds are noisy.
- Current state: the mask is not yet wired into `api/pipeline.py` (`Pipeline.road_mask` still returns `None`), and `defects.infer.find_defects` is still a stub.
- Follow-ups:
  - Shoulder-edge cracks: two of 51 crack tiles are lost at 0.5 because the model treats the shoulder as undrivable. Dilating the mask is a candidate.
  - Hard-negative training for over-firing.
  - Set `min_road` from more annotated tiles.
  - Nordic snow robustness check. The model was trained mostly on US and highway frames, so Nordic snow roads are not guaranteed.
  - Wire the mask into the pipeline and into `find_defects`.
  - Licence review of the chain (see above).
