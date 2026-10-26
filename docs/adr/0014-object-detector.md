Purpose: ADR 0014: objects come from zero-shot torchvision COCO detectors, with weights fetched and hash-checked, not fine-tuned or committed.

# ADR 0014: Object detector (zero-shot COCO)

Status: accepted (owner decisions, 2026-10-25): zero-shot COCO first; fine-tuning only after a measured gap and explicit licence approval; fail loudly when the detector is unavailable; CPU target p95 of 2 seconds per image, with the lighter model preferred if the heavier one cannot meet it. The score threshold (0.5) is provisional.

## Context
Fusion weights a hazard by nearby road users, and the tile filter skips tiles that objects cover, so `perception.detect()` must return `Object(label, score, box)` with the labels in `configs/fusion.yaml`. The workstream A brief (docs/agents/A-perception.md) planned a Faster R-CNN fine-tuned on nuScenes. Two things argued against starting there:

- No weights exist from the source repo (`nuscenes-stability` keeps its outputs out of git), so its F1 0.534 and AP@0.50 0.312 could only be reproduced by retraining.
- nuScenes is non-commercial and restricts image redistribution (UNVERIFIED, see DATA_AND_LICENSES.md), and weights trained on it plausibly inherit that.

COCO already has person, bicycle, car, motorcycle, bus and truck, which map onto the six fusion labels (`perception/labels.py`).

## Decision
Use torchvision's COCO-trained Faster R-CNN models as published, with no fine-tuning, and measure them on nuScenes before training anything.

- Two models are pinned in `perception/detector.py`: `fasterrcnn_resnet50_fpn_v2` and `fasterrcnn_mobilenet_v3_large_fpn`. Default: MobileNet (config `perception.model`, env `DETECTOR_MODEL`).
- Weights are never committed or downloaded implicitly. `tools/fetch_detector.py` downloads the published `.pth` over https, checks its SHA-256 (each also matches the 8-hex fragment in its filename), opens it with torch's weights-only loader, saves safetensors, checks that file's SHA-256, and moves it into `<checkpoint dir>/detector`. `perception/detector.py` loads only that safetensors file, after hashing it.
- COCO classes map to fusion labels: person to pedestrian, bicycle to cyclist (a rider is not detected separately), car, motorcycle, bus, truck. Pinned by `tests/perception/test_labels.py`.
- Update 2026-10-26: fusion also knows child, construction_worker, police_officer, personal_mobility, construction_vehicle, barrier, traffic_cone and debris, with provisional weights. A COCO model cannot say these. Child, worker, police officer and personal mobility count as pedestrian, and a construction vehicle as truck, when it is scored. Barrier, cone and debris have no COCO family, so they are counted as "outside the vocabulary" and not scored. The owner chose to fine-tune on nuScenes for them, which this ADR said needs a measured gap and licence approval (see the nuScenes terms in DATA_AND_LICENSES.md).
- Score threshold `perception.score_threshold = 0.5`. This is provisional.
- Failure: with stubs off, a missing, changed or unloadable detector raises `DetectorUnavailable`, and the API answers 503 with a generic message; the reason is logged. An empty object list would silently lower hazard scores, so it is never returned.

## Evidence
nuScenes v1.0-mini, 404 CAM_FRONT frames, `tools/eval_detector.py`. Scenes are split 80/20 by sorted token; the threshold is chosen on the 8 training scenes and scored on the 2 held-out scenes (80 frames, so differences of a few points there are noisy). Ground truth is the projected 3D boxes; "evaluable" means at least 40% visible and at least 32 x 32 px, a cut-off chosen for this tool, so the 0.534 and 0.312 baselines are context and not a like-for-like comparison.

| | ResNet50 FPN v2 | MobileNetV3 FPN |
|---|---|---|
| F1-best threshold on training scenes | 0.90 | 0.70 |
| Held-out F1 / AP@0.50 | 0.61 / 0.51 | 0.50 / 0.34 |
| All frames F1 / AP@0.50 | 0.58 / 0.37 | 0.50 / 0.30 |
| Day / night F1 | 0.58 / 0.55 | 0.52 / 0.42 |
| Recall by size, <32 px / 32-96 px / >96 px | 13% / 34% / 54% | 0.3% / 23% / 50% |
| CPU p95 at 24 / 4 / 2 threads | 1.2 / 2.65 / 4.99 s | 0.19 / 0.34 / 0.53 s |

Timing is one laptop CPU with the thread count limited to imitate a smaller host. ResNet misses the 2 second target at 4 threads or fewer, so MobileNet is the default.

Effect on tile defects (all frames, road mask on, shipped classifier thresholds): the defect count was 87 with no objects, 79 with ground-truth objects, and 80 with detected objects from either model. All 79 ground-truth defects were found with detected objects, and one extra tile survived because its object was missed. MobileNet at score 0.3 and at 0.5 gave the same 80 defects, so the threshold makes no measurable difference to this use. The effect of object masking on this data is small (8 of 87 defects).

## Alternatives considered
- Fine-tune on nuScenes now (the original brief): rejected for now. It needs a licence decision and nuScenes mini is small (10 scenes). It stays open if a measured gap justifies it.
- ResNet50 as the default: more accurate, but misses the CPU target on small hosts. It stays available through config for local or GPU use.
- Choose the threshold by best F1 (0.70 for MobileNet): rejected, because masking tiles values recall over precision, and the data shows no difference between 0.3 and 0.5.

## Consequences
- Setup step added: `python -m tools.fetch_detector --model NAME` must run before the detector works. See [RUNBOOK.md](../RUNBOOK.md).
- Weak spots: small objects (under 32 px) are almost never found, and MobileNet's night recall is 33%. Neither affects the tile filter much on this data, but both limit the evidence given to fusion.
- Licences: torchvision code is BSD-3; the weights are trained on COCO. Not verified by counsel (see OPEN_QUESTIONS.md, D1).
- The detector's labels follow COCO. Norwegian RDD2022 photos mask vehicles as grey blobs, so detection there is untested (D3).
- The API now has a 503 path; it is not declared on the routes, so the frozen `openapi.json` is unchanged.
- Follow-ups are D1 to D5 in OPEN_QUESTIONS.md.
