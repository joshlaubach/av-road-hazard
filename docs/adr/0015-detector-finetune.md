Purpose: ADR 0015: the plan for fine-tuning the object detector on nuScenes so it can name traffic cones and barriers, and the tests it must pass before fusion uses it.

# ADR 0015: Detector fine-tune for work-zone labels

Status: accepted (2026-10-26). The owner approved the plan and the pass limits; the 10-fold training started the same day.

## Context
The zero-shot COCO detector ([ADR 0014](0014-object-detector.md)) has no word for traffic cones or barriers. On nuScenes mini (404 front-camera frames) that is 20% of all annotated objects, and 457 of the clearly visible ones. The owner chose to fine-tune. Motional confirmed that a portfolio project counts as non-commercial and that a public demo may serve a model fine-tuned on nuScenes ([DATA_AND_LICENSES.md](../DATA_AND_LICENSES.md)). The weights still carry the nuScenes terms and stay out of the repo.

The data is the problem, not the code. What nuScenes mini holds (clearly visible boxes, all six cameras, 2,424 images from 10 scenes):

| Scene | Barriers | Traffic cones |
|---|---|---|
| scene-0061 | 975 | 493 |
| scene-0553 | 492 | 0 |
| scene-0655 | 0 | 199 |
| scene-0757 | 0 | 129 |
| scene-1077 (night) | 44 | 0 |
| scene-0103 | 0 | 42 |
| the other 4 scenes | 0 | 0 |

So barriers appear in only 3 scenes, and one construction scene holds most of everything. Frames in a scene are 0.5 s apart, so neighbors are near copies: 2,374 boxes are far fewer independent examples. The other work-zone labels are too rare to learn: 193 construction workers, 196 construction vehicles, 13 pieces of debris (all annotations, not only the visible ones).

## Options
1. **Fine-tune the pinned COCO detector on nuScenes (recommended).** Start from the published weights, keep what it knows about the six road users, and add two new classes. Cheap (minutes per run on the RTX 5080) and uses the code we already have.
2. **Train a small second detector for only cones and barriers.** Keeps the first model untouched, but doubles the work at run time and the CPU speed target (p95 2 s) is already tight.
3. **Wait for more data** (full nuScenes, about 300 GB, or our own labeled photos). The right long-term answer, but it blocks everything and is Parked until the mini result shows what is missing.

## Decision (proposed)
Option 1, with a deliberately small scope and strict tests, so a thin result cannot slip into the product.

**Scope.** Eight classes: the six existing (pedestrian, cyclist, motorcycle, car, truck, bus) plus `barrier` and `traffic_cone`. Construction worker, construction vehicle and debris stay as they are (family labels), because there are too few examples. Revisit with more data.

**Model.** Same MobileNetV3 Faster R-CNN. Replace only the last classification and box layers. The six old classes start from their COCO rows, so the model begins by knowing what it knew; the two new rows start at random. Train all layers at a small learning rate, with horizontal flips and mild color jitter. A fixed schedule chosen before looking at any held-out result (about 12 epochs, cosine learning rate), not tuned on the test scenes. The two new classes get their own score cutoff, chosen on the training scenes only (the same way as the crack classifier's thresholds), so the balance between false and missed objects is set without looking at the held-out scenes. The six old classes keep 0.5.

**Data.** All six cameras for training (2,424 images), ground truth from the projected 3D boxes exactly as `tools/eval_detector.py` does today (same visibility and size rules, so numbers stay comparable).

**Splits: leave one scene out.** Mini has 10 scenes, and a normal 80/20 split would hold out two scenes that may contain no barriers at all. Instead train 10 models, each leaving out one scene, and score every scene only with the model that never saw it. About an hour of GPU time in total. Scenes are the unit, never frames, because neighboring frames are near copies. Per-scene results are reported next to the pooled numbers.

## What it must pass (provisional, set in `configs/detector_finetune.yaml`)
Measured at the shipped score 0.5 on the held-out scenes, front camera first (to match the Done numbers), all cameras second, with 90% ranges.

1. **New labels:** for traffic cones and for barriers, precision of at least 70% and recall of at least 40%. Precision is the stricter limit on purpose: a missed cone leaves us where the zero-shot detector is today (it finds none), but a false one raises a hazard score for no reason (barrier and cone weights are 0.5, above a car's 0.3). Counted only over scenes that contain at least 20 clearly visible boxes of that class. Barriers pass only if at least two held-out scenes do so on their own. A number that passes but whose 90% range reaches below its limit is reported as a pass with a caveat, not a clean pass.
2. **No regression:** for each of the six old classes, recall no more than 3 points below the zero-shot detector on the same scenes and precision no more than 5 points below.
3. **Speed:** the CPU p95 stays at or below 2 s per image at 24, 4 and 2 threads.
4. **Sanity outside nuScenes:** on the 20 local images from `smoke_analyze` (nuScenes and RDD2022 frames) and the Stellenbosch photos, the fine-tuned model must not invent objects on empty road. Counted, shown to the owner side by side, not scored.

If any item fails, the zero-shot detector stays, the numbers are recorded in the roadmap, and the next step is more data. A result that passes on 2 or 3 scenes is reported as "works on these scenes", never as general accuracy.

## Work, in order (each step is a PR)
1. `perception/finetune.py` and `tests/perception/`: the 8-class label set, building training samples from nuScenes boxes, the head swap that keeps the six COCO rows, the leave-one-scene-out folds. Pure functions, tested without a GPU.
2. `tools/finetune_detector.py` and `configs/detector_finetune.yaml`: train one fold, with a progress bar and a run id (same habits as the classifier). A 1-epoch smoke run first, to check the loss falls.
3. `tools/eval_detector.py` and `eval/detector.py`: score the new labels, per scene and pooled, with the zero-shot model on the same scenes. The report keeps its provenance block.
4. Run all 10 folds and the evaluation. Record the result in the roadmap.
5. Only if the tests pass: train the final model on all 10 scenes, hash-pin it like the other weights (`perception/detector.py`), keep it out of the repo, and let fusion use the two new labels (weights already in `configs/fusion.yaml`).

## Consequences and risks
- **Overfitting to a few scenes.** The main risk. Leave-one-scene-out shows it honestly; it cannot remove it. Boston and Singapore only.
- **Weights carry the nuScenes terms.** Not committed, not shared; the demo may serve them (Motional's reply).
- **Fine-tuning can erase what was learned.** The old-class regression test and the outside-nuScenes check are there for this.
- **The final model is trained on all scenes,** so no honest score exists for it. It is judged by the fold results, which is the usual practice.
- **Small objects** (under 32 x 32 px) are almost never found by the zero-shot model; fine-tuning may not change that. We report by size.

## What the owner decides
Decided 2026-10-26: the scope (two new classes only) and the pass limits above (precision 70%, recall 40%, own cutoffs for the new classes). Still open: whether a thin pass should be shipped behind a flag. The recommendation is no flag: a detector that passed on two scenes should not change hazard scores for users.

## Updates
- 2026-10-26, time: training runs at about 24 images per second, so one fold (about 2,200 images, 12 epochs) takes about 22 minutes and the 10 folds about 3.7 hours, not "about an hour" as first written.
- 2026-10-26, new-class cutoffs: "chosen on the training scenes only" is done as leave-one-scene-out on the held-out detections. For each scene, the cutoff is the lowest score on the grid at which precision on the OTHER nine scenes' held-out detections reaches 70% (the model that scored each of those scenes never saw it). A scene's own detections never influence its own cutoff, and no model is scored on images it trained on. Choosing the cutoff from the training scenes themselves would have used detections from images the model had already fit.
- 2026-10-26, verdict data: the pass or fail uses all six cameras (more boxes, more scenes that reach 20 boxes). The front camera alone is reported next to it so the numbers can be compared with the earlier ones, and if the two disagree the report says so.
- 2026-10-26, counting rule, set before any held-out result was seen (found by an independent Codex review of the scoring code): a prediction that finds a real box that is too hidden or too small to be counted (visibility under 2 or under 32 x 32 px) is set aside, neither a hit nor a false alarm. Before, such a find counted as a false alarm, which pushed precision down for correct detections. The rule applies equally to the fine-tuned and the zero-shot detector. Also from that review: the scorer refuses a model whose record does not say it left out exactly that scene, was a smoke run or used other settings; a thin passing scene whose 90% range dips below a limit is flagged as a caveat even if the pooled range does not; and the tool's exit status needs every measured CPU speed to pass as well. The outside-nuScenes sanity check stays manual.
