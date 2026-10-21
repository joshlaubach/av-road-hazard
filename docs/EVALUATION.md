Purpose: evaluation protocol for av-road-hazard: the three automatic evaluations, metrics, baselines, confounds, synthetic fusion design, retrieval query generation, and reporting format.

# Evaluation

No hand labeling. The owner explicitly does not want to label anything by hand, and the plan rejects hand-labeling about 100 frames. <!-- F:U-10 --> Rejected alternatives: hand-labeling, and only separate evaluation without a fusion check. Three automatic checks: per-model metrics, retrieval tests auto-generated from nuScenes annotations, and a synthetic fusion test pasting RDD2022 defect patches onto nuScenes/CARLA road regions at known positions. <!-- F:R-06 --> See [adr/0007-no-hand-labeling-eval.md](adr/0007-no-hand-labeling-eval.md).

## Why a synthetic fusion test

Fusion has no ground truth: nuScenes has no road-defect labels and RDD2022 has no object labels. The synthetic paste test is the substitute. Its limits must be stated wherever its numbers appear: pasted patches are easier than real defects (no perspective, lighting, or blending match), so pasted-patch results are not real-world accuracy. <!-- F:R-07 -->

## Evaluation 1: per-model

| Model | Metrics | Source of metric code |
|---|---|---|
| Detector | Precision, recall, F1 (micro, IoU-matched), COCO AP via `torchmetrics`, slices by day/night, traffic density, location, rain | Reuse `nuscenes-stability` metrics (`src/metrics.py`, `src/utils_nuscenes.py`) |
| Defect classifier | Macro-F1 on the RDD2022 validation split; per-class F1; false-positive rate on non-road tiles; macro-AUROC (not reported by the source) | New in `eval/models.py` |

Validation split: the official RDD2022 test split has no annotations, so metrics come from a deterministic 15% hold-out of train (as in the source). The detector split is by scene (deterministic 80/20 by sorted scene token), never by frame; tune only on validation; touch the test split once per release.

### Baselines to reproduce

| Baseline | Value | Tolerance |
|---|---|---|
| Detector v1.0-mini overall F1 | 0.534 | to be set (open question) |
| Detector v1.0-mini AP@0.50 | 0.312 | to be set |
| Detector v1.0-mini day-vs-night F1 gap | about 0.07 | informational |
| Classifier RDD2022 val macro-F1 | about 0.73 (Pothole 0.69, Longitudinal 0.65, Transverse 0.84, Alligator 0.73; label names unverified, see S-11) | to be set |
| Classifier train F1 | about 0.92 (overfit gap to address) | informational |
| Classifier over-firing | 31 of 64 patches on a sample frame at threshold 0.5 | goal: reduce, measure on non-road tiles |

Baselines live in `eval/baselines.json` (placeholder in Sprint 0). A metric regression fails if it falls below baseline minus tolerance. Reproduction is exact only on the same hardware and library versions (`ROIAlign` is non-deterministic).

## Evaluation 2: retrieval

Queries are auto-generated from nuScenes ground-truth annotations, for example "frames with a pedestrian at night". Answers are derived from the same annotations (category, scene conditions such as night, location). Score recall@k against the annotation-derived answers. Query templates and k values are defined in the G brief. nuScenes v1.0-mini is enough to build and test but too small to show retrieval quality; scale to v1.0-full (700 scenes) once the pipeline works. <!-- F:R-10 -->

## Evaluation 3: synthetic fusion

Design:

1. Take road regions of nuScenes and CARLA frames (road mask).
2. Paste RDD2022 defect patches at known positions, some inside the road, some outside, some near known objects.
3. Run the pipeline.
4. Verify against the known placement: detected defect box position, `in_road`, and `near_objects`.

Pass criteria and thresholds (box IoU, distance tolerance) are defined in the G brief and tuned in config only. Report separately: detection rate versus distance (this gives the effective detection range), and false-positive rate on non-road tiles. Use of RDD2022 patches is local; no pasted images are published until the RDD2022 license is verified.

## Known confounds to report honestly

In v1.0-mini, night and Singapore location are perfectly correlated, there are no rainy frames, and v1.0-mini is for pipeline validation. Evaluation on v1.0-full resolves this. Report confidence intervals or multiple seeds where cheap.

## Reporting format

- A single `eval/results.json` with all three evaluations.
- A markdown table generated from it for the README.
- Every README claim cites a number from `eval/results.json` or is marked unmeasured.
- Effective detection range and non-road false-positive rate are mandatory README entries.

## Test layers that touch evaluation

Metric regression (on model or data changes), retrieval eval (on RAG changes), synthetic fusion (on fusion or defects changes): see ENGINEERING.md section 3.
