Purpose: evaluation protocol for av-road-hazard: the three automatic evaluations, metrics, baselines, confounds, synthetic fusion design, retrieval query generation, and reporting format.

# Evaluation

No hand labeling: all evaluation is automatic, and the plan rejects hand-labeling about 100 frames. <!-- F:U-10 --> Rejected alternatives: hand-labeling, and only separate evaluation without a fusion check. Three automatic checks: per-model metrics, retrieval tests auto-generated from nuScenes annotations, and a synthetic fusion test pasting RDD2022 defect patches onto nuScenes road regions at known positions. <!-- F:R-06 --> See [adr/0007-no-hand-labeling-eval.md](adr/0007-no-hand-labeling-eval.md).

## Why a synthetic fusion test

Fusion has no ground truth: nuScenes has no road-defect labels and RDD2022 has no object labels. The synthetic paste test is the substitute. Its limits must be stated wherever its numbers appear: pasted patches are easier than real defects (no perspective, lighting, or blending match), so pasted-patch results are not real-world accuracy. <!-- F:R-07 -->

## Evaluation 1: per-model

| Model | Metrics | Source of metric code |
|---|---|---|
| Detector | Precision, recall, F1 (micro, IoU-matched), COCO AP via `torchmetrics`, slices by day/night, traffic density, location, rain | Reuse `nuscenes-stability` metrics (`src/metrics.py`, `src/utils_nuscenes.py`) |
| Defect classifier | Macro-F1 on the RDD2022 validation split; per-class F1; false-positive rate on non-road tiles; macro-AUROC (not reported by the source) | New in `eval/models.py` |

Validation: the headline classifier numbers are measured on the fold-0 validation tiles, the same split that chose the checkpoint, so they are not a hold-out. The RDD2022 test split (the `test` key of `splits.json`, 4,745 images) is never used for training, choosing the checkpoint or choosing thresholds; it is scored once, through `tools/eval_test_split.py` (see Test split (sealed) below). The detector split is by scene (deterministic 80/20 by sorted scene token), never by frame; tune only on validation.

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

### Eval notebooks

`notebooks/03_eval_classifier.ipynb` compares tile-classifier checkpoints (validation precision and recall at the shipped thresholds, false alarms on held-out confirmed-clean tiles, and the human review of the nuScenes alerts). `notebooks/04_eval_detector.ipynb` runs and displays `tools/eval_detector.py` (accuracy, slices, threshold sweep, CPU speed). They hold no logic: it lives in `eval/classifier.py`, `eval/detector.py` and the two `tools/eval_*.py` files, which have tests. They need the `notebook` extra (`pip install -e ".[notebook,ml]"`) and local data. Only these two notebooks are tracked, with outputs cleared, because outputs can embed nuScenes and RDD2022 images; `tests/notebooks/test_eval_notebooks.py` fails on outputs, machine-specific paths or function definitions. The other files in `notebooks/` stay git-ignored.

### Metric definitions (confirmed)

Implemented in `eval/models.py` as pure functions; the detector scene split, ground-truth filtering, thresholds and scoring are in `eval/detector.py` (used by `tools/eval_detector.py`). Synthetic-case tests in `tests/eval/`.

| What | Definition |
|---|---|
| Detector, "found" | Same label and IoU >= 0.5, matched greedily by descending score (same rule as the source repo, so the F1 0.534 and AP@0.50 0.312 baselines stay comparable). |
| Detector, numbers | Precision, recall, F1 at a score threshold (micro-averaged), AP@0.50 per class then averaged, and recall by ground-truth object-size bin. Bin cutoffs are arguments, set from the data. |
| AP method | COCO-style 101-point interpolation, reimplemented. It can differ slightly from `torchmetrics`, so check both on the same predictions before comparing with the 0.312 baseline. |
| Defect classifier | Scored per tile. Per-class precision, recall, F1, macro-F1, macro-AUROC, and the headline: recall at a fixed false-positive rate on non-road tiles (the cap is set after the baseline run). |
| Domain gap | On synthetic frames, detection rate against distance and tile size, which gives the effective detection range. |
| Video | Share of known pasted defects seen in at least one processed frame, looks per defect, processing time per clip. Not yet implemented. |
| Fusion | Box overlap with the pasted defect, `in_road` accuracy, `near_objects` correctness, and an ordering test: a pothole next to a pedestrian must score above the same pothole on an empty shoulder. Not yet implemented. |

Priorities: recall first for defect detection with the false-positive guardrail above; equal weight on precision and recall for perception.

### Test split (sealed)

The RDD2022 test split is scored once. `eval/test_split_protocol.json` is committed before any run and fixes the checkpoint hash (`tile-fold0-hn2`), the hashes of `splits.json` and `tile_index.json`, the shipped thresholds and the pooled sampling (seed 42, one clean tile per positive, at least one per image). Thresholds are read from the protocol, not from a config.

Run once, with the committed protocol and real paths:

```bash
python -m tools.eval_test_split --protocol eval/test_split_protocol.json --checkpoint PATH     --splits PATH --tile-index PATH --sample-index PATH --out eval/test_split_result.json
```

Guards, each a refusal with a nonzero exit before any image or model is loaded:

- the output file already exists (there is no `--force`: the result is sealed);
- the protocol is not tracked by git, has uncommitted changes, or the repository has uncommitted changes to tracked files;
- the SHA-256 of the checkpoint, `splits.json` or the tile index differs from the protocol.

It scores only the test images, in two measurements: `pooled` (the training pool, directly comparable to the validation row) and `all_tiles` (every tile not labelled ignore). Each class reports precision, recall, AUROC, flagged and positive counts, tile count, and 90% Wilson intervals for precision and recall. The result file holds numbers and hashes only. `tools/build_results.py --test-split PATH` adds it to `eval/results.json` and the README as one row.

Rule: run once. No checkpoint, threshold or config may be chosen or changed using these results.

## Evaluation 2: retrieval

Queries are auto-generated from nuScenes ground-truth annotations, for example "frames with a pedestrian at night". Answers are derived from the same annotations (category, scene conditions such as night, location). Score recall@k against the annotation-derived answers. Query templates and k values are defined in the G brief. nuScenes v1.0-mini is enough to build and test but too small to show retrieval quality; scale to v1.0-full (700 scenes) once the pipeline works. <!-- F:R-10 -->

## Evaluation 3: synthetic fusion

Design (nuScenes only):

1. Take road regions of nuScenes frames (road mask).
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
