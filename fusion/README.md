Purpose: how fusion turns objects, raw defects and a road mask into a HazardRecord, including the hazard_score formula, its settings and its limits.

# Fusion

`fusion/fuse.py` builds one `HazardRecord` per frame from detector objects, raw defects and an optional road mask. It is pure (no model imports, no I/O) and deterministic. Settings live in `configs/fusion.yaml` and are validated by `fusion/config.py`.

## hazard_score

An explainable urgency ranking, **not** a probability of a collision.

For each defect:

```
object_relevance = vulnerability_weight * object_confidence * exp(-normalized_distance / distance_scale)
proximity        = max(object_relevance over ALL objects)          # 0 with no objects
defect_score     = road_multiplier * severity_weight * defect_confidence * (0.5 + 0.5 * proximity)
hazard_score     = max(defect_score over defects)                  # 0.0 with no defects
```

With weights and inputs in [0, 1] the score is in [0, 1].

| Piece | Rule |
|---|---|
| `road_multiplier` | `in_road` False -> 0, True -> 1, null (unknown) -> 1 |
| `severity_weight` | low 0.3, medium 0.6, high 1.0, unknown 0.6 (config). The level comes from the damage type: D00 and D10 low, D20 medium, D40 high ([ADR 0016](../docs/adr/0016-defect-merge-and-severity.md)) |
| Baseline `0.5` | An empty road can still hold a dangerous pothole; multiplying by proximity alone would score it 0 |
| Maximum, not sum | Duplicate detections or a crowd cannot inflate the score |
| `normalized_distance` | Pixel distance from the object's bottom-center to the nearest point of the defect box (0 if inside), divided by `max(object box height, min_object_height_px)` |

Worked examples (high severity, defect confidence 0.9): in road with no objects 0.45; in road beside a pedestrian of relevance 0.9 about 0.855; confirmed off-road 0.

## in_road

`true` when at least `road_overlap_threshold` of the defect box's in-image area lies on road pixels (pixels the box only partly overlaps count in proportion to the overlap), `false` when below it, `null` when there is no mask or the box has no in-image area. Unknown is scored like on-road so that uncertainty never hides a hazard; the record keeps `null` so the uncertainty stays visible.

**Readers must not test truthiness**: `null` is falsy. Use `in_road is not False` to keep unknown hazards and `in_road is True` only for confirmed on-road filtering. See `CONTRACT_CHANGELOG.md` v0.2.0.

## near_objects

A local report: objects within `near_radius` object heights of the defect, nearest first, with box IoU and pixel distance. The score uses all objects, so an object outside the radius can still contribute; `near_objects` is **not** a complete explanation of the score.

## Settings (`configs/fusion.yaml`)

All numeric settings must be finite and are validated whenever a `FusionConfig` is built (including `dataclasses.replace`); the weight mappings are read-only. Weights are in [0, 1]. `distance_scale` and `min_object_height_px` must be > 0, `near_radius` >= 0, `road_overlap_threshold` in [0, 1]. Severity weights must have exactly the keys low, medium, high, unknown. Vulnerability weight keys must match the labels the detector emits (not fixed until the detector lands); unlisted labels use `default_vulnerability`.

## Limits

- Pixel proximity is not physical proximity. Dividing by the object's own height is a perspective heuristic.
- "Walking toward" the defect is out of scope: the contract has no velocity, trajectory or lane assignment.
- Severity comes from the damage type alone (a judgment, not a measurement, listed in `defect_severity` in `configs/fusion.yaml`). A class not in the table is `unknown`. `fuse()` still takes a `severity_fn` that overrides the table.
- A defect may be a merged box of neighboring tiles, so it can be large and can include non-road; `in_road` is judged on the whole box.
- All weights are uncalibrated starting values.
- A truthful `in_road` needs a trustworthy road mask. Where the mask comes from is not decided.
