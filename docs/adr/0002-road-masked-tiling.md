Purpose: ADR 0002: Road-masked overlapping tiling replaces the fixed 8x8 grid.

# ADR 0002: Road-masked overlapping tiling replaces the fixed 8x8 grid

Status: accepted

## Context
The source classifier trains on RDD2022 close-ups and uses a fixed 8x8 grid. On a 1600x900 nuScenes frame a cell is about 200x112 px and mostly sky, buildings and vehicles; defects are visible only on near-field road (a pothole 30 m away is a few pixels); RDD2022 has no non-road negatives; fixed cells are the wrong size at different distances. RDD2022 is the training data for the classifier, not a filter; road-masked tiling plus negatives is the fix for the wide-view mismatch. <!-- F:S-12 --> <!-- F:U-17 -->

## Decision
Crop to the drivable-road region (nuScenes map data or a road segmentation model; exact method undecided), tile only that region with overlap at native resolution (focus on the bottom third), upscale tiles to 380x380 for the classifier. Add non-road tiles from nuScenes frames as automatically labeled negatives (labeled by the road mask, no hand labeling). Measure first, fine-tune only if needed. <!-- F:R-11 -->

## Alternatives considered
- Keep the fixed 8x8 grid: wrong scale and perspective. Rejected.
- Use RDD2022 as a filter: not what it is. Rejected.

## Consequences
Needs a road mask for every frame. Far-field defects will still be missed, so the effective detection range must be stated. Measurement plan: (1) false-positive rate on non-road tiles before and after adding negatives; (2) macro-F1 on RDD2022 validation; (3) synthetic paste test detection rate versus distance; (4) tiling unit tests on fixtures.
Update 2026-10-23: the road mask comes from the comma10k U-Net (ADR 0013).
