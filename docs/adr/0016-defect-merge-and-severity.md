Purpose: ADR 0016: why neighboring flagged tiles are merged into one defect, and how a defect gets its severity.

# ADR 0016: Merging neighboring tiles, and severity from the damage type

Status: accepted (2026-10-26) as a first draft. The owner approved the severity table and large merged boxes.

## Context
The classifier scores overlapping square tiles (384 px), so one long crack flags several tiles and the record listed each tile as its own defect (one Norwegian frame gave 15). Every defect also had severity "unknown", which scores with a fixed weight of 0.6, so a hairline crack and a pothole ranked alike.

## Decision
1. **Merge.** In `find_defects` (not in the per-tile scorer, so the alert counts in the evaluation tools stay per tile), defects of the same class whose boxes overlap or touch become one defect, chained (A touches B and B touches C is one defect). The box is the smallest rectangle covering the group, the confidence is the strongest tile's (not an average), and the tile position is the strongest tile's. Different classes never merge. There is no size cap: merged boxes may be large, and a long crack gives a wide box.
2. **Severity from the damage type.** `configs/fusion.yaml` maps the class code to a severity: D00 and D10 (single cracks) low, D20 (alligator crack) medium, D40 (pothole) high. A code not in the table is "unknown". `fuse()` uses the table unless a severity function is passed in.
3. The record format does not change.

## Why
A pothole is a hole a wheel can drop into, alligator cracking means the surface is failing, and a single crack is the least urgent. The values are **a judgment, not a measurement**: nothing here is calibrated against crashes, repair cost or real severity labels, and the class is a coarse stand-in (a hairline pothole and a deep one both count as high). Severity from the picture itself (size, depth) is a later step.

## Consequences
- A low-severity crack now scores about half as much as before (weight 0.3 against 0.6 for unknown), and a pothole about 1.7 times as much (1.0 against 0.6). Hazard scores for the same photo change.
- A merged box can include non-road (grass, trees) and its `in_road` is judged on the whole box. If this ever marks a real hazard off-road, judge `in_road` per tile before merging. In the synthetic test it did not: `in_road` agreed on 52 of 54 found defects both before and after.
- That test pastes one patch per run, so it rarely exercises multi-tile merges: its found rate (54 of 170) and `near_objects` numbers did not move, and the median box overlap fell slightly (0.121 to 0.112) because merged boxes are larger. It cannot show that merging helps. The real-photo check can: the Norwegian frame went from 15 boxes to 2, and 17 defects over 20 local images became 4.
- Reported boxes are bigger, so "a box contains the patch center" is slightly easier to meet. The found rate did not change here, but a later change that merges more could flatter that number; read it with the box size.

## Limits
Severity ignores confidence, size and context. Chaining can join two separate defects through a flagged tile between them (not seen in the 20 images, so unmeasured). The table is one reviewer's draft.
