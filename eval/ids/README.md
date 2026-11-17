# eval/ids

Purpose: RDD2022 ID lists that the shipped classifier runs and the eval used, kept here so the numbers can be reproduced. They hold only RDD2022 sample IDs (like `Norway/train/annotations/xmls/Norway_005480`) and tile row/column numbers. No images, no paths, nothing from nuScenes.

The files are byte-for-byte copies of the originals (Windows line endings, no final newline). `.gitattributes` here marks them `-text` so git does not change the line endings, which would change the SHA-256. Do not reformat them.

| File | Source (git-ignored, local) | SHA-256 | Used by |
|---|---|---|---|
| `extra_negatives_batch3.json` | `data/rdd2022/derived/extra_negatives_batch3.json` | `5269526479c65963c3e21f2615497bf15fe65b65c2732740ddb6640181e823b8` | Training runs `tile-fold0-hn1` (repeat 20, 1360 extra tiles) and `tile-fold0-hn2` (repeat 10, 680 extra tiles); matches `extra_negatives_sha256` in both runs' `best.json` |
| `probe_negatives_batch3.json` | `data/rdd2022/derived/probe_negatives_batch3.json` | `0d83b44a245778b69c71ffc058a9b0f42a2874be1471109be2af82eee09422f6` | The eval's "23 held-out clean tiles" probe (`classifier.probe` in `eval/results.json`); matches `provenance.classifier.files.probe` there |

- `extra_negatives_batch3.json`: 68 hard-negative tiles, as `{"image_id", "row", "col"}`. These are clean tiles the classifier wrongly flagged; training adds them as extra "no defect" examples.
- `probe_negatives_batch3.json`: 23 clean tiles in the same format, held out from training and used only to count how many the classifier still flags.

Not here: `splits.json` (the train/val/test split, SHA-256 `ac18b6b881c9c9a187066243ac609be2c376c1168bea3dc845eec9d5a5b6e4b6`, used by both runs). At 7.9 MB it is over the repo's 1024 KB `check-added-large-files` limit, so it is left for the owner to decide.
