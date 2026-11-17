# REPORT-E: README honesty (branch fix/readme-honesty, commit 1873bdc)

## What changed (README.md only; lines 1-19, line 242 and the results table untouched)
1. New callout after the status block, before "How to read this page", headed "Read this before trusting any number on this page", with three bullets:
   - The zero-shot detector finds about 39% of pedestrians and 13% of cyclists (recall; nuScenes mini, six cameras, cutoff 0.5). The ROADMAP's other check (default model) gives 38% and 21% (cyclists from only 57 objects). Every figure is well under half.
   - The road-damage accuracy is scored on validation photos, the same ones used to pick the saved model. The 15% RDD2022 test split (4,745 images) has not been scored yet.
   - Of the 40 demo photos, 18 were in the training set and 2 in validation. Only 20 (11 outer fold, 9 test) were never seen while training or choosing the model.
2. One sentence added to the "Search on 40 road-damage photos" paragraph: 20 of the 40 photos were in the training or validation set, and the written descriptions come from that model, so description search may do better here than on new photos.

## Facts checked
- eval/results.json `detector.fine_tune.old_classes.*.zero_shot`: pedestrian recall 0.388 (1320/3402), cyclist 0.132 (24/182).
- tools/eval_classifier.py ~119 scores fold-0 validation; defects/train.py:450 keeps the lowest val_loss checkpoint on the same set; the test split holds 4,745 images.
- Demo photos matched against splits.json fold 0: 18 train, 2 val, 11 outer, 9 test.
- tools/eval_corpus_retrieval.py builds descriptions from the real pipeline's records.

## For the owner: hero kicker and tagline (not changed; baked into docs/images/hero.png)
- Kicker, replacing "Autonomous-vehicle pothole detection": "Research demo: road damage + objects from one dashcam photo" or "Portfolio project: road-hazard scoring from dashcam photos".
- Tagline, replacing "Spotting potholes before the car hits them": "Flagging potholes and who is near them, one photo at a time", "Which road damage to look at first, from a single dashcam photo", or "A research demo that ranks road damage by urgency".

## For the owner: hero image does not match the stored record (not regenerated)
- hero.png shows 50 / 100 and "nobody nearby".
- web/public/examples/stellenbosch-potholes/example.json: hazard_score 0.5112 (shown as 51) and 1 object, a "pedestrian" at score 0.583, box (379, 1184, 432, 1338) in a 3680 x 1760 image. It may be the bollard on the left (UNVERIFIED).
- Options: regenerate the image from the JSON, or change the caption.

## Skipped
Tagline and hero (owner), line 9 and line 242 (another task and owner), results table (generated and test-enforced), docs/ROADMAP.md (outside the brief).

## Tests
pytest (default): 1200 passed, 4 skipped. With `-o addopts=""` the Docker e2e test also runs and fails with a connection error because no stack is running; it is excluded by default.
