Purpose: entry point for av-road-hazard, a system that fuses road-object detection and road-defect detection into per-frame hazard records, written in plain language with the math beside it.

# av-road-hazard

> **Status (2026-10-27): the core is built and measured, with honest limits.** The object detector, road finder, crack-and-pothole finder, fusion step and evaluation tools run on real models, and every number below comes from `eval/results.json`. The pipeline finds large, near damage and mostly misses small, far damage. Search (RAG) and the hosted demo are not built yet; the web page for trying a photo is. [docs/ROADMAP.md](docs/ROADMAP.md) is the live list of what is done and what comes next.

New words are explained the first time they appear, in **bold**. Every term is also in [docs/GLOSSARY.md](docs/GLOSSARY.md). Math sits in short "The math" notes, so you can skip them and still follow the story.

## The idea in one minute

Imagine a passenger who watches the road through the windshield and warns the driver. A good passenger notices two things: *what is out there* (a person walking, a parked truck, a traffic cone) and *what is wrong with the road* (a pothole, a crack). A good passenger also knows that a pothole in the lane where someone is walking is more urgent than the same pothole on an empty shoulder.

This project is that passenger, as software. It takes one photo from a car's front camera and produces one **hazard record**: a small JSON file listing the objects, the road damage, whether each damage spot is on the road, which objects are near it, and one **hazard score** from 0 to 100.

It joins two earlier projects, `nuscenes-stability` (finds objects) and `defect-detector` (finds road damage), into one pipeline instead of two separate demos.

## How it works, step by step

```
photo ─► 1. objects      (detector)   ─► boxes: "pedestrian here, car there"
     ├─► 2. road         (road finder) ─► which pixels are road
     └─► 3. damage       (tile classifier, on road tiles only) ─► boxes: "crack here"
                              │
                     4. fusion (rules, no learning)
                              │
                       hazard record (JSON) ─► score 0-100
```

### 1. Objects: a **detector** draws boxes

A **detector** is a neural network that looks at a photo and returns **boxes** (rectangles) around things, each with a **label** ("car") and a **confidence** between 0 and 1. We use a pretrained COCO detector (a Faster R-CNN with a MobileNet backbone, [ADR 0014](docs/adr/0014-object-detector.md)) and keep boxes with confidence at least 0.5.

Its weakness: it only knows the 80 things in COCO. It has no word for traffic cones or barriers, which is about 20% of everything annotated in the nuScenes test images. We tried fine-tuning it on nuScenes to name those two things. Cones mostly worked, barriers did not, so the original detector stays ([ADR 0015](docs/adr/0015-detector-finetune.md)).

> **The math.** Two boxes $A$ and $B$ overlap by their **intersection over union**:
> $$\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}\in[0,1].$$
> A detection counts as correct when its label matches and its $\mathrm{IoU}$ with a true box is high enough.

### 2. Road: a **segmentation** model colors the road

**Segmentation** means labeling every pixel. Our road finder ([ADR 0013](docs/adr/0013-road-mask-model.md)) outputs a **road mask**: a picture the same size as the photo, where each pixel is 1 if road and 0 if not. Without it, we would call the damage on a building wall or a car roof a "pothole".

### 3. Damage: cut the road into **tiles**, ask a **classifier** about each

A **classifier** is a neural network that looks at one image and returns a probability for each category. We cut the photo into overlapping square **tiles** (384 pixels wide) and skip tiles that are less than 40% road or are mostly covered by an object. The remaining tiles go to an EfficientNet-B4 classifier trained on RDD2022 road-damage photos. For each tile it returns one probability per damage type:

| Code | Damage | Flag the tile when probability is at least |
|---|---|---|
| D00 | longitudinal crack (along the road) | 0.90 |
| D10 | transverse crack (across the road) | 0.75 |
| D20 | alligator crack (a web of cracks) | 0.65 |
| D40 | pothole | 0.75 |

The cutoffs differ per class and are provisional, not final. A **false alarm** (false positive) is a flag where nothing is wrong.

One long crack lights up several overlapping tiles, so flagged tiles of the same damage type that touch are merged into one defect: the box covers the group and the confidence is the strongest tile's ([ADR 0016](docs/adr/0016-defect-merge-and-severity.md)). On one Norwegian photo this turned 15 boxes into 2.

We also trained on **hard negatives**: tiles of ordinary road that the classifier wrongly flagged, shown again with the label "no damage". Repeating them 10 times cut false alarms on nuScenes photos from 80 to 53, with no class losing more than 3 points of recall (details in the Done table of [docs/ROADMAP.md](docs/ROADMAP.md)).

> **The math.** Flag a tile for damage type $c$ when its probability $p_c\ge\tau_c$, where $\tau_c$ is the cutoff in the table. Raising $\tau_c$ trades **recall** for **precision**:
> $$\text{precision}=\frac{TP}{TP+FP},\qquad \text{recall}=\frac{TP}{TP+FN},$$
> where $TP$ counts correct flags (true positives), $FP$ false alarms, and $FN$ misses (false negatives). Precision answers "when it says damage, how often is it right?". Recall answers "of the real damage, how much did it find?".

### 4. Fusion: plain rules that combine the answers

**Fusion** takes the objects, the damage tiles and the road mask and applies fixed rules (no learning, same input gives same output). For each damage box $k$:

1. **On the road?** If at least half of the box lies on the road mask, it is on the road. If the mask is missing, the answer is "unknown" and we treat it as on the road, so a possible hazard is never hidden.
2. **Which objects are near?** Distance is measured in object heights, so a person far away (small box) and a person close up (big box) are judged fairly. Objects within 3 object heights are listed in `near_objects`.
3. **Score.** Each object has a **vulnerability** $v$ (a person is 1.0, a car is 0.3, a cone is 0.5). These weights are guesses, not measured.

> **The math.** For object $j$ and damage box $k$, the distance and relevance are
> $$d_{jk}=\frac{\text{pixel gap between object bottom-center and damage box}}{\max(h_j,\,16)},\qquad r_{jk}=v_j\,c_j\,e^{-d_{jk}},$$
> with $h_j$ the object's box height in pixels and $c_j$ its detector confidence. The closeness of the most relevant object is $\pi_k=\max_j r_{jk}$ (the **maximum**, not the sum, so a crowd cannot inflate the score). The damage score and the frame's **hazard score** are
> $$s_k=m_k\,w_k\,c_k\,\Bigl(\tfrac12+\tfrac12\pi_k\Bigr),\qquad H=\min\Bigl(1,\ \max_k s_k\Bigr)\ \ (H=0\text{ with no damage}).$$
> Here $m_k=0$ if the box is confidently off-road and $1$ otherwise, $w_k$ is the severity weight (0.3 for a single crack, 0.6 for an alligator crack, 1.0 for a pothole; a first draft chosen by judgment, not measured), and $c_k$ is the classifier probability.
>
> **Worked example.** A pothole with $c_k=0.8$, on the road ($m_k=1$), severity high ($w_k=1.0$). With nobody near, $\pi_k=0$ and $s_k=1.0\cdot0.8\cdot0.5=0.40$. With a pedestrian ($v=1$, $c=0.9$) half an object height away, $r=0.9\,e^{-0.5}\approx0.55$, so $s_k=0.8\cdot(0.5+0.27)\approx0.62$. The same pothole scores 40 alone and 62 with a person beside it. The record stores $H$ in $[0,1]$ and the UI shows $\mathrm{round}(100H)$.

The score ranks urgency. It is not the probability of a crash.

## What we have measured so far

All numbers below come from our own evaluation code (`eval/`), with a 90% range where it counts something. A **90% range** is a Wilson interval: if we saw $k$ successes in $n$ tries with observed rate $\hat p=k/n$, the true rate is probably between
$$\frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}},\qquad z\approx1.645.$$
Few tries means a wide range, so a range is more honest than a single number.

<!-- results:start -->
| Measure | Result | In `eval/results.json` |
|---|---|---|
| Tile classifier on RDD2022 validation tiles, precision / recall per class | D00 95% / 57% · D10 85% / 61% · D20 88% / 72% · D40 92% / 57% | `classifier.validation` |
| False alarms on 404 nuScenes driving frames (alerts per frame) | baseline 0.198 (80 alerts), chosen 0.131 (53 alerts, 90% range 0.106 to 0.161) | `classifier.nuscenes` |
| Flagged nuScenes tiles that a reviewer called a real crack | baseline 6 of 79 (8%), chosen 7 of 52 (13%, 90% range 7% to 23%) | `classifier.nuscenes.*.buckets` |
| Clean road tiles the classifier still flags (23 held-out tiles) | baseline 23 of 23, chosen 13 of 23 | `classifier.probe` |
| Non-road tiles flagged (nuScenes, shipped thresholds) | 1 of 2,483 tiles with under 5% road (0.04%, 90% range 0.01% to 0.18%); 5 of 1,701 tiles with 5% to 40% road, which the pipeline skips (0.29%, 90% range 0.14% to 0.60%) | `classifier.non_road_false_positives` |
| Pasted-patch test: damage found on the road | 54 of 170 (32%, 90% range 26% to 38%); by patch width under 64 / 64 to 128 / 128 px and up: 5% / 30% / 60% | `synthetic_fusion.on_road` |
| Pasted-patch test: off-road patches reported as defects | 0 of 45 | `synthetic_fusion.off_road` |
| Pasted-patch test: `in_road` agrees with the hand-drawn road | 52 of 54 (96%, 90% range 89% to 99%) | `synthetic_fusion.in_road` |
| Pasted-patch test: objects near a defect | 71% of the true neighbors found, 88% of those reported correct | `synthetic_fusion.near_objects` |
| Pasted-patch test verdict | FAIL: detection rate, large patches: 0.60 is below the minimum 0.70 | `synthetic_fusion.verdict` |
| Effective range, from the pasted-patch test | defects 128 px wide or more in the frame (about 4.95 m ahead for a 0.5 m wide defect) are found 60% of the time; 64 to 128 px (between about 4.95 and 9.89 m) 30%; under 64 px (farther than about 9.89 m) 5% | `effective_range` |
| Zero-shot COCO detector on nuScenes (six cameras, cutoff 0.5), precision / recall | pedestrian 42% / 39% · cyclist 11% / 13% · motorcycle 37% / 13% · car 45% / 60% · truck 46% / 39% · bus 54% / 40% | `detector.fine_tune.old_classes` |
| Fine-tune for cones and barriers, held-out nuScenes scenes | cones: precision 74%, recall 58% over 4 scenes, but only 1 scene reaches both limits alone; barriers: 2 of 1511 found. FAIL: the zero-shot detector stays | `detector.fine_tune` |
| Real dashboard photos (14 dashboard photos, Stellenbosch pothole dataset, South Africa) | 10 of 44 marked potholes found; 15 other boxes (not necessarily mistakes). Run by hand, not reproducible from this repo | `real_photo_check` |
| Retrieval (RAG) | not measured | `retrieval` |
<!-- results:end -->

Rows marked *pasted-patch* come from a test that pastes real damage patches onto driving photos. It is easier than real damage, so those numbers are not real-world accuracy.

The pattern: **large damage close to the camera is found; small, far damage mostly is not.** A pothole 30 meters away is only a few pixels wide. We report this as a limit, not a fixable detail: the effective range row above says how large a defect must look to be found.

## What did not work

The detector fine-tune ([ADR 0015](docs/adr/0015-detector-finetune.md)) trained 10 models, each holding out one nuScenes scene, and graded each only on the scene it never saw. Cones were found and judged right often enough across scenes, but only one scene cleared the limits on its own, and barriers were almost never found in a new scene (they appear in only 3 of the 10 scenes). By the plan, the original detector stays. More scenes are the next step if work-zone objects matter.

## Limits and open problems

- **Small data.** nuScenes mini has 10 scenes (about 400 front-camera frames). In it, night driving and the Singapore district Holland Village are the same scenes, so you cannot separate "night" from "that place". It is for checking the pipeline, not for sweeping conclusions.
- **Domain gap.** The classifier learned from close-up road photos, but a windshield frame is a wide view. Tiling only the road ([ADR 0002](docs/adr/0002-road-masked-tiling.md)) is our fix, and the numbers above show it is only partly working.
- **Label names.** The original `defect-detector` repo had the four damage codes mapped to the wrong names. The mapping above matches the RDD2022 paper and the annotation files, and `tests/defects/test_label_mapping.py` pins it.
- **No ground truth for fusion.** No dataset has both objects and road damage labeled. The paste test is the substitute and is easier than real damage, so its results are not real-world accuracy.
- **Severity is a guess from the damage type** (single cracks low, alligator crack medium, pothole high). It ignores the size and depth of the damage, and the weights were chosen by judgment, not measured.
- **Licenses.** nuScenes images and any model fine-tuned on them stay local and non-commercial. RDD2022 names no license in one source and CC BY-SA 4.0 in another, so we use the strictest reading, never use its US subset, and publish none of its images. See [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md).
- **Evaluation is mostly, not fully, automatic** ([ADR 0007](docs/adr/0007-no-hand-labeling-eval.md)). The two hand steps are the road outlines drawn on 30 photos for the paste test and one person's review of flagged tiles.

## Not built yet

- **RAG** (retrieval-augmented generation): search over saved records, for example "frames with a pothole near a pedestrian", answered by a language model using the retrieved records as its source.
- **Web UI, the rest:** a Search view and a Settings view. The Analyze view (photo, boxes, a 0-100 gauge, plain-language explanations and an animated walkthrough) exists.
- **Hosted demo** (Hugging Face Spaces plus Qdrant Cloud free tier, about $20 a month at most).

By rule ([ADR 0011](docs/adr/0011-core-first-gate.md)) these stay at stub level until the core produces numbers worth reporting.

## Try it

This runs the stub API (fixed fake answers, no models needed):

```
pip install -r requirements-dev.lock && pip install --no-deps -e .
uvicorn api.main:app --port 8000      # then open http://localhost:8000/docs
python -m pytest
```

To use the web page (it needs Node 22 or newer):

```
cd web && npm ci && npm run dev      # then open http://localhost:3000, with the API running on port 8000
```

It shows the hazard score on a 0 to 100 gauge, explains each result in plain language, and can replay what the pipeline did to your photo (see [web/README.md](web/README.md)).

To run the real models on a photo, set `USE_STUBS=0` and the `DEFECT_CHECKPOINT` environment variable (the app reads real environment variables, not `.env`), then run `python -m tools.smoke_analyze`. All commands are in [AGENTS.md](AGENTS.md) section 4.

## Where to read more

| Document | What it is |
|---|---|
| [docs/ROADMAP.md](docs/ROADMAP.md) | What is done, what is next, what "production-ready" means |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Every term in one place |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Data flow, `HazardRecord`, API |
| [docs/EVALUATION.md](docs/EVALUATION.md) | The three evaluations, metrics, caveats |
| [docs/QUALITY_BAR.md](docs/QUALITY_BAR.md) | Evidence standards and the core-first gate |
| [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md) | Datasets, licenses, what may ship where |
| [docs/adr/](docs/adr/0000-template.md) | Decision records (why each choice was made) |
| [SPEC.md](SPEC.md), [BUILD_PLAN.md](BUILD_PLAN.md), [ENGINEERING.md](ENGINEERING.md) | Spec, build plan, engineering practices |
| [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md), [CONTRIBUTING.md](CONTRIBUTING.md) | Rules for coding agents and contributors |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md), [docs/RUNBOOK.md](docs/RUNBOOK.md) | Machines and versions; known failures |
| [docs/SOURCE_REPOS.md](docs/SOURCE_REPOS.md), [CONTRACT_CHANGELOG.md](CONTRACT_CHANGELOG.md) | The two source projects; interface changes |
| [docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md), [docs/TRACEABILITY.md](docs/TRACEABILITY.md), [docs/PROMPT_HISTORY.md](docs/PROMPT_HISTORY.md), [docs/AGENT_WORKFLOW.md](docs/AGENT_WORKFLOW.md), [docs/agents/](docs/agents/00-scaffold.md) | Project-management records |

Repo: `joshlaubach/av-road-hazard` (private). Audience: portfolio and recruiters.

## License

MIT (a recommendation recorded as the default, not a final owner decision). Third-party datasets have their own terms: see [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md).
