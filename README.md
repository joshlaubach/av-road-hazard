Purpose: entry point for av-road-hazard, a system that fuses road-object detection and road-defect detection into per-frame hazard records, written in plain language with the math beside it.

# av-road-hazard

> **Status (2026-10-26): the core works and is being measured.** The object detector, road finder, crack-and-pothole finder, fusion step and evaluation tools run on real models. Search (RAG), the web UI and the hosted demo are not built yet. [docs/ROADMAP.md](docs/ROADMAP.md) is the live list of what is done and what comes next.

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

Its weakness: it only knows the 80 things in COCO. It has no word for traffic cones or barriers, which is about 20% of everything annotated in the nuScenes test images. A fine-tune that teaches it those two words is running now ([ADR 0015](docs/adr/0015-detector-finetune.md)).

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
> Here $m_k=0$ if the box is confidently off-road and $1$ otherwise, $w_k$ is the severity weight (0.6 for "unknown", which every defect has until severity is built), and $c_k$ is the classifier probability.
>
> **Worked example.** A pothole with $c_k=0.8$, on the road ($m_k=1$), severity unknown ($w_k=0.6$). With nobody near, $\pi_k=0$ and $s_k=0.6\cdot0.8\cdot0.5=0.24$. With a pedestrian ($v=1$, $c=0.9$) half an object height away, $r=0.9\,e^{-0.5}\approx0.55$, so $s_k=0.48\cdot(0.5+0.27)\approx0.37$. The same pothole scores 24 alone and 37 with a person beside it. The record stores $H$ in $[0,1]$ and the UI shows $\mathrm{round}(100H)$.

The score ranks urgency. It is not the probability of a crash.

## What we have measured so far

All numbers below come from our own evaluation code (`eval/`), with a 90% range where it counts something. A **90% range** is a Wilson interval: if we saw $k$ successes in $n$ tries with observed rate $\hat p=k/n$, the true rate is probably between
$$\frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}},\qquad z\approx1.645.$$
Few tries means a wide range, so a range is more honest than a single number.

| Test | What we did | Result |
|---|---|---|
| False alarms on driving photos | The classifier run over 404 nuScenes frames (no known damage) | 0.131 alerts per frame after hard-negative training, down from 0.198 |
| Synthetic fusion test | Pasted 215 real damage patches onto 30 road photos, ran the full pipeline | Damage found on the road: 54 of 170. By patch width: 5% under 64 px, 30% at 64-128 px, 60% at 128 px and up. Reported off-road: 0 of 45. Fusion's road answer matched the hand-drawn road on 52 of 54. **Verdict: fail** (large patches need 70%) |
| Real photos | 14 dashboard photos from South Africa (Stellenbosch pothole dataset, 44 marked potholes) | 10 of 44 found (23%) |
| Detector gap | Zero-shot detector on nuScenes | Cannot name barriers, cones or debris; stopped vehicles found 26% of the time |

The pattern: **large damage close to the camera is found; small, far damage mostly is not.** A pothole 30 meters away is only a few pixels wide. We report this as a limit, not a fixable detail.

## What is in progress

The detector fine-tune ([ADR 0015](docs/adr/0015-detector-finetune.md)) teaches the detector to name cones and barriers. Because nuScenes mini has only 10 scenes, we train 10 models, each holding out one scene, and grade each model only on the scene it never saw. Cones and barriers must be right at least 70% of the time and find at least 40% of the clearly visible ones, and the old classes must not get worse. If it fails, the original detector stays.

## Limits and open problems

- **Small data.** nuScenes mini has 10 scenes (about 400 front-camera frames). In it, night driving and the Singapore district Holland Village are the same scenes, so you cannot separate "night" from "that place". It is for checking the pipeline, not for sweeping conclusions.
- **Domain gap.** The classifier learned from close-up road photos, but a windshield frame is a wide view. Tiling only the road ([ADR 0002](docs/adr/0002-road-masked-tiling.md)) is our fix, and the numbers above show it is only partly working.
- **Label names.** The original `defect-detector` repo had the four damage codes mapped to the wrong names. The mapping above matches the RDD2022 paper and the annotation files, and `tests/defects/test_label_mapping.py` pins it.
- **No ground truth for fusion.** No dataset has both objects and road damage labeled. The paste test is the substitute and is easier than real damage, so its results are not real-world accuracy.
- **Severity is unknown** for every defect for now, so the severity weight is a constant 0.6.
- **Licenses.** nuScenes images and any model fine-tuned on them stay local and non-commercial. RDD2022 names no license in one source and CC BY-SA 4.0 in another, so we use the strictest reading, never use its US subset, and publish none of its images. See [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md).
- **Evaluation is mostly, not fully, automatic** ([ADR 0007](docs/adr/0007-no-hand-labeling-eval.md)). The two hand steps are the road outlines drawn on 30 photos for the paste test and one person's review of flagged tiles.

## Not built yet

- **RAG** (retrieval-augmented generation): search over saved records, for example "frames with a pothole near a pedestrian", answered by a language model using the retrieved records as its source.
- **Web UI** with the photo, boxes, a 0-100 gauge, plain-language explanations and an animation of the pipeline.
- **Hosted demo** (Hugging Face Spaces plus Qdrant Cloud free tier, about $20 a month at most).

By rule ([ADR 0011](docs/adr/0011-core-first-gate.md)) these stay at stub level until the core produces numbers worth reporting.

## Try it

This runs the stub API (fixed fake answers, no models needed):

```
pip install -r requirements-dev.lock && pip install --no-deps -e .
uvicorn api.main:app --port 8000      # then open http://localhost:8000/docs
python -m pytest
```

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
