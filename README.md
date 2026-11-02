Purpose: entry point for av-road-hazard, a system that fuses road-object detection and road-defect detection into per-frame hazard records, written in plain language with the math beside it.

# av-road-hazard

![AV Road Hazard banner: a dashcam photo from Stellenbosch, South Africa, with a gold box around a pothole labeled 99.9% sure, and a half-circle hazard score gauge reading 50 out of 100 (pothole on the road, nobody nearby). Below, four steps for the search feature: photo, numbers, similar cases, answer.](docs/images/hero.png)

*Photo: Nienaber, Kroon and Booysen, Stellenbosch University (CC0).*

> **Status (2026-11-02): the core pipeline works and has been measured. The search feature is built but not shown to the public yet.**
>
> - **Built and measured:** the object finder, the road finder, the crack-and-pothole finder, the step that combines them, and the tools that grade them. They run on real models, and every number below comes from `eval/results.json`.
> - **Built, with a web page for trying a photo:** the "Analyze" page (upload a photo, see boxes and a 0 to 100 gauge).
> - **Built and tested, but only on 404 test photos:** search. You can ask "which photos have a pothole near a person?" and get an answer that points to real photos. A check in the code rejects any answer that cites a photo that was not found.
> - **Not built yet:** the Search and Settings pages in the web UI, running the Docker setup for real, and the hosted public demo.
> - **The honest limit:** the pipeline finds large, close damage and mostly misses small, far-away damage.
>
> [docs/ROADMAP.md](docs/ROADMAP.md) is the live list of what is done and what comes next.

**How to read this page.** New words are explained the first time they appear, in **bold**. Every term is also in [docs/GLOSSARY.md](docs/GLOSSARY.md). The math sits in short "The math" boxes. You can skip them and still follow the story.

## The idea in one minute

Picture a passenger who watches the road through the windshield and warns the driver. A good passenger notices two things:

- **What is out there:** a person walking, a parked truck, a traffic cone.
- **What is wrong with the road:** a pothole, a crack.

A good passenger also knows that a pothole in the lane where someone is walking is more urgent than the same pothole on an empty shoulder.

This project is that passenger, written as software. You give it one photo from a car's front camera. It gives back one **hazard record**. That is a small file that lists the objects, the road damage, whether each damage spot is really on the road, which objects are close to it, and one **hazard score** from 0 to 100.

It joins two earlier projects, `nuscenes-stability` (finds objects) and `defect-detector` (finds road damage), into one pipeline instead of two separate demos.

**What the score means.** A higher score means "look at this sooner". It ranks urgency. It is not the chance of a crash, and it is not a safety certificate. This project is a research and portfolio demo. It does not run on live video, and a self-driving car would need much more work before it could rely on it.

## How it works, step by step

Four steps. Steps 1 to 3 each use a neural network (a program that learned from many labeled examples). Step 4 is a short list of fixed rules.

![A two-lane diagram. Lane A: a photo goes to an object detector, a road finder and a damage classifier, fixed fusion rules combine them into a hazard record with a 0 to 100 hazard score. Lane B: saved records are embedded, found in a database, turned into an answer by a language model, and checked so the answer only cites photos that were found.](docs/images/pipeline.svg)

The picture shows the whole project: lane A is the four steps below, and lane B is the Search section further down. The text version of lane A:

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

A **detector** is a neural network that looks at a photo and draws **boxes** (rectangles) around things. Each box gets a **label** ("car") and a **confidence**, a number from 0 to 1 that says how sure the network is. We throw away boxes below 0.5, because they are too likely to be mistakes.

Our detector is a standard one called Faster R-CNN, trained by others on a public set of everyday photos called COCO ([ADR 0014](docs/adr/0014-object-detector.md)). We use it as it comes, which people call **zero-shot**: we did not teach it anything about roads.

Its weakness is that it only knows the 80 things in COCO. It has no word for traffic cones or barriers. Those make up about 20% of everything marked in the nuScenes test photos, and they matter on a road.

So we tried teaching it. We **fine-tuned** it, meaning we kept training it on nuScenes photos where cones and barriers were marked. The result was mixed ([ADR 0015](docs/adr/0015-detector-finetune.md)):

- Cones were found well enough in most scenes.
- Barriers were hard. In new scenes the model almost never found them.
- The project owner decided to ship the fine-tuned model anyway, but **switched off by default**. The pipeline uses the plain detector unless you set the path to the fine-tuned weights file. A wrong or missing file stops the program with an error instead of quietly using the other model.
- The weights stay on this machine because the nuScenes license does not allow sharing them.

> **The math.** To decide whether a box is right, we measure how much two boxes overlap. For boxes $A$ and $B$, the **intersection over union** is
> $$\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}\in[0,1].$$
> That is the area they share divided by the area they cover together. 0 means they do not touch and 1 means they are identical. A detection counts as correct when its label matches and its $\mathrm{IoU}$ with a true box is high enough.

### 2. Road: a **segmentation** model colors the road

**Segmentation** means giving every single pixel a label. Our road finder ([ADR 0013](docs/adr/0013-road-mask-model.md)) makes a **road mask**. That is a black-and-white copy of the photo where white means "road" and black means "not road". Without it, the pipeline would call a stain on a building wall or a car roof a "pothole".

### 3. Damage: cut the road into **tiles** and ask a **classifier** about each

A **classifier** is a neural network that looks at one small image and says how likely each category is. We cannot show it the whole photo, because the damage is tiny compared with the full view. So we cut the photo into overlapping squares called **tiles**, each 384 pixels wide, and we only look at tiles that matter:

- We skip any tile that is less than 40% road.
- We skip any tile that is mostly covered by an object, such as a car.

The remaining tiles go to a classifier called EfficientNet-B4, which learned from a public set of road-damage photos called RDD2022. For each tile it gives one probability per damage type. We flag the tile when the probability passes a cutoff:

| Code | Damage | Flag the tile when probability is at least |
|---|---|---|
| D00 | longitudinal crack (runs along the road) | 0.90 |
| D10 | transverse crack (runs across the road) | 0.75 |
| D20 | alligator crack (a web of cracks, like reptile skin) | 0.65 |
| D40 | pothole | 0.75 |

The cutoffs differ per type, and they are provisional, not final. If a cutoff is too low, the classifier cries wolf. If it is too high, it misses real damage. A flag where nothing is wrong is a **false alarm** (also called a false positive).

**Merging.** One long crack lights up several overlapping tiles. So flagged tiles of the same type that touch are merged into one defect. The merged box covers the whole group, and its confidence is the strongest tile's ([ADR 0016](docs/adr/0016-defect-merge-and-severity.md)). On one Norwegian photo, this turned 15 boxes into 2.

**Teaching it what is not damage.** We also trained on **hard negatives**. These are tiles of ordinary road that the classifier wrongly flagged. We showed them to it again with the label "no damage", repeating each one 10 times. That cut false alarms on nuScenes driving photos from 80 to 53, and no damage type lost more than 3 points of recall. (**Recall** is the share of real damage that gets found. The next box explains it.) Details are in the Done table of [docs/ROADMAP.md](docs/ROADMAP.md).

> **The math.** Flag a tile for damage type $c$ when its probability $p_c\ge\tau_c$, where $\tau_c$ is the cutoff in the table. Moving $\tau_c$ trades two things against each other:
> $$\text{precision}=\frac{TP}{TP+FP},\qquad \text{recall}=\frac{TP}{TP+FN}.$$
> Here $TP$ counts correct flags (true positives), $FP$ counts false alarms, and $FN$ counts misses (false negatives). **Precision** answers "when it says damage, how often is it right?". **Recall** answers "of the real damage, how much did it find?". A higher cutoff raises precision and lowers recall. For this project, a miss is worse than a false alarm, so we only accept a new model if recall drops by no more than 3 points for every damage type.

### 4. Fusion: fixed rules that combine the answers

**Fusion** takes the objects, the damage tiles and the road mask and applies fixed rules. It does no learning, and the same input always gives the same output. For each damage box $k$, it does three things:

1. **Is it on the road?** If at least half of the box lies on the road mask, it is on the road. If the road mask is missing, the answer is "unknown", and we treat it as on the road. That way a possible hazard is never hidden.
2. **Which objects are close?** We measure distance in "object heights". A person far away has a small box and a person nearby has a big one, so measuring in heights treats both fairly. Objects within 3 heights are listed in `near_objects`.
3. **How urgent is it?** Each object type gets a **vulnerability** number $v$ that says how much harm a hazard could do to it. A person is 1.0, a car is 0.3 and a cone is 0.5. These weights are educated guesses, not measured.

> **The math.** For object $j$ and damage box $k$, the distance and the relevance are
> $$d_{jk}=\frac{\text{pixel gap between object bottom-center and damage box}}{\max(h_j,\,16)},\qquad r_{jk}=v_j\,c_j\,e^{-d_{jk}},$$
> where $h_j$ is the object's box height in pixels and $c_j$ is the detector's confidence. The $e^{-d}$ part makes relevance fade quickly as the object gets farther away. The closeness of the most relevant object is $\pi_k=\max_j r_{jk}$. We take the **maximum**, not the sum, so a crowd cannot inflate the score. The damage score and the frame's **hazard score** are
> $$s_k=m_k\,w_k\,c_k\,\Bigl(\tfrac12+\tfrac12\pi_k\Bigr),\qquad H=\min\Bigl(1,\ \max_k s_k\Bigr)\ \ (H=0\text{ with no damage}).$$
> Here $m_k=0$ if the box is confidently off the road and $1$ otherwise. $w_k$ is the **severity weight**: 0.3 for a single crack, 0.6 for an alligator crack and 1.0 for a pothole. These are a first draft chosen by judgment, not measured. $c_k$ is the classifier's probability.
>
> **Worked example.** Take a pothole with $c_k=0.8$ on the road ($m_k=1$) and severity high ($w_k=1.0$). With nobody near, $\pi_k=0$, so $s_k=1.0\cdot0.8\cdot0.5=0.40$. Now put a pedestrian ($v=1$, $c=0.9$) half an object height away. Then $r=0.9\,e^{-0.5}\approx0.55$, so $s_k=0.8\cdot(0.5+0.27)\approx0.62$. The same pothole scores 40 alone and 62 with a person beside it. The record stores $H$ between 0 and 1, and the web page shows $\mathrm{round}(100H)$.

Because the frame's score is the **maximum** over all damage, one dangerous pothole outweighs ten harmless cracks. That was a deliberate choice.

The score ranks urgency. It is not the probability of a crash.

## Search: ask questions about saved photos

Each analyzed photo becomes a record. The search part lets you ask questions across many records. It works in four steps ([ADR 0017](docs/adr/0017-search-core.md), [ADR 0018](docs/adr/0018-grounded-answers.md), [ADR 0019](docs/adr/0019-search-api.md)):

1. **Embed.** An **embedding** turns a photo, or a sentence, into a list of numbers so that similar things get similar numbers. We use a model called OpenCLIP, which can do this for both photos and text.
2. **Find.** A database called Qdrant stores the lists of numbers and finds the closest ones. A search can use the photo's numbers, the description's numbers, or both fused together, plus filters such as "only frames with a pothole".
3. **Answer.** A language model (Claude, or a model running on your own computer) reads only the question and the records that were found. It must cite photo ids in brackets.
4. **Check.** The code checks every cited id against the records that were found. A made-up id is reported and hidden. If nothing was found, the language model is not called at all.

This is called **grounded** answering: the answer may only use what search found. Each visitor has a private session, so one visitor can never see another visitor's uploaded photos. The measured results are in the table below. In short, among the top 5 results, search finds about half of the photos it should (0.54 when text and photo are fused; random order finds 0.18), and the written answers cite exactly the right photos on most of the questions that have an answer. The questions come from the nuScenes labels, so the match against how real people phrase questions is untested.

## What we have measured so far

All numbers below come from our own evaluation code (`eval/`). Where a number counts something, it comes with a 90% range. A **90% range** (a Wilson interval) says: "if we saw $k$ successes in $n$ tries, the true rate is probably between these two values." Few tries make a wide range, so a range is more honest than a single number.

> **The math.** With $k$ successes in $n$ tries and observed rate $\hat p=k/n$, the range is
> $$\frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}},\qquad z\approx1.645.$$
> The search rows are different. Each question gets a score between 0 and 1, not a pass or fail, so their range comes from re-drawing the set of questions many times (resampling).

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
| Retrieval (RAG) | Recall at 5 on 404 nuScenes frames and 21 queries made from the annotations: description 0.41, photo 0.43, both fused 0.54, random order 0.18 (1.0 is perfect) | `retrieval` |
| Written answers (RAG) | 36 questions, top 5 photos given; claude-haiku-5-5 alone: 22 of 23 answerable questions cited exactly the right photos, $0.00017 each; claude-sonnet-5-5 alone: 20 of 23 answerable questions cited exactly the right photos, $0.00338 each; claude-haiku-5-5 then claude-sonnet-5-5: 22 of 23 answerable questions cited exactly the right photos, $0.00017 each; qwen2.5:14b (local): 23 of 23 answerable questions cited exactly the right photos, $0.00000 each. Checked against the records the model was given, not the photos | `answers` |
<!-- results:end -->

Rows marked *pasted-patch* come from a test that pastes real damage patches onto driving photos, because no dataset has both objects and road damage labeled. It is easier than real damage, so those numbers are not real-world accuracy.

The pattern: **large damage close to the camera is found, and small, far damage mostly is not.** A pothole 30 meters away is only a few pixels wide. We report this as a limit, not a detail we can easily fix. The "effective range" row above says how large a defect must look to be found.

## What did not work

- **The first detector fine-tune.** We trained 10 models, each holding out one nuScenes scene, and graded each only on the scene it never saw. Cones were found and judged right often enough across scenes. But only one scene cleared the limits on its own, and barriers were almost never found in a new scene (they appear in only 3 of the 10 scenes). By the plan, the plain detector stayed. A second round with 66 scenes passed the automatic checks, and the owner then chose to ship the result switched off by default ([ADR 0015](docs/adr/0015-detector-finetune.md)).
- **The pasted-patch test failed one of its four checks.** Large patches were found 60% of the time, and the minimum was 70%.
- **Real dashboard photos.** On 14 photos from a South African pothole dataset, the pipeline found 10 of 44 marked potholes. It did best when the damage was large in the frame.

## Limits and open problems

- **Small data.** nuScenes mini has 10 scenes (about 400 front-camera frames). In it, night driving and the Singapore district Holland Village are the same scenes, so you cannot tell "night" apart from "that place". It is for checking that the pipeline works, not for sweeping conclusions.
- **Domain gap.** The classifier learned from close-up road photos, but a windshield frame is a wide view. Tiling only the road ([ADR 0002](docs/adr/0002-road-masked-tiling.md)) is our fix, and the numbers above show it is only partly working.
- **Painted markings.** The most common false alarm is painted road markings (crosswalks, lane dashes and arrows) that the classifier mistakes for transverse cracks. We kept the cutoff for that class where it is, because only two real cracks were available to tune it, which is too few.
- **Label names.** The original `defect-detector` repo had the four damage codes mapped to the wrong names. The mapping above matches the RDD2022 paper and the annotation files, and `tests/defects/test_label_mapping.py` pins it.
- **No ground truth for fusion.** No dataset has both objects and road damage labeled. The paste test is the substitute, and it is easier than real damage.
- **Severity is a guess from the damage type** (single cracks low, alligator crack medium, pothole high). It ignores the size and depth of the damage, and the weights were chosen by judgment, not measured.
- **Photos, not video.** One photo at a time. There is no tracking over time and no live camera input.
- **Licenses.** nuScenes images and any model trained on them stay on this machine and are non-commercial. RDD2022 names no license in one source and CC BY-SA 4.0 in another, so we use the strictest reading, never use its US subset, and publish none of its images. See [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md).
- **Evaluation is mostly, not fully, automatic** ([ADR 0007](docs/adr/0007-no-hand-labeling-eval.md)). The two hand steps are the road outlines drawn on 30 photos for the paste test and one person's review of flagged tiles.

## Not built yet

- **Web UI, the rest.** A Search page and a Settings page. The Analyze page (photo, boxes, a 0 to 100 gauge, plain-language explanations and an animated walkthrough) exists.
- **Docker for real.** The Docker files have never been run with the real models, so treat them as a sketch.
- **Hardening for the public.** Request size limits and a security review of the exact bundle that will go online.
- **Hosted demo.** Hugging Face Spaces plus the free tier of Qdrant Cloud, about $20 a month at most. A set of 40 non-US RDD2022 photos and 20 CC0 pothole photos from Stellenbosch is ready for it. The RDD2022 photos stay on this machine until their license is verified.

## Try it

This runs the stub API, which gives fixed fake answers and needs no models:

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
