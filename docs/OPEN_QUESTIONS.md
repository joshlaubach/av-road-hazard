Purpose: every open question, unverified fact, assumption, and source conflict for av-road-hazard, with owner, blocking sprint, and a recommended default.

# Open questions

Every recommended default is a recommendation, not a decision. Owner is the repo owner (Joshua Laubach) unless stated.

## Blockers before agents start

| # | Question | Owner | Blocks | Recommendation (not a decision) |
|---|---|---|---|---|
| 1 | Push access to create the remote repo (`joshlaubach/av-road-hazard`, private) <!-- F:P-06 --> | Owner | Before Sprint 0 | Owner creates the remote and pushes `main`, then enables branch protection (PRs required, CI required once CI exists) |
| 2 | **[CRITICAL]** Which machine trains the models: Windows 11 + RTX 5080 (per source READMEs) or the macOS MacBook (linked dev computer)? UNCONFIRMED in the source docs. **Answered: Windows 11 + RTX 5080** (maintainer, setup session); README and ENVIRONMENT.md still say unconfirmed and need a follow-up edit. <!-- F:E-02 --> | Owner | Sprint 1 | Confirm; document both until then. Training and Docker GPU passthrough need an NVIDIA host |
| 3 | **[CRITICAL]** RDD2022 license (figshare record names none). UNVERIFIED <!-- F:L-02 --> | Owner | Sprint 5 (publishing), earlier for fixtures | No RDD2022 images published or committed until confirmed |
| 5 | Which of the owner's coding skills each agent loads (not yet enumerated) <!-- F:U-14 --> | Owner | Before Sprint 0 | Fill into briefs. Placeholders: frontend-design for F, dataviz for charts |
| 6 | Python version of the new repo <!-- F:E-11 --> | Owner | Sprint 0 | 3.12 (sources use 3.11 and 3.12). Applied in Sprint 0 as `requires-python >=3.12`; CI runs 3.12 and 3.13; Docker uses 3.12; lockfiles were compiled on 3.13 |
| 7 | Experiment tracker: MLflow vs W&B <!-- F:R-13 --> | Owner | Sprint 1 | MLflow (self-hostable, free; UNVERIFIED preference) |
| 8 | License for `LICENSE` | Owner | Before push | MIT (applied as the default; revisit if the owner prefers another) |
| 9 | Accept core-first gating (ADR 0011)? <!-- F:Q-03 --> | Owner | Before Sprint 2 | Closed 2026-10-26: accepted (ADR 0011). D, F and I stay at stub level until A, B, C and the synthetic fusion evaluation produce numbers |
| 10 | Hosted corpus source: RDD2022 samples (after license verification), the owner's own photos, or both | Owner | Before deploy | Open. Decide before the hosted deploy; RDD2022 samples need license verification first |

## Verification needed (UNVERIFIED facts)

Sprint 0 additions (the scaffold author had no Docker daemon, so none of these ran):

| Fact | Where | Blocks | Recommendation |
|---|---|---|---|
| `docker build` and `docker compose up` of the API image never ran; only the same install and `uvicorn` start were checked in a clean virtualenv. CI job `docker-and-e2e` is the first real test | Sprint 0 | Merge of the scaffold PR | Watch the first CI run; fix the Dockerfile if it fails |
| Compose image tags: `qdrant/qdrant:v1.12.0` unverified on a live pull; `ollama/ollama:latest` unpinned | E | Sprint 2 | Pin exact tags after a pull |
| GitHub rulesets are not enforced on a private repo without a paid plan; `main` protection relies on convention, the pre-commit hook, and CI | Owner | Before more agents start | Go public (after the history scrub) or upgrade |
| `hazard_score` upper bound and the pinned `box`/`image_size` conventions are the scaffold's reading of SPEC | C | Sprint 1 | Confirm in the scaffold review; record changes in CONTRACT_CHANGELOG.md |


| Fact | Where | Blocks | Recommendation |
|---|---|---|---|
| RDD2022 label names: source maps D00 pothole; the paper says D40 pothole <!-- F:S-11 --> | Workstream B | Sprint 1 (before porting) | Closed 2026-10-26: verified against the paper and the annotation box counts, pinned by `tests/defects/test_label_mapping.py` |
| nuScenes license terms (non-commercial, image redistribution) <!-- F:L-01 --> | Owner | Before any hosted text/metadata from nuScenes | Keep images local; show nothing nuScenes-derived on the hosted demo until confirmed |
| Whether nuScenes-derived records may be shown as text/metadata/charts hosted <!-- F:L-07 --> | Owner | Sprint 5 | Default: do not show |
| Docker base image exact tag `nvidia/cuda:12.8.x-cudnn-runtime-ubuntu22.04` <!-- F:E-07 --> | E | Sprint 2 | Pick an existing 12.8 tag matching the PyTorch wheel |
| 11B vision model VRAM 8-12 GB at 4-bit <!-- F:E-05 --> | E | Sprint 2 | Measure on the training machine |
| nuScenes full size (~300 GB+) <!-- F:E-10 --> | Owner | Before v1.0-full download | Check free disk first |
| v1.0-mini has about 400 keyframes, 10 scenes <!-- F:S-06 --> | A | Sprint 1 | Check on download |
| `joshua-laubach-portfolio` as link target for the finished project <!-- F:S-01 --> | Owner | Sprint 5 | Inspect later |
| Contents of tool folders in the MacBook home (`.codex`, `.ollama`, `.kaggle`, `.conda`, ...) <!-- F:E-01 --> | Owner | none | Observed names only |

## From SPEC.md section 10 (Appendix A)

| Item | Owner | Blocks | Recommendation |
|---|---|---|---|
| Confirm RDD2022 license terms (see 3 above) | Owner | Sprint 5 | See above |
| Hosted corpus source (see 10 above) | Owner | Before deploy | Open (10 above) |

## From BUILD_PLAN.md section 7 (Appendix B)

| Item | Status |
|---|---|
| Push access to create the repo | Open (1 above) |
| RDD2022 license check | Open (3 above) |
| Which of your coding skills each agent should load (fill into the briefs) | Open (5 above) |

## From ADR 0013 (road-mask model)

Status of every item is open. No due date has been set by the owner, so the Due column reads "Not set" rather than a date.

| # | Question | Owner | Due | Status | Recommendation (not a decision) |
|---|---|---|---|---|---|
| R1 | Legal review of the road-model licence chain (smp MIT, weights MIT, comma10k MIT; UNVERIFIED: timm licence, third-party image rights in comma10k) | Owner (counsel) | Not set | Open | Review before the road-mask PR is treated as final; no change to the pinned weights until then |
| R2 | Provenance and licence of the encoder's ImageNet pre-training (the model card does not say) | Owner | Not set | Open | Find the source of the pre-training weights; record the result in DATA_AND_LICENSES.md |
| R3 | Shoulder-edge crack loss (2 of 51 crack tiles lost at min_road 0.5) and mask dilation as a fix | Owner | Not set | Open | Measure dilation on the same 148 tiles before changing the mask |
| R4 | Set `min_road` from more human-annotated tiles (the current 0.4 is provisional, from 148 tiles) | Owner | Not set | Open | Annotate more tiles and re-measure drop and crack-loss rates before changing the default |
| R5 | Nordic snow robustness check (model trained mostly on US and highway frames) | Owner | Not set | Open | Test on Nordic snow frames before relying on the mask there |
| R6 | Wire the road mask into `api/pipeline.py` and `defects.infer.find_defects` | Owner | Not set | Done | Wired (PRs #31-#33) and the detector now exists (ADR 0014). Re-measuring the classifier effect on annotated data is still open |
| R7 | CI audits only `requirements.lock` (`pip-audit`), so the `ml` extra, now including `segmentation-models-pytorch`, `safetensors` and `huggingface_hub` with their transitive packages, is not audited for known vulnerabilities | Owner | Not set | Open | Run `pip-audit` on an environment installed with `.[ml]`, or lock and audit the extra; the weights are hash-pinned and loaded from safetensors, so the runtime risk is low |

## From ADR 0014 (object detector)

Status of every item is open. No due date has been set by the owner, so the Due column reads "Not set".

| # | Question | Owner | Due | Status | Recommendation (not a decision) |
|---|---|---|---|---|---|
| D1 | Licence of the COCO-trained torchvision weights (BSD-3 code; COCO training data terms not verified by counsel) | Owner (counsel) | Before any hosted release | Open | Review before the hosted demo; no change to the pinned weights until then |
| D2 | Fine-tuning on nuScenes: needs explicit licence approval (nuScenes is non-commercial and restricts redistribution) and a measured gap that justifies it | Owner | Not set | Open | Do not fine-tune until both hold; the zero-shot numbers are in ADR 0014 |
| D3 | Detection on Norwegian RDD2022 photos, where vehicles are masked as grey blobs | Owner | Not set | Open | Check a sample of Norwegian images before relying on object masking there |
| D4 | The score threshold 0.5 is provisional; nuScenes mini has 10 scenes and the held-out split has 2 | Owner | Not set | Open | Re-measure on more scenes (v1.0-full) before changing it; on this data 0.3 and 0.5 give the same tile defects |
| D5 | CPU speed on the real hosted machine (measured only with 24, 4 and 2 threads on one laptop CPU) | Owner | Before the hosted demo | Open | Run `tools/eval_detector.py --threads N` on the target host |

## Appendix summary for ADR 0013

| Item | Status |
|---|---|
| Road-mask licence chain legal review (R1) | Open |
| Encoder ImageNet pre-training provenance (R2) | Open |
| Shoulder-crack loss and mask dilation (R3) | Open |
| `min_road` from more annotated tiles (R4) | Open |
| Nordic snow robustness check (R5) | Open |
| Mask wiring into pipeline and `find_defects` (R6) | Open |
| Audit of the `ml` extra dependencies (R7) | Open |

## Assumptions made while writing the docs

| Assumption | Recommendation |
|---|---|
| Detector regression tolerance (S-05 baseline F1 0.534, AP@0.50 0.312) is not specified <!-- F:S-05 --> | The first run is in ADR 0014 (zero-shot, different ground-truth rules, so not like-for-like); set a small absolute band from it once the ground-truth rules are settled |
| Pass criteria for the synthetic fusion test (box IoU, distance tolerance) are not specified | Define in the G brief; tune in config only |
| Retrieval query templates and k values are not specified | Define in the G brief |
| Road mask method (nuScenes map data vs a road segmentation model) is undecided <!-- F:R-11 --> | Try nuScenes map data first for nuScenes; decide a model for uploads after measuring |
| `image_size` ordering in `HazardRecord` (the example is `[1600, 900]`, i.e. width, height) | Pin (width, height) in `fusion/schema.py` and add a test |
| `data/MANIFEST.json` and `checkpoints/MANIFEST.json` exact JSON shape | Define in Sprint 0 (BUILD_PLAN lists "data/checkpoint MANIFEST.json format") |
| Hosted location of the web UI | Decide in Sprint 5 |
| LLM used for hosted answers | Claude API (the HF Space has no GPU for Ollama); needs a spend cap |

## Conflicts between appendices and the Fact Register (both kept)

| Conflict | Resolution proposed |
|---|---|
| SPEC.md section 2 says "Weights: retrain both locally (RTX 5080)" and BUILD_PLAN says "launch on your machine", while E-02 says the training machine is UNCONFIRMED | Open question 2; the SPEC line is read as the intended plan, not confirmed |
| ENGINEERING.md section 1 names ADRs `docs/adr/NNN-title.md`; this repo uses four-digit names `NNNN-title.md` (Part 2) | Use four digits; update ENGINEERING.md wording if the owner agrees |
| SPEC.md section 4 says `data/` holds download scripts only, but `.gitignore` ignores `data/` as required | `.gitignore` ignores `data/*` and unignores `MANIFEST.json`, `*.py`, `*.md` (recommendation); confirm |
| Parallel-agent timeline (1-2 weeks) vs two plans with at most two concurrent agents | Resolved by sequenced sprints (ADR 0008) |
| SPEC.md section 6 says streams 3-6 can begin immediately against stubs; BUILD_PLAN sprint table sequences them over days 2-9 | The sprint table governs because of the two-agent limit |
| SPEC.md/BUILD_PLAN "Days 1-2" milestone vs sprint 0 "1 day" | Sprint table governs |
| BUILD_PLAN.md section 2 asks for `ENGINEERING.md` "committed as-is" in Sprint 0; it is already in the first commit | No action |
| BUILD_PLAN.md section 3 F says "your frontend/design skill" while U-14 says skills are unenumerated | Placeholder until the owner fills it in |

## Git identity

The machine already has a global git identity configured. The prompt says to use "Joshua Laubach" and the email in U-01 only if unset, so the commit uses the existing global identity. Owner to confirm or set a local identity for this repo before pushing.

## Quality/process items

- Tolerances for metric regression in `eval/baselines.json`.
- Which source repos are worth keeping: a candid review of the existing repos is a possible follow-up. <!-- F:Q-07 -->
- Whether branch protection settings include required reviews from the other tool (cannot be enforced by GitHub; process only).
