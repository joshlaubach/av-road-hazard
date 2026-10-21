Purpose: every open question, unverified fact, assumption, and source conflict for av-road-hazard, with owner, blocking sprint, and a recommended default.

# Open questions

Every recommended default is a recommendation, not a decision. Owner is the repo owner (Joshua Laubach) unless stated.

## Blockers before agents start

| # | Question | Owner | Blocks | Recommendation (not a decision) |
|---|---|---|---|---|
| 1 | Push access to create the remote repo (`joshlaubach/av-road-hazard`, private) <!-- F:P-06 --> | Owner | Before Sprint 0 | Owner creates the remote and pushes `main`, then enables branch protection (PRs required, CI required once CI exists) |
| 2 | **[CRITICAL]** Which machine trains the models: Windows 11 + RTX 5080 (per source READMEs) or the macOS MacBook (linked dev computer)? UNCONFIRMED <!-- F:E-02 --> | Owner | Sprint 1 | Confirm; document both until then. Training and Docker GPU passthrough need an NVIDIA host |
| 3 | **[CRITICAL]** RDD2022 license (figshare record names none). UNVERIFIED <!-- F:L-02 --> | Owner | Sprint 5 (publishing), earlier for fixtures | No RDD2022 images published or committed until confirmed |
| 4 | CARLA defect approach: decals/textures vs skipping CARLA defects in v1 (stock CARLA has no potholes or cracks) <!-- F:L-05 --> | Owner | Sprint 3 | Decide before G; if skipped, the CARLA corpus exercises only the object side |
| 5 | Which of the owner's coding skills each agent loads (not yet enumerated) <!-- F:U-14 --> | Owner | Before Sprint 0 | Fill into briefs. Placeholders: frontend-design for F, dataviz for charts |
| 6 | Python version of the new repo <!-- F:E-11 --> | Owner | Sprint 0 | 3.12 (sources use 3.11 and 3.12) |
| 7 | Experiment tracker: MLflow vs W&B <!-- F:R-13 --> | Owner | Sprint 1 | MLflow (self-hostable, free; UNVERIFIED preference) |
| 8 | License for `LICENSE` | Owner | Before push | MIT (applied as the default; revisit if the owner prefers another) |
| 9 | Accept core-first gating (ADR 0011)? <!-- F:Q-03 --> | Owner | Before Sprint 2 | Accept: gate D, F, I on A, B, C and the synthetic fusion evaluation producing numbers |

## Verification needed (UNVERIFIED facts)

| Fact | Where | Blocks | Recommendation |
|---|---|---|---|
| **[CRITICAL]** RDD2022 label names: source maps D00 pothole; common convention is D40 pothole. UNVERIFIED <!-- F:S-11 --> | Workstream B | Sprint 1 (before porting) | Verify against official docs and annotation XML; add a pinning test |
| nuScenes license terms (non-commercial, image redistribution) <!-- F:L-01 --> | Owner | Before any hosted text/metadata from nuScenes | Keep images local; show nothing nuScenes-derived on the hosted demo until confirmed |
| Whether nuScenes-derived records may be shown as text/metadata/charts hosted <!-- F:L-07 --> | Owner | Sprint 5 | Default: do not show |
| Waymo Open Dataset terms and approval <!-- F:L-03 --> | Owner | Optional, post-v1 | Excluded from v1 |
| CARLA license terms for generated frames <!-- F:L-05 --> | Owner | Sprint 3 | Check before publishing corpus |
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
| Decide how to add defects to CARLA (see 4 above) | Owner | Sprint 3 | See above |
| Whether drivenet-carla-av's driving policy joins later as a "drive-aware" extension (out of scope for v1) | Owner | Post-v1 | Defer <!-- F:U-04 --> |

## From BUILD_PLAN.md section 7 (Appendix B)

| Item | Status |
|---|---|
| Push access to create the repo | Open (1 above) |
| RDD2022 license check | Open (3 above) |
| CARLA defect approach (decals vs skip in v1) | Open (4 above) |
| Which of your coding skills each agent should load (fill into the briefs) | Open (5 above) |

## Assumptions made while writing the docs

| Assumption | Recommendation |
|---|---|
| Detector regression tolerance (S-05 baseline F1 0.534, AP@0.50 0.312) is not specified <!-- F:S-05 --> | Set after the first reproducible run; start with a small absolute band |
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

- Whether to run Waymo at all (optional, local, separate approval).
- Tolerances for metric regression in `eval/baselines.json`.
- Which source repos are worth keeping: a candid review of the existing repos is a possible follow-up. <!-- F:Q-07 -->
- Whether branch protection settings include required reviews from the other tool (cannot be enforced by GitHub; process only).
