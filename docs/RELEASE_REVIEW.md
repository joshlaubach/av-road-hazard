Purpose: the checklist for the release review that must happen before anything goes online (ROADMAP, Next row 2); written for a reviewer who has not seen the build.

# Release review

## What you are reviewing

One commit, and exactly the files `python -m tools.build_bundle` lists for it. That is what `docker/Dockerfile.space` copies into the image. Review that set and nothing else; anything that changes afterwards, including a fix for your own findings, needs a fresh look at that change.

1. Check out the commit the owner names. Run `python -m tools.build_bundle --out bundle_manifest.json`. It stops if a bundle file differs from the commit, so you know you are looking at the commit.
2. Read `bundle_manifest.json`: the commit, the number of files, and `bundle_sha256`. The owner must show you the same `bundle_sha256` for what is actually deployed.

## What the tool already checks (do not rely on it alone)

An image, weight or data file that is not on the allow list (only the two shrunk example photos are); a nuScenes file name; text that looks like an Anthropic, Hugging Face or AWS key, a private key, or a key assigned in code; a hardcoded `C:\Users\`, `/Users/` or `/home/` path. A pattern match is not proof of safety in either direction: read the code around anything it reports, and look for what a pattern cannot see.

## Checklist

**Contracts**
- [ ] `fusion/schema.py` and `openapi.json` match the last entry in `CONTRACT_CHANGELOG.md`. No contract change without a `contract-change` PR.
- [ ] `web/src/api/schema.d.ts` is generated from `openapi.json` (CI checks this; confirm it ran on this commit).

**Secrets**
- [ ] Every key and token is read from the environment (`common/config.py`), none is written in code, config or docs. The Space secrets list in `docs/DEPLOY.md` matches the names the code reads.
- [ ] Claude spend caps are set and the cutoff and fallback message work (`rag/` spend guard tests).

**Data and license boundaries**
- [ ] No nuScenes image, nuScenes-derived image or frame in the bundle, the image layers, the example data or the corpus. nuScenes-derived text, numbers and charts only if the owner has said so in writing.
- [ ] The photos shown are only the decided corpus: 40 non-US RDD2022 images and 20 Stellenbosch photos. `web/src/lib/demo_corpus.json` counts match what was indexed.
- [ ] `/credits` names every source, the demo says it is non-commercial, and the license status is stated as an assumption (`docs/DATA_AND_LICENSES.md`).
- [ ] Model weights: the hosted demo uses the public COCO detector, the public comma10k road model and public OpenCLIP, fetched from their sources, plus the owner's tile classifier from a private Hugging Face repo (SHA-256 pinned in `configs/hosted_classifier.json`). No nuScenes-fine-tuned weights are shipped: the 8-class detector stays off unless `DETECTOR_CHECKPOINT` is set. The license for each public model is an owner assumption (see `docs/DATA_AND_LICENSES.md`). No weight file in git.

**Abuse and privacy**
- [ ] Request size limits, upload limits, pixel limit and rate limits are on in the hosted configuration (`api/limits.py`, `configs/default.yaml`).
- [ ] An uploaded photo is decoded to pixels and dropped; no GPS or other EXIF reaches a record (tested in `tests/integration/test_request_limits.py`).
- [ ] One visitor's uploads cannot be searched by another; a client cannot write to the `corpus` session (`rag/index.py`).

**Honesty of claims**
- [ ] Every number in the README and on the site traces to `eval/results.json` or is marked unmeasured. The limits (small, faraway damage is mostly missed; shrunk photos miss damage) are stated.
- [ ] `/health` reports `ready` only for models that loaded.

**Build and run**
- [ ] The image builds from the commit; the container runs as a non-root user; killing the API stops the container (`docker/space/start.sh`).
- [ ] No hardcoded paths (the tool checks the obvious ones; read `tools/` and `configs/`).

## Writing it up

For each item: pass, fail or not checked, and why. A "not checked" is not a pass. Report findings with the file and line. The owner fixes them, and you look at the fix before the release.

## Cover note to send with the checklist (copy, fill in the three blanks)

Hi, I am about to put a small demo online and want a second pair of eyes first. You have not seen the code, and that is the point.

**What it is.** A web demo that takes a dashcam photo, finds cars, people and road damage with neural networks, and gives a 0 to 100 hazard score with a plain-language explanation. It also searches 60 demo photos. It is a non-commercial portfolio project, not a safety system. The README has a one-page overview.

**What I need.** Check the one commit below against `docs/RELEASE_REVIEW.md`, and tell me pass, fail or not checked for each item, with the file and line for any problem. About 1 to 2 hours. You do not need any keys, tokens or Space access, and you should not be given any.

- Repo (read access only): `joshlaubach/av-road-hazard`
- Commit: `__________`
- Bundle hash (`bundle_sha256` from `python -m tools.build_bundle`): `__________`
- Reply by: `__________`

**How to start.** Check out the commit, then run `python -m tools.build_bundle --out bundle_manifest.json`. It stops if anything differs from the commit, and it lists the files that are deployed and flags images, weights, key-looking text and machine paths. Its checks are pattern matches; please look for what a pattern cannot see.

**Things I already know are weak** (so you do not need to rediscover them, but do tell me if you disagree):
- The photo licenses (RDD2022 and Stellenbosch) and some model licenses (COCO weights, the road model's pre-training) rest on my assumption that non-commercial use is fine. They are not verified. See `docs/DATA_AND_LICENSES.md`.
- Whether the nuScenes dataset owner approved hosting a model fine-tuned on their data is unrecorded, so the demo does not use that model. Please confirm nothing nuScenes-derived is in the bundle.
- The Claude spend cap has been tested in unit tests only, not live. The demo runs without a Claude key by default.
- The server logs visitors' IP addresses (the host shows them). I have not decided how long to keep them.
- The tile classifier is fetched at start from a private Hugging Face repo with a read-only token, and its SHA-256 is pinned in `configs/hosted_classifier.json`.
- The Qdrant Cloud key can read and write the whole database. The code refuses uploads into the shared `corpus` session; please try to break that.

**What is out of scope.** Model accuracy (it is measured in `eval/results.json` and described honestly in the README), and the visual design.
