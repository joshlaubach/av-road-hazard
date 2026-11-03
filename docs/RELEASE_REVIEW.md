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
- [ ] Model weights: the owner has confirmed in writing that hosting each one is allowed (detector and road model were fine-tuned on nuScenes; open item). Nothing in git.

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
