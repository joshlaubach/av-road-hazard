Purpose: step-by-step instructions for the outside reviewer of the release bundle; send with the cover note at the end of RELEASE_REVIEW.md.

# Release review: instructions

Thanks for doing this. You are checking one commit of a small demo before it goes online. You do not need to understand the machine learning. You are looking for things that would be embarrassing, unsafe or not allowed once the site is public. The checklist itself is [RELEASE_REVIEW.md](RELEASE_REVIEW.md); this page says how to work through it. Plan on 1 to 2 hours.

## Ground rules

- Review only the files in the bundle (step 2). Anything else is out of scope.
- You will get no keys, tokens or Space access. If you find one anywhere, report it as a finding.
- Give each checklist item one answer: **pass**, **fail** or **not checked**, with the reason. "Not checked" does not count as a pass.
- For every problem, give the file and line.

## Second round: re-checking the fixes

If you reviewed an earlier commit (`05e892a`), start here and then do only the parts of step 3 the fixes touch. Review the diff since `05e892a`, plus the evidence the owner sends. Each fix has a test that failed before it.

| Your earlier finding | What changed (PR) | What to re-check |
|---|---|---|
| Item 4, spend cap: a second model call could start after the first hit the cap | The budget is checked before every paid call (#131) | `rag/llm.py` `ClaudeLLM.generate`; the spend count is still in memory and resets on restart, which is now stated in `docs/DEPLOY.md` |
| Item 6, exact corpus: 61 records after importing 60 | The import replaces and reads back the corpus; duplicate names refused; a manifest of photo hashes (#135) | `tools/index_corpus.py`, `configs/corpus_manifest.json`, the owner's read-back of the live corpus |
| Item 9, abuse limits: a refused request evicted other visitors | Limits run before any session is created (#130) | `api/main.py` `caller()` |
| Item 9, forged `X-Forwarded-For` | Tested on the private Space: a forged value is ignored (#133). No code change. | `docs/adr/0019-search-api.md`, `tests/unit/test_client_address.py`; open question: whether each visitor has a separate count |
| Retention ("this visit only") | 2-hour cookie, honest text, visitor photos cleared at startup (#132) | `api/session.py`, `rag/index.py` `delete_visitor_points`, `web/src/components/SearchView.tsx` |
| README promises | Removed the bring-your-own-key claim; hash-check claim narrowed (#131) | `README.md` hosting section |
| `parallel_jobs` only covered answers | Slots cover analysis and question embedding; uploads run off the event loop; analyses still run one at a time (#134) | `api/main.py`, `rag/service.py` |

Items 5 and 16 (nuScenes imagery in built layers, Docker build and shutdown) need the owner's evidence: a build and run result for `docker/Dockerfile.space` with the user id and the API-kill result, and the live corpus read-back.

## 1. Set up (about 10 minutes)

1. Clone the repository (read access) and check out the commit the owner gave you.
2. Read `README.md` for the one-page overview, then skim `docs/RELEASE_REVIEW.md`.
3. Create a Python environment and run `pip install -r requirements-dev.lock`.

## 2. Confirm what you are looking at (about 5 minutes)

```
python -m tools.build_bundle --out bundle_manifest.json
```

- If it stops with an error, a bundle file differs from the commit. Tell the owner and stop.
- Open `bundle_manifest.json`. The commit must match, and `bundle_sha256` must equal the hash the owner gave you. If it differs, you are reviewing something other than what will be deployed. Stop and tell the owner.
- The tool already flags unlisted images or weights, nuScenes file names, key-looking text and hardcoded machine paths. It only matches patterns, so treat it as a starting point.

## 3. What to look for (about 1 hour)

**Contracts**
- Compare `fusion/schema.py` and `openapi.json` with the newest entry in `CONTRACT_CHANGELOG.md`. Nothing should differ.
- Check that CI passed on this commit, including the job that confirms the web client matches `openapi.json`.

**Secrets**
- Search the bundle for anything that looks like a key or token. Read every environment variable the code uses in `common/config.py` and compare with the Space secrets listed in `docs/DEPLOY.md`. The names should match.
- Find the Claude spend cap in `rag/`, read its tests, and confirm that reaching the cap leads to the fixed "answer unavailable" notice.

**Data and license boundaries** (the most important part)
- Confirm there is no nuScenes image, or image derived from one, in the bundle, the Docker layers, the example data or the demo corpus.
- Open `web/src/lib/demo_corpus.json`. It should list 40 non-US RDD2022 images and 20 Stellenbosch photos. Check that `/credits` names every source and says the licenses are an assumption.
- Model weights: the hosted demo should use three public models (a COCO detector, the comma10k road model, OpenCLIP) fetched from their public sources. The only model the owner hosts is the tile classifier, in a private Hugging Face repo. Confirm that no weight file is in git and that no nuScenes-trained model is used by default (the 8-class detector must stay off unless `DETECTOR_CHECKPOINT` is set).

**Abuse and privacy**
- Read `api/limits.py` and `configs/default.yaml`. Confirm the request size limits, the 10 MiB upload limit, the pixel limit and the rate limits are on in the hosted setup.
- Open `tests/integration/test_request_limits.py` and confirm it checks that GPS data in an uploaded JPEG never reaches a record.
- Check that one visitor's uploads cannot be searched by another, and that no client can write into the shared `corpus` session (`rag/index.py`). The database key can read and write everything, so please try hard to break the `corpus` protection.

**Honesty of claims**
- Pick 5 to 10 numbers from the README or the website and trace each to `eval/results.json`. Each should trace there or be marked unmeasured.
- Confirm the known limits are stated: small, faraway damage is mostly missed, and shrunk photos miss damage.
- Confirm `/health` reports a model as `ready` only if it loaded.

**Build and run**
- Check that `docker/Dockerfile.space` builds the image, the container runs as a non-root user, and `docker/space/start.sh` stops the container if the API dies.
- Read `tools/` and `configs/` for hardcoded paths.

## 4. Things the owner already knows are weak

You do not need to rediscover these, but say so if you disagree.

- The RDD2022 and Stellenbosch photo licenses, and some model licenses, rest on an assumption that non-commercial use is fine. They are not verified (`docs/DATA_AND_LICENSES.md`).
- Whether the nuScenes owner approved hosting a model fine-tuned on their data is unrecorded. The demo avoids that model, so please confirm nothing nuScenes-derived is in the bundle.
- The Claude spend cap has been tested in unit tests only, not live.
- Visitor IP addresses appear in server logs, and no retention time is decided.
- The tile classifier is fetched at startup from a private Hugging Face repo, with its SHA-256 pinned in `configs/hosted_classifier.json`.

## 5. Out of scope

Model accuracy (measured in `eval/results.json` and described in the README) and the visual design.

## 6. How to report back

Reply by the date the owner gave you. For each checklist item give pass, fail or not checked, with the reason, and the file and line for any problem. Add anything that worries you that is not on the list.

The owner will fix what you find, and you will look at each fix before release. Any change after your review, including a fix for your own finding, needs a fresh look at that change.
