Purpose: how the hosted demo is put together and what the owner must do by hand; read before deploying to a Hugging Face Space.

# Deploying the demo

## What is built

`docker/Dockerfile.space` makes one image for a Hugging Face Space (CPU, Docker SDK). The website listens on port 7860, the only public port. It forwards `/api/*` to the Python API on `127.0.0.1:8000` inside the same container. `docker/space/start.sh` runs both and stops the container if either one dies, so the Space restarts it. `docker/space/README.md` is the front matter a Space needs (`sdk: docker`, `app_port: 7860`); it becomes the `README.md` at the root of the Space repository, not of this repo.

Checked 2026-11-03 on this PC: the stub image builds (about 90 seconds), runs as user 1000 as Spaces require, `/api/health` and `/credits` answer through port 7860, and killing the API stops the container. Not checked: a real Space, the real-model image (`--build-arg WITH_ML=1`), Qdrant Cloud.

Rate limits count by client address, and uvicorn takes that address from `X-Forwarded-For` when the website server in the same container forwards a request. Checked on the private Space on 2026-11-04: a forged value did not reset the count (25 requests, 429 from the 21st). Not checked: whether visitors on different networks get separate counts or all share the proxy's address; a local `next start` plus uvicorn is dodgeable with a forged header, so do not expose that setup directly.

## Secrets and settings (Space settings, not the repo)

| Name | Kind | Value |
|---|---|---|
| `USE_STUBS` | variable | `0` for the real models; the image default is `1` |
| `QDRANT_URL`, `QDRANT_API_KEY` | secret | from the Qdrant Cloud free-tier cluster |
| `LLM_BACKEND` | variable | `claude` for the hosted answer writer (a Space has no Ollama) |
| `CLASSIFIER_REPO` | variable | the private Hugging Face model repo holding the classifier, `OWNER/NAME` |
| `HF_TOKEN` | secret | a read-only token for that repo |
| `ANTHROPIC_API_KEY` | secret | a key made for this demo only |
| `ANTHROPIC_DAILY_SPEND_CAP_USD`, `ANTHROPIC_MONTHLY_SPEND_CAP_USD` | variable | caps; the app checks before every Claude call (including the second, larger model) and stops at the cap with its fallback message. The count lives in memory and restarts at $0 when the app restarts, and calls already running can finish, so the Anthropic Console limit is the real cap. Leave the server Claude key unset until that limit is set and checked |

The monthly budget is about $20 (Claude is billed separately from any subscription).

## Steps for the owner

1. Create a Qdrant Cloud free cluster; note its URL and key.
2. Create a Space (SDK: Docker, hardware: CPU basic, visibility: private until the release review is done).
3. Upload `best.pt` and `best.json` of the chosen classifier (`runs/tile-fold0-hn2/fold0/`) to a new private Hugging Face model repo; their SHA-256 must equal `configs/hosted_classifier.json` (a different checkpoint means new hashes in a PR). Add the secrets and variables above.
4. Put the reviewed commit on a clean checkout, then run `python -m tools.make_space_repo --out space_repo` (a new folder). It copies the bundle files and changes three things only: a top-level `Dockerfile` (= `docker/Dockerfile.space` with `WITH_ML=1`, so the Space installs the real models), a top-level `README.md` (= the Space front matter), and a `BUNDLE_MANIFEST.json` with the commit and the bundle hash. Push it with `hf upload OWNER/SPACE space_repo . --repo-type space`. A Docker Space needs a Hugging Face PRO plan (the free CPU tier no longer allows Docker Spaces, checked 2026-11-03).
5. Index the demo corpus into Qdrant Cloud from a machine that has the images (they never go into the image or the Space repo): set `USE_STUBS=0`, `DEFECT_CHECKPOINT`, `QDRANT_URL` and `QDRANT_API_KEY`, first make the full-size Stellenbosch photos with `python -m tools.prepare_stellenbosch --raw data/stellenbosch/raw` (the same 20 photos as `tests/fixtures/stellenbosch/`, dashboard cropped off; the shrunk fixtures must not be indexed because shrunk photos miss damage), then run `python -m tools.index_corpus --images rdd2022=data/demo_corpus --images stellenbosch=data/stellenbosch/corpus --records-out cache/records`. It stops if the photo counts differ from `web/src/lib/demo_corpus.json` (40 and 20). Add `--dry-run` to try it in memory first. The cached records in `cache/records` hold no pixels.
6. Smoke check: `/api/health` says `ok` with `qdrant: configured`, one analysis, one search.

## Open before this can go live

- **Weights, decided 2026-11-03:** with `USE_STUBS=0`, `docker/space/start.sh` fetches the public COCO detector and the comma10k road model from their original sources and the classifier from your private repo, each checked against a pinned SHA-256, and stops the container on any failure. OpenCLIP downloads itself from the Hub. No nuScenes-derived weights are used (`docs/DATA_AND_LICENSES.md`, Hosted model weights). The classifier repo is `joshlaubach/road-hazard-classifier` (private); a real fetch from it passed the pinned hashes on 2026-11-03. Not tried: the real-model image (`WITH_ML=1`) running on a Space.
- **Cached outputs need no extra serving.** The two example photos are static files (`web/public/examples/*/example.json`, a saved trace of the real pipeline), and the 60 corpus records live in Qdrant, which Search reads without running any model. Only a visitor's own upload runs the pipeline on CPU, and that cannot be cached. The `--records-out` files are a local backup of what was indexed, not something the Space reads.
- **Release review** of the exact bundle (ROADMAP Next row 2).
