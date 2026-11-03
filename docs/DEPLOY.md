Purpose: how the hosted demo is put together and what the owner must do by hand; read before deploying to a Hugging Face Space.

# Deploying the demo

## What is built

`docker/Dockerfile.space` makes one image for a Hugging Face Space (CPU, Docker SDK). The website listens on port 7860, the only public port. It forwards `/api/*` to the Python API on `127.0.0.1:8000` inside the same container. `docker/space/start.sh` runs both and stops the container if either one dies, so the Space restarts it. `docker/space/README.md` is the front matter a Space needs (`sdk: docker`, `app_port: 7860`); it becomes the `README.md` at the root of the Space repository, not of this repo.

Checked 2026-11-03 on this PC: the stub image builds (about 90 seconds), runs as user 1000 as Spaces require, `/api/health` and `/credits` answer through port 7860, and killing the API stops the container. Not checked: a real Space, the real-model image (`--build-arg WITH_ML=1`), Qdrant Cloud.

## Secrets and settings (Space settings, not the repo)

| Name | Kind | Value |
|---|---|---|
| `USE_STUBS` | variable | `0` for the real models; the image default is `1` |
| `QDRANT_URL`, `QDRANT_API_KEY` | secret | from the Qdrant Cloud free-tier cluster |
| `LLM_BACKEND` | variable | `claude` for the hosted answer writer (a Space has no Ollama) |
| `ANTHROPIC_API_KEY` | secret | a key made for this demo only |
| `ANTHROPIC_DAILY_SPEND_CAP_USD`, `ANTHROPIC_MONTHLY_SPEND_CAP_USD` | variable | caps; the app stops calling Claude at the cap and shows its fallback message |

The monthly budget is about $20 (Claude is billed separately from any subscription).

## Steps for the owner

1. Create a Qdrant Cloud free cluster; note its URL and key.
2. Create a Space (SDK: Docker, hardware: CPU basic, visibility: private until the release review is done).
3. Add the secrets and variables above.
4. Push the reviewed bundle plus `docker/space/README.md` as the Space's `README.md` to the Space remote.
5. Index the demo corpus into Qdrant Cloud from a machine that has the images (they never go into the image or the Space repo): set `USE_STUBS=0`, `DEFECT_CHECKPOINT`, `QDRANT_URL` and `QDRANT_API_KEY`, then run `python -m tools.index_corpus --images rdd2022=data/demo_corpus --images stellenbosch=FOLDER --records-out cache/records`. It stops if the photo counts differ from `web/src/lib/demo_corpus.json` (40 and 20). Add `--dry-run` to try it in memory first. The cached records in `cache/records` hold no pixels.
6. Smoke check: `/api/health` says `ok` with `qdrant: configured`, one analysis, one search.

## Open before this can go live

- **Where the weights live.** The image has none, by design. The Space needs the detector, road model, classifier and OpenCLIP weights at run time. They are not in git. Options: a private Hugging Face model repo downloaded at start (needs a token secret and a download step in `start.sh`), or baked into the image at build. Not decided; whichever is chosen, check that the detector and road weights, which were fine-tuned on nuScenes, may be hosted (open item in `docs/DATA_AND_LICENSES.md`).
- **The 20 Stellenbosch photos are not picked yet.** The indexing script needs a folder with exactly 20; today only the two example photos and the raw download exist locally.
- **Cached records are made but not served.** `--records-out` writes them; nothing in the Space reads them yet, so the demo still computes on CPU.
- **Release review** of the exact bundle (ROADMAP Next row 2).
