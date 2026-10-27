Purpose: how to run, test and change the av-road-hazard web UI (Next.js and TypeScript).

# Web UI

The Analyze view: choose a road photo (or one of two bundled examples) and see what the pipeline found, in plain language, with the hazard score on a 0 to 100 gauge. The math is behind an expandable section and typeset with KaTeX.

## Run it

```
# terminal 1: the API (real models; set USE_STUBS=1 for the fixed fake answers)
USE_STUBS=0 DEFECT_CHECKPOINT=path/to/best.pt python -m uvicorn api.main:app --port 8000
# terminal 2: the page
cd web && npm ci && npm run dev        # http://localhost:3000
```

The browser calls `/api/*` on the web server, which forwards to the API (`API_URL`, default `http://localhost:8000`; see `next.config.mjs`). One origin, so no CORS setup.

## Check it

`npm run typecheck`, `npm run lint`, `npm test`, `npm run build`. CI runs all four, and regenerates the client to check that it matches `openapi.json`.

## The typed client

`src/api/schema.d.ts` is generated from `../openapi.json` (`npm run generate-client`); never edit it by hand. After an API change: `python -m tools.export_openapi`, then `npm run generate-client`.

## Examples

`public/examples/<id>/` holds a shrunk display photo and the real trace of the full-size photo (`example.json`). Make one with `python -m tools.make_example` (see its docstring). Only photos we may show go here, with credit: the Stellenbosch pothole photos (CC0 per the Kaggle and Dataset Ninja listings). Never a nuScenes or RDD2022 image; a test checks this.

## Rules this page follows

- Plain-language (ELI16) text is the default; the technical detail is mathematics in KaTeX.
- The stored `hazard_score` is 0 to 1; people see a whole number 0 to 100.
- Photos are analyzed at full size; only the copy shown on the page is shrunk.
