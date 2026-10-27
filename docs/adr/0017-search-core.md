Purpose: ADR 0017: how photo search is built (embedder, index, retrieval) and what the first measurement showed.

# ADR 0017: The search core

Status: accepted (2026-10-27), first slice of the RAG row. No language model yet.

## Context
The Search view needs to find frames from words, from a photo, and from plain filters. The roadmap asks for recall at k on queries made from the nuScenes annotations before any answer text is written.

## Decision
1. **Embedder.** OpenCLIP ViT-B/32 with the LAION-2B weights (`rag/embed.py`): about 150 million parameters, runs on a CPU, and puts photos and sentences in one space. The larger ViT-L/14 was left out because it is about four times slower and too heavy for free hosting.
2. **Index.** Qdrant, one collection, one point per photo with two vectors: the photo itself, and a short plain-English description of its record (`rag/text.py`: light, object counts, damage names, hazard score). Metadata (source, time of day, hazard score, worst severity, object labels, damage codes) sits beside them. Locally it runs in memory or in a folder; the hosted version uses a Qdrant server (`rag/index.py`).
3. **Retrieval.** Filters first (plain conditions), then a search on each vector, then reciprocal rank fusion: a photo scores the sum of 1/(60 + rank) over the lists.
4. **Sessions.** Every point has a session ("corpus" for the shared demo photos, an id for a visitor's uploads). A search must name the sessions it may see (an empty list is an error), and reading one record checks the same thing. A test shows one session cannot see another's photos. The API wiring comes in a later slice.
5. **Measurement.** `python -m tools.eval_retrieval` runs the real pipeline on all 404 nuScenes CAM_FRONT frames, indexes them, and scores queries such as "a road scene with traffic cones". A frame is relevant if the human annotations have a clearly visible object of that kind (at least 40% visible and at least 1024 square pixels in area, the same rule the detector evaluation uses; a thin box can qualify). Recall at k is hits in the top k over min(k, relevant frames). Queries with fewer than 3 or more than half the frames relevant are dropped, which leaves 21. Ranges are 90% resampling ranges over queries. Random order is the floor.

## Result (404 frames, 21 queries, recall at 5)
Both vectors fused 0.54, photo 0.43, description 0.41, random order 0.18 (the exact average over all orders). The details, with 90% resampling ranges and the provenance block, are in `eval/results.json` under `retrieval`.

## What this means
- Fusing the two rankings beats either alone, so equal-weight fusion stays the default. Weights are not tuned: 21 queries would only fit the test.
- The description search is best for object queries (0.63 at 5, against 0.40 for the photo), because the description lists the detector's objects. The photo search is the only one that can answer light and appearance words.
- A first measurement gave the description 0.63 and fused 0.52. It was wrong: the records were given the day or night label from the same scene annotations that decide which frames are relevant, so the description "knew" the answer to light queries. A Codex inspection caught it. The pipeline does not know the time of day, so the records are now used as the pipeline makes them, and the numbers above come from that run.
- The description inherits the detector's misses (it does not see barriers or cones well, ADR 0015), so "traffic cones" queries cannot do well through it.

## Limits
Only 1 of the 21 queries is about light (daylight is over half of all frames), 7 are about objects and 13 about objects at night or in daylight. The light label used to judge relevance comes from the scene annotations; the search never sees it. Cached records are reused only for the same frames, dataset version, model files and settings. Annotated nuScenes frames only; the hosted corpus is checked separately once it exists.
