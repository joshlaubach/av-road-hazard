"""Measure photo search on nuScenes CAM_FRONT: recall at k for queries made from the annotations.

Usage: python -m tools.eval_retrieval --nuscenes-root data/nuscenes/raw [--version v1.0-mini]
           [--records records.json] [--out report.json]
Runs the real pipeline on every frame (records are cached in --records, because that is the slow
part), embeds each frame and its record description with OpenCLIP, indexes them in an in-memory
Qdrant and scores four searches on queries from eval/retrieval.py: the photo vector only, the
description vector only, both fused (reciprocal rank), and a random order as the floor. Relevance
comes from the human annotations (boxes at least 40% visible and 32 x 32 px), not from the
pipeline, so the numbers include the pipeline's own mistakes. The nuScenes images and records stay
on this machine; the report holds only numbers and hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.detector import evaluable
from eval.provenance import provenance
from eval.retrieval import (
    FrameFacts,
    Query,
    Ranker,
    make_queries,
    random_ranker,
    score_ranker,
    summarize,
)
from fusion.schema import HazardRecord, Meta
from rag.embed import MODEL, PRETRAINED, ClipEmbedder, Embedder
from rag.index import CORPUS, IMAGE_VECTOR, TEXT_VECTOR, Index
from rag.text import describe
from tools.eval_detector import _load_frames, _rgb


def build_records(frames: list[Any], cache: Path | None) -> list[HazardRecord]:
    """One record per frame from the real pipeline; frame_id is the image file name."""
    if cache is not None and cache.exists():
        return [HazardRecord.model_validate(r) for r in json.loads(cache.read_text("utf-8"))]
    from api.pipeline import Pipeline

    pipeline = Pipeline()
    records = []
    for _, night, path, _ in frames:
        record = pipeline.analyze(path.read_bytes(), "nuscenes")
        meta = Meta(time_of_day="night" if night else "day")
        records.append(record.model_copy(update={"frame_id": path.stem, "meta": meta}))
    if cache is not None:
        cache.write_text(json.dumps([r.model_dump(mode="json") for r in records]), "utf-8")
    return records


def frame_facts(frames: list[Any]) -> list[FrameFacts]:
    return [
        FrameFacts(path.stem, night, frozenset(b.label for b in boxes if evaluable(b)))
        for _, night, path, boxes in frames
    ]


def rankers(index: Index, embedder: Embedder, frame_ids: list[str]) -> dict[str, Ranker]:
    def search(names: tuple[str, ...]) -> Ranker:
        def rank(query: Query) -> list[str]:
            vector = embedder.embed_texts([query.text])[0]
            hits = index.search({n: vector for n in names}, [CORPUS], top_k=len(frame_ids))
            return [h.record.frame_id for h in hits]

        return rank

    return {
        "photo_vector": search((IMAGE_VECTOR,)),
        "description_vector": search((TEXT_VECTOR,)),
        "fused": search((IMAGE_VECTOR, TEXT_VECTOR)),
        "random": random_ranker(frame_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=str(args.nuscenes_root), verbose=False)
    frames = _load_frames(nusc)
    records = build_records(frames, args.records)
    embedder = ClipEmbedder(args.device)
    images = embedder.embed_images([_rgb(path) for _, _, path, _ in frames])
    texts = embedder.embed_texts([describe(r) for r in records])
    index = Index(embedder.dim)
    index.add(records, images, texts)
    frame_ids = [r.frame_id for r in records]
    queries = make_queries(frame_facts(frames))
    report: dict[str, Any] = {
        "version": args.version,
        "frames": len(frames),
        "queries": len(queries),
        "embedder": {"model": MODEL, "pretrained": PRETRAINED},
        "methods": {
            name: summarize(score_ranker(ranker, queries), queries)
            for name, ranker in rankers(index, embedder, frame_ids).items()
        },
        "provenance": provenance({}, thresholds={}, repo=Path(__file__).resolve().parents[1]),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
