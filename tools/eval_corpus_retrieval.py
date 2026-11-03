"""Measure photo search on the 40-image demo corpus: recall at k for damage and clean-road queries.

Usage: python -m tools.eval_corpus_retrieval [--corpus data/demo_corpus] [--device cuda]
           [--out report.json]
Needs USE_STUBS=0, the models and DEFECT_CHECKPOINT (the same setup as tools/eval_retrieval.py);
it refuses to run on stubs. Runs the real pipeline on every corpus image, embeds each image and
its record description with OpenCLIP, indexes them in an in-memory Qdrant and scores the same
searches as the nuScenes check: photo vector, description vector, both fused, and a random floor.

Queries (fixed before the first run): "a road with <damage>" for each of the four RDD2022 damage
types and "a road with no damage". A result is relevant when the dataset's own boxes (the corpus
file WHAT_IT_SHOWS.json) say that damage is in the image, never what the pipeline found, so the
numbers include the pipeline's own mistakes.

Pass rule (fixed before the first run): fused recall@5 beats the random floor's recall@5 by at
least 0.15 and the low end of its 90% range is above the floor. Five queries make the range wide,
so the verdict is a direction, not a guarantee. The images and records stay on this machine; the
report holds only numbers and hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.config import get_config
from eval.retrieval import Query, random_expectation, score_ranker, summarize
from fusion.schema import HazardRecord
from rag.embed import MODEL, PRETRAINED, ClipEmbedder
from rag.index import Index
from rag.text import describe
from tools.eval_detector import _rgb
from tools.eval_retrieval import parse_weights, pipeline_provenance, rankers

MIN_MARGIN = 0.15  # fused recall@5 must beat random by at least this
DAMAGE_PHRASES = {
    "D00": "a longitudinal crack",
    "D10": "a transverse crack",
    "D20": "an alligator crack",
    "D40": "a pothole",
}


def corpus_queries(images: list[dict[str, Any]]) -> list[Query]:
    """One query per damage type and one for a clean road; relevance from the dataset's boxes."""
    stems = {i["file"]: Path(i["file"]).stem for i in images}
    queries = []
    for code, phrase in DAMAGE_PHRASES.items():
        relevant = frozenset(
            stems[i["file"]] for i in images if code in {b["code"] for b in i["boxes"]}
        )
        queries.append(Query(f"a road with {phrase}", "damage", relevant))
    clean = frozenset(stems[i["file"]] for i in images if not i["boxes"])
    queries.append(Query("a road with no damage", "clean", clean))
    return [q for q in queries if q.relevant]


def verdict(fused: dict[str, Any], random: dict[str, Any]) -> dict[str, Any]:
    """The pre-set rule on recall@5 over all queries."""
    f, r = fused["all"]["recall@5"], random["all"]["recall@5"]
    margin = f["mean"] - r["mean"]
    passed = margin >= MIN_MARGIN and f["low"] > r["mean"]
    return {
        "rule": (
            f"fused recall@5 beats random by >= {MIN_MARGIN} and its 90% low end is above random"
        ),
        "fused_recall@5": f,
        "random_recall@5": r["mean"],
        "margin": round(margin, 4),
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/demo_corpus"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--photo-weights", default="", help="e.g. 0.5: extra fused searches")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    prov = pipeline_provenance()
    images = json.loads((args.corpus / "WHAT_IT_SHOWS.json").read_text("utf-8"))["images"]
    from api.pipeline import Pipeline

    pipeline = Pipeline()
    paths = [args.corpus / i["file"] for i in images]
    records: list[HazardRecord] = []
    for path in paths:
        record = pipeline.analyze(path.read_bytes(), "rdd2022")
        records.append(record.model_copy(update={"frame_id": path.stem}))
    embedder = ClipEmbedder(args.device)
    pictures = embedder.embed_images([_rgb(p) for p in paths])
    texts = embedder.embed_texts([describe(r) for r in records])
    index = Index(embedder.dim)
    index.add(records, pictures, texts)
    frame_ids = [r.frame_id for r in records]
    queries = corpus_queries(images)
    seed = int(get_config()["seed"])
    methods = {
        **{
            name: summarize(score_ranker(ranker, queries), queries, seed)
            for name, ranker in rankers(
                index, embedder, frame_ids, parse_weights(args.photo_weights)
            ).items()
        },
        "random": summarize(random_expectation(queries, len(frame_ids)), queries, seed),
    }
    report: dict[str, Any] = {
        "images": len(images),
        "queries": [{"text": q.text, "relevant": len(q.relevant)} for q in queries],
        "embedder": {"model": MODEL, "pretrained": PRETRAINED},
        "methods": methods,
        "verdict": verdict(methods["fused"], methods["random"]),
        "provenance": prov,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
