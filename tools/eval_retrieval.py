"""Measure photo search on nuScenes CAM_FRONT: recall at k for queries made from the annotations.

Usage: python -m tools.eval_retrieval --nuscenes-root data/nuscenes/raw [--version v1.0-mini]
           [--records records.json] [--device cuda] [--out report.json]
Needs USE_STUBS=0, the models and DEFECT_CHECKPOINT (the same setup as tools/smoke_analyze.py); it
refuses to run on stubs. Runs the real pipeline on every frame (records are cached in --records,
because that is the slow part; the cache is used only if it was made for the same frames, dataset
version and model files, otherwise it is rebuilt), embeds each frame and its record description with
OpenCLIP, indexes them in an in-memory Qdrant and scores four searches on queries from
eval/retrieval.py: the photo vector only, the description vector only, both fused (reciprocal rank),
and a random order as the floor. Relevance comes from the human annotations (boxes at least 40%
visible and at least 1024 square pixels in area, the detector evaluation's rule), not from the
pipeline, so the numbers include the pipeline's own mistakes. The nuScenes images and records stay
on this machine; the report holds only numbers and hashes (with a provenance block).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from common.config import get_config
from defects.road_crop import MODEL_SUBDIR
from eval.detector import evaluable
from eval.provenance import provenance
from eval.retrieval import (
    FrameFacts,
    Query,
    Ranker,
    cache_matches,
    make_queries,
    random_expectation,
    score_ranker,
    summarize,
)
from fusion.schema import HazardRecord
from perception.detector import DETECTOR_SUBDIR, DETECTORS
from perception.infer import detector_settings
from rag.embed import MODEL, PRETRAINED, ClipEmbedder, Embedder
from rag.index import CORPUS, IMAGE_VECTOR, TEXT_VECTOR, Index
from rag.text import describe
from tools.eval_detector import _load_frames, _rgb

REPO = Path(__file__).resolve().parents[1]


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def pipeline_provenance() -> dict[str, Any]:
    """Hashes of the model files the pipeline will use and its settings; refuses stub mode."""
    cfg = get_config()
    if cfg["pipeline"]["use_stubs"]:
        raise SystemExit("refusing to measure on stubs: set USE_STUBS=0 (and DEFECT_CHECKPOINT)")
    checkpoint = cfg["defects"]["classifier_checkpoint"]
    if not checkpoint:
        raise SystemExit("set DEFECT_CHECKPOINT to the classifier checkpoint")
    root = Path(cfg["paths"]["checkpoint_dir"])
    name, score = detector_settings(cfg)
    files = {
        "checkpoint/defects": Path(checkpoint),
        "detector": root / DETECTOR_SUBDIR / DETECTORS[name].filename,
        "road_model": root / MODEL_SUBDIR / "model.safetensors",
    }
    settings = {
        "object_score": score,
        **{k: v for k, v in cfg["defects"].items() if _number(v)},
        **{f"cutoff_{code}": v for code, v in cfg["defects"]["thresholds"].items()},
    }
    configs = {
        "config/default": REPO / "configs" / "default.yaml",
        "config/fusion": REPO / "configs" / "fusion.yaml",
    }
    return provenance(files, thresholds=settings, repo=REPO, text_files=configs)


def build_records(
    frames: list[Any],
    version: str,
    config: dict[str, Any],
    cache: Path | None,
    *,
    rebuild: bool = True,
) -> list[HazardRecord]:
    """One record per frame from the real pipeline; frame_id is the image file name."""
    frame_ids = [path.stem for _, _, path, _ in frames]
    if cache is not None and cache.exists():
        stored = json.loads(cache.read_text("utf-8"))
        if cache_matches(stored, version, frame_ids, config):
            return [HazardRecord.model_validate(r) for r in stored["records"]]
        if not rebuild:
            raise SystemExit(
                "the cached records were made for other frames, model files or settings; "
                "refusing to rebuild them (--no-rebuild)"
            )
        print("record cache does not match these frames or models; rebuilding it")
    from api.pipeline import Pipeline

    pipeline = Pipeline()
    records = []
    for _, _, path, _ in frames:
        record = pipeline.analyze(path.read_bytes(), "nuscenes")
        records.append(record.model_copy(update={"frame_id": path.stem}))
    if cache is not None:
        stored = {
            "version": version,
            "frame_ids": frame_ids,
            "config": config,
            "records": [r.model_dump(mode="json") for r in records],
        }
        cache.write_text(json.dumps(stored), "utf-8")
    return records


def frame_facts(frames: list[Any]) -> list[FrameFacts]:
    return [
        FrameFacts(path.stem, night, frozenset(b.label for b in boxes if evaluable(b)))
        for _, night, path, boxes in frames
    ]


def rankers(
    index: Index,
    embedder: Embedder,
    frame_ids: list[str],
    photo_weights: Sequence[float] = (),
) -> dict[str, Ranker]:
    """The three searches, plus one fused search per photo weight (description weight stays 1)."""

    def search(names: tuple[str, ...], weights: Mapping[str, float] | None = None) -> Ranker:
        def rank(query: Query) -> list[str]:
            vector = embedder.embed_texts([query.text])[0]
            hits = index.search(
                {n: vector for n in names}, [CORPUS], top_k=len(frame_ids), weights=weights
            )
            return [h.record.frame_id for h in hits]

        return rank

    both = (IMAGE_VECTOR, TEXT_VECTOR)
    return {
        "photo_vector": search((IMAGE_VECTOR,)),
        "description_vector": search((TEXT_VECTOR,)),
        "fused": search(both),
        **{f"fused_photo_weight_{w}": search(both, {IMAGE_VECTOR: w}) for w in photo_weights},
    }


def parse_weights(text: str) -> list[float]:
    """'1,0.5,0.25' -> [1.0, 0.5, 0.25]; the empty string means no extra searches."""
    return [float(part) for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="stop instead of re-running the pipeline when the cached records do not match",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--photo-weights", default="", help="e.g. 0.5,0.25: extra fused searches")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    from nuscenes.nuscenes import NuScenes

    prov = pipeline_provenance()
    nusc = NuScenes(version=args.version, dataroot=str(args.nuscenes_root), verbose=False)
    frames = _load_frames(nusc)
    config = {"files": prov["files"], "thresholds": prov["thresholds"]}
    records = build_records(frames, args.version, config, args.records, rebuild=not args.no_rebuild)
    embedder = ClipEmbedder(args.device)
    images = embedder.embed_images([_rgb(path) for _, _, path, _ in frames])
    texts = embedder.embed_texts([describe(r) for r in records])
    index = Index(embedder.dim)
    index.add(records, images, texts)
    frame_ids = [r.frame_id for r in records]
    queries = make_queries(frame_facts(frames))
    seed = int(get_config()["seed"])
    report: dict[str, Any] = {
        "version": args.version,
        "frames": len(frames),
        "queries": len(queries),
        "embedder": {"model": MODEL, "pretrained": PRETRAINED},
        "pipeline": {"use_stubs": False},
        "methods": {
            **{
                name: summarize(score_ranker(ranker, queries), queries, seed)
                for name, ranker in rankers(
                    index, embedder, frame_ids, parse_weights(args.photo_weights)
                ).items()
            },
            "random": summarize(random_expectation(queries, len(frame_ids)), queries, seed),
        },
        "light_note": (
            "Records come straight from the pipeline, which does not know the time of day, so "
            "light queries are answered by the photo vector alone."
        ),
        "provenance": prov,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
