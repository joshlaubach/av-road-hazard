"""Index the hosted demo corpus into Qdrant and save each photo's record as a cached output.

Usage: python -m tools.index_corpus --images rdd2022=data/demo_corpus --images stellenbosch=DIR
           [--records-out cache/records] [--device cpu] [--dry-run]
Runs the real pipeline and OpenCLIP on every image, writes the points to the Qdrant named by
QDRANT_URL (and QDRANT_API_KEY) in the "corpus" session, and writes one JSON record per photo to
--records-out. The records hold boxes, scores and text, never pixels, so they may be shipped with
the Space; the images stay on this machine. Running it again overwrites the same points.

The photo counts must match web/src/lib/demo_corpus.json (the file behind the credits page), or the
run stops before writing anything: the page must not credit photos that are not in the demo.
Refuses stubs and a missing QDRANT_URL unless --dry-run, which indexes in memory and writes nothing
to Qdrant (for a trial on this machine).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from common.config import get_config
from fusion.schema import HazardRecord, Source
from rag.embed import Embedder, HashEmbedder
from rag.index import CORPUS, Index
from rag.text import describe

CORPUS_FILE = Path("web/src/lib/demo_corpus.json")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BATCH = 16  # images embedded at a time, to bound memory on a small machine
# The frozen contract (fusion/schema.py) has no "stellenbosch" source; those photos are filed as
# "upload" and told apart by the prefix of their frame id.
RECORD_SOURCE: dict[str, Source] = {"rdd2022": "rdd2022"}


def expected_counts(corpus_file: Path | None = None) -> dict[str, int]:
    """Photos per source that the credits page promises."""
    data = json.loads((corpus_file or CORPUS_FILE).read_text("utf-8"))
    return {s["id"]: int(s["images"]) for s in data["sources"] if s["include"]}


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def check_counts(found: Mapping[str, int], expected: Mapping[str, int]) -> list[str]:
    """One message per source whose photo count differs from the credits file; empty when fine."""
    problems = []
    for source in sorted(set(found) | set(expected)):
        if found.get(source, 0) != expected.get(source, 0):
            problems.append(
                f"{source}: found {found.get(source, 0)} photos, "
                f"the credits file says {expected.get(source, 0)}"
            )
    return problems


def frame_id(source: str, path: Path) -> str:
    return f"{source}_{path.stem}"


def index_images(
    sources: Mapping[str, Sequence[Path]],
    analyze: Callable[[bytes, Source], HazardRecord],
    embedder: Embedder,
    load_pixels: Callable[[Path], Any],
    index: Index,
) -> list[HazardRecord]:
    """Analyze, embed and store every photo; return the records in order."""
    jobs = [(name, path) for name, paths in sources.items() for path in paths]
    records: list[HazardRecord] = []
    for start in range(0, len(jobs), BATCH):
        batch = jobs[start : start + BATCH]
        made = [
            analyze(path.read_bytes(), RECORD_SOURCE.get(name, "upload")).model_copy(
                update={"frame_id": frame_id(name, path)}
            )
            for name, path in batch
        ]
        pictures = embedder.embed_images([load_pixels(path) for _, path in batch])
        texts = embedder.embed_texts([describe(r) for r in made])
        index.add(made, pictures, texts, CORPUS)
        records.extend(made)
    return records


def write_records(records: Sequence[HazardRecord], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        (out_dir / f"{record.frame_id}.json").write_text(record.model_dump_json(indent=2), "utf-8")


def parse_images(values: Sequence[str]) -> dict[str, Path]:
    named: dict[str, Path] = {}
    for value in values:
        name, sep, folder = value.partition("=")
        if not sep or not name or not folder:
            raise SystemExit(f"--images wants NAME=FOLDER, got {value!r}")
        named[name] = Path(folder)
    return named


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", action="append", required=True, metavar="NAME=FOLDER")
    parser.add_argument("--records-out", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    folders = parse_images(args.images)
    sources = {name: find_images(folder) for name, folder in folders.items()}
    problems = check_counts({n: len(p) for n, p in sources.items()}, expected_counts())
    if problems:
        raise SystemExit(
            "photo counts do not match web/src/lib/demo_corpus.json:\n" + "\n".join(problems)
        )

    cfg = get_config()
    qdrant = cfg["qdrant"]
    stubs = bool(cfg["pipeline"]["use_stubs"])
    if not args.dry_run:
        if stubs:
            raise SystemExit(
                "refusing to index stub records: set USE_STUBS=0 (and DEFECT_CHECKPOINT)"
            )
        if not qdrant["url"]:
            raise SystemExit("set QDRANT_URL (and QDRANT_API_KEY) to the Qdrant Cloud cluster")

    from api.pipeline import Pipeline
    from rag.embed import DIM, ClipEmbedder
    from tools.eval_detector import _rgb

    pipeline = Pipeline()
    index = Index(DIM, None if args.dry_run else qdrant["url"], qdrant.get("api_key"))
    embedder: Embedder = HashEmbedder() if stubs else ClipEmbedder(args.device)
    records = index_images(sources, pipeline.analyze, embedder, _rgb, index)
    if args.records_out:
        write_records(records, args.records_out)
    print(f"indexed {len(records)} photos" + (" (dry run, in memory)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
