"""Make a web example: a shrunk display photo and the real trace of the full-size photo.

Usage: python -m tools.make_example IMAGE --id ID --title TITLE --credit TEXT
           --out web/public/examples [--note TEXT] [--display-width 1600]
Needs USE_STUBS=0 and the models for a real trace. The pipeline runs on the full-size image (a
shrunk copy can miss defects the tile size expects to see); only the display copy is shrunk, and
that is the only image the page shows. Writes OUT/ID/display.jpg and OUT/ID/example.json, which
holds the title, credit, note and the TracedAnalysis exactly as POST /analyze/trace returns it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from api.pipeline import Pipeline

DISPLAY_QUALITY = 85


def make_example(
    image: Path,
    *,
    example_id: str,
    title: str,
    credit: str,
    note: str,
    out_dir: Path,
    display_width: int = 1600,
    pipeline: Pipeline | None = None,
) -> Path:
    """Write the example folder and return it."""
    folder = out_dir / example_id
    folder.mkdir(parents=True, exist_ok=True)
    traced = (pipeline or Pipeline()).analyze_traced(image.read_bytes(), source="upload")
    with Image.open(image) as full:
        rgb = full.convert("RGB")
        scale = min(1.0, display_width / rgb.width)
        shown = rgb.resize(
            (round(rgb.width * scale), round(rgb.height * scale)), Image.Resampling.LANCZOS
        )
        shown.save(folder / "display.jpg", quality=DISPLAY_QUALITY, optimize=True)  # no EXIF
    payload: dict[str, Any] = {
        "id": example_id,
        "title": title,
        "credit": credit,
        "note": note,
        "analyzed_size": list(traced.record.image_size),
        "display_size": [shown.width, shown.height],
        "analysis": traced.model_dump(mode="json"),
    }
    (folder / "example.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--credit", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--display-width", type=int, default=1600)
    args = parser.parse_args(argv)
    folder = make_example(
        args.image,
        example_id=args.id,
        title=args.title,
        credit=args.credit,
        note=args.note,
        out_dir=args.out,
        display_width=args.display_width,
    )
    print(f"wrote {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
