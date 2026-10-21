"""Draw a HazardRecord's boxes on its source image (debugging aid for coordinate bugs).

Usage: python -m tools.overlay IMAGE RECORD.json OUT.png
Objects are drawn in blue; defects are colored by severity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from fusion.schema import HazardRecord

# Defect box colors by severity; unknown is magenta so it stands out as "not yet scored".
SEVERITY_COLOR = {"high": "red", "medium": "orange", "low": "yellow", "unknown": "magenta"}


def draw(image: Image.Image, record: HazardRecord) -> Image.Image:
    """Return a copy of the image with the record's objects and defects drawn on it.

    This is the first tool to reach for when a box looks wrong: seeing the boxes on the frame
    separates coordinate bugs (swapped width/height, wrong scale) from model errors.
    """
    out = image.convert("RGB").copy()
    draw_ = ImageDraw.Draw(out)  # trailing underscore: avoids shadowing this function's name
    for o in record.objects:
        draw_.rectangle(o.box, outline="deepskyblue", width=3)
        draw_.text((o.box[0] + 2, o.box[1] + 2), f"{o.label} {o.score:.2f}", fill="deepskyblue")
    for d in record.defects:
        color = SEVERITY_COLOR[d.severity]
        draw_.rectangle(d.box, outline=color, width=3)
        draw_.text((d.box[0] + 2, d.box[1] + 2), f"{d.name} {d.confidence:.2f}", fill=color)
    return out


def main() -> int:
    """Command line entry: read the image and the record JSON, write the annotated PNG.

    Prints a warning (but still draws) when the image size differs from the record's image_size,
    which is itself a common sign of a resize or width/height mix-up.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("record", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    record = HazardRecord.model_validate_json(args.record.read_text())
    with Image.open(args.image) as img:
        if img.size != record.image_size:
            print(f"warning: image size {img.size} != record image_size {record.image_size}")
        draw(img, record).save(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
