"""Behavioral checks for the local nuScenes threshold review."""

from pathlib import Path

from PIL import Image

from defects.patches import Tile
from defects.review_nuscenes import ReviewHit, annotate_frame, select_hits


def _hit(name: str, score: float) -> ReviewHit:
    return ReviewHit(
        image=Path(name),
        score=score,
        class_code="D00",
        tile=Tile(0, 0, (40, 20, 120, 100)),
        image_size=(400, 200),
    )


def test_selection_respects_both_score_bands_and_input_order() -> None:
    hits = [
        _hit("low.jpg", 0.399),
        _hit("b1.jpg", 0.4),
        _hit("b2.jpg", 0.48),
        _hit("b3.jpg", 0.499),
        _hit("a1.jpg", 0.5),
        _hit("a2.jpg", 0.55),
        _hit("a3.jpg", 0.599),
        _hit("high.jpg", 0.6),
    ]

    chosen = select_hits(hits, threshold=0.5, margin=0.1, per_side=2, seed=42)
    reversed_chosen = select_hits(
        list(reversed(hits)), threshold=0.5, margin=0.1, per_side=2, seed=42
    )

    assert chosen == reversed_chosen
    assert len(chosen["below"]) == len(chosen["above"]) == 2
    assert all(0.4 <= hit.score < 0.5 for hit in chosen["below"])
    assert all(0.5 <= hit.score < 0.6 for hit in chosen["above"])
    assert len({hit.image for group in chosen.values() for hit in group}) == 4


def test_annotation_scales_tile_box_on_nonsquare_frame() -> None:
    image = Image.new("RGB", (400, 200), "white")

    annotated = annotate_frame(image, _hit("frame.jpg", 0.51), max_size=(200, 100))

    assert annotated.size == (200, 100)
    assert annotated.getpixel((20, 10)) == (255, 0, 255)
    assert annotated.getpixel((60, 50)) == (255, 0, 255)
    assert annotated.getpixel((100, 50)) == (255, 255, 255)
    assert image.getpixel((40, 20)) == (255, 255, 255)
