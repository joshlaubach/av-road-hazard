"""Near-duplicate grouping uses generated images only; no dataset image is read or committed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from defects.splits import make_splits
from tools.make_groups import cluster_near_duplicates, dhash_bits, group_ids, main

MAX_BITS = 24  # 0.10 of the 16x16 hash


def textured(seed: int) -> Image.Image:
    """A smooth random image: its hash is stable under small noise but differs between seeds."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (12, 12), dtype=np.uint8)
    return Image.fromarray(small).resize((96, 96), Image.Resampling.BICUBIC).convert("RGB")


def perturbed(image: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = np.asarray(image, dtype=np.int16) + rng.integers(-3, 4, (96, 96, 3))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def test_hash_is_fixed_width_and_stable_under_small_noise() -> None:
    base = textured(1)
    first, second = dhash_bits(base), dhash_bits(perturbed(base, 9))
    assert first.shape == second.shape == (32,)
    assert first.dtype == np.uint8
    assert int(np.unpackbits(first ^ second).sum()) <= MAX_BITS


def test_duplicates_share_a_cluster_and_distinct_images_do_not() -> None:
    base = textured(1)
    hashes = np.stack([dhash_bits(base), dhash_bits(perturbed(base, 2)), dhash_bits(textured(2))])
    labels, nearest = cluster_near_duplicates(hashes, MAX_BITS)
    assert labels[0] == labels[1] != labels[2]
    assert nearest[0] == nearest[1] <= MAX_BITS < nearest[2]


def test_clustering_is_transitive() -> None:
    """A~B and B~C put A and C together even if A and C are farther apart than the threshold."""
    a = np.zeros(32, dtype=np.uint8)
    b = a.copy()
    b[0] = 0b1111_1111  # 8 bits from a
    c = b.copy()
    c[1] = 0b1111_1111  # 8 bits from b, 16 from a
    labels, _ = cluster_near_duplicates(np.stack([a, b, c]), 10)
    assert labels[0] == labels[1] == labels[2]


def test_single_image_has_no_neighbour_and_its_own_cluster() -> None:
    labels, nearest = cluster_near_duplicates(np.zeros((1, 32), dtype=np.uint8), MAX_BITS)
    assert labels.tolist() == [0]
    assert nearest.tolist() == [256]  # sentinel: farther than any real hash distance


def test_identical_images_in_different_countries_are_never_merged() -> None:
    hashes = np.stack([dhash_bits(textured(1))] * 2)
    groups, _ = group_ids(["Czech/a", "India/b"], hashes, MAX_BITS)
    assert groups["Czech/a"] != groups["India/b"]
    assert groups["Czech/a"].startswith("Czech:")
    assert groups["India/b"].startswith("India:")


def write_index(tmp_path: Path, images: dict[str, Image.Image]) -> Path:
    rows = []
    for sample_id, image in images.items():
        path = tmp_path / "images" / f"{sample_id.replace('/', '_')}.png"
        path.parent.mkdir(exist_ok=True)
        image.save(path)
        rows.append(
            {
                "id": sample_id,
                "image_path": path.relative_to(tmp_path).as_posix(),
                "labels": [1, 0, 0, 0],
            }
        )
    index = tmp_path / "sample_index.json"
    index.write_text(json.dumps(rows))
    return index


def test_cli_groups_cover_every_id_and_are_deterministic(tmp_path: Path) -> None:
    images = {f"Czech/img{i:02d}": textured(i) for i in range(10)}
    images["Czech/img10"] = perturbed(images["Czech/img00"], 5)
    index = write_index(tmp_path, images)
    outputs = []
    for name in ("one.json", "two.json"):
        out = tmp_path / name
        assert (
            main(["--sample-index", str(index), "--out", str(out), "--max-bits", str(MAX_BITS)])
            == 0
        )
        outputs.append(out.read_text())
    assert outputs[0] == outputs[1]
    groups = json.loads(outputs[0])
    assert set(groups) == set(images)
    assert groups["Czech/img00"] == groups["Czech/img10"]
    assert len(set(groups.values())) == 10


def test_duplicates_never_straddle_a_split(tmp_path: Path) -> None:
    images = {f"Czech/img{i:02d}": textured(i) for i in range(12)}
    pairs = {"Czech/dup03": "Czech/img03", "Czech/dup07": "Czech/img07"}
    for dup, original in pairs.items():
        images[dup] = perturbed(images[original], 4)
    out = tmp_path / "groups.json"
    index = write_index(tmp_path, images)
    main(["--sample-index", str(index), "--out", str(out), "--max-bits", str(MAX_BITS)])
    groups = json.loads(out.read_text())
    splits = make_splits(sorted(images), groups.__getitem__, seed=42)
    for dup, original in pairs.items():
        for part in (splits.test, *(p for fold in splits.folds for p in fold)):
            assert (dup in part) == (original in part)


def test_unreadable_image_is_an_error_not_a_silent_skip(tmp_path: Path) -> None:
    index = write_index(tmp_path, {"Czech/a": textured(1)})
    (tmp_path / "images" / "Czech_a.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Czech/a"):
        main(["--sample-index", str(index), "--out", str(tmp_path / "g.json")])


def test_report_records_threshold_and_nearest_distance_histogram(tmp_path: Path) -> None:
    index = write_index(tmp_path, {f"Czech/img{i}": textured(i) for i in range(4)})
    report = tmp_path / "report.json"
    main(["--sample-index", str(index), "--out", str(tmp_path / "g.json"), "--report", str(report)])
    data = json.loads(report.read_text())
    assert data["images"] == 4
    assert data["max_bits"] == 24
    assert data["groups"] == 4
    assert sum(data["nearest_distance_histogram"].values()) == 4
