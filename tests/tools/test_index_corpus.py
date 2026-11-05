"""Tests for tools/index_corpus.py with stub models: counts, indexing, cached records, safety."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from api.pipeline import Pipeline
from fusion.schema import HazardRecord
from rag.embed import DIM, HashEmbedder
from rag.index import CORPUS, Index
from tools import index_corpus
from tools.index_corpus import (
    check_counts,
    check_manifest,
    expected_counts,
    index_images,
    parse_images,
    write_records,
)


def make_images(folder: Path, names: list[str]) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, name in enumerate(names):
        path = folder / f"{name}.png"
        Image.new("RGB", (32, 32), (10 * i, 90, 90)).save(path)
        paths.append(path)
    return paths


def pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def run(sources: dict[str, list[Path]], index: Index) -> list[HazardRecord]:
    return index_images(sources, Pipeline().analyze, HashEmbedder(), pixels, index)


def two_sources(tmp_path: Path) -> dict[str, list[Path]]:
    return {
        "rdd2022": make_images(tmp_path / "a", ["x", "y"]),
        "stellenbosch": make_images(tmp_path / "b", ["z"]),
    }


def test_every_photo_is_stored_in_the_corpus_session(tmp_path: Path) -> None:
    index = Index(DIM)
    records = run(two_sources(tmp_path), index)
    assert [r.frame_id for r in records] == ["rdd2022_x", "rdd2022_y", "stellenbosch_z"]
    assert index.count([CORPUS]) == 3
    stored = index.get("stellenbosch_z", [CORPUS])
    assert stored is not None and stored.frame_id == "stellenbosch_z"


def test_sources_outside_the_contract_are_filed_as_upload(tmp_path: Path) -> None:
    records = run(two_sources(tmp_path), Index(DIM))
    assert [r.source for r in records] == ["rdd2022", "rdd2022", "upload"]


def test_running_twice_does_not_duplicate_points(tmp_path: Path) -> None:
    sources = {"rdd2022": make_images(tmp_path, ["x", "y"])}
    index = Index(DIM)
    run(sources, index)
    run(sources, index)
    assert index.count([CORPUS]) == 2


def test_cached_records_are_json_with_no_pixels(tmp_path: Path) -> None:
    records = run({"rdd2022": make_images(tmp_path / "a", ["x"])}, Index(DIM))
    write_records(records, tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "rdd2022_x.json").read_text("utf-8"))
    assert saved["frame_id"] == "rdd2022_x"


def test_a_count_that_differs_from_the_credits_file_is_reported() -> None:
    same = {"rdd2022": 40, "stellenbosch": 20}
    assert check_counts(same, same) == []
    problems = check_counts({"rdd2022": 39}, same)
    assert len(problems) == 2 and "rdd2022: found 39" in problems[0]


def test_the_credits_file_promises_the_decided_counts() -> None:
    assert expected_counts() == {"rdd2022": 40, "stellenbosch": 20}


def test_images_argument_needs_a_name_and_a_folder() -> None:
    assert parse_images(["a=b"]) == {"a": Path("b")}
    with pytest.raises(SystemExit):
        parse_images(["nofolder"])


def test_a_wrong_count_stops_the_run_before_anything_is_written(tmp_path: Path) -> None:
    folder = tmp_path / "a"
    make_images(folder, ["x"])
    with pytest.raises(SystemExit, match="do not match"):
        index_corpus.main(["--images", f"rdd2022={folder}", "--records-out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_a_dry_run_on_stubs_writes_the_cached_records_and_touches_no_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credits = tmp_path / "credits.json"
    credits.write_text(json.dumps({"sources": [{"id": "rdd2022", "include": True, "images": 2}]}))
    monkeypatch.setattr(index_corpus, "CORPUS_FILE", credits)
    folder = tmp_path / "a"
    make_images(folder, ["x", "y"])
    out = tmp_path / "out"
    manifest = tmp_path / "manifest.json"
    args = ["--images", f"rdd2022={folder}", "--manifest", str(manifest), "--dry-run"]
    code = index_corpus.main([*args, "--records-out", str(out)])
    assert code == 0 and sorted(p.name for p in out.iterdir()) == [
        "rdd2022_x.json",
        "rdd2022_y.json",
    ]
    assert sorted(json.loads(manifest.read_text("utf-8"))) == ["rdd2022_x", "rdd2022_y"]
    Image.new("RGB", (32, 32), (255, 0, 0)).save(folder / "x.png")  # a photo changes afterwards
    with pytest.raises(SystemExit, match="rdd2022_x: file changed"):
        index_corpus.main(args)


def test_a_real_run_refuses_stub_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    credits = tmp_path / "credits.json"
    credits.write_text(json.dumps({"sources": [{"id": "rdd2022", "include": True, "images": 1}]}))
    monkeypatch.setattr(index_corpus, "CORPUS_FILE", credits)
    monkeypatch.setattr(index_corpus, "MANIFEST_FILE", tmp_path / "manifest.json")
    make_images(tmp_path / "a", ["x"])
    with pytest.raises(SystemExit, match="stub"):
        index_corpus.main(["--images", f"rdd2022={tmp_path / 'a'}"])


def test_a_stray_record_in_the_corpus_is_removed_by_the_import(tmp_path: Path) -> None:
    index = Index(DIM)
    stray = Pipeline().analyze(Path(make_images(tmp_path / "s", ["old"])[0]).read_bytes(), "upload")
    index.add([stray.model_copy(update={"frame_id": "old"})], np.ones((1, DIM)), np.ones((1, DIM)))
    run(two_sources(tmp_path), index)
    assert index.count([CORPUS]) == 3 and index.get("old", [CORPUS]) is None


def test_the_import_fails_if_the_corpus_does_not_end_up_as_exactly_the_photos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = Index(DIM)
    stray = Pipeline().analyze(Path(make_images(tmp_path / "s", ["old"])[0]).read_bytes(), "upload")
    index.add([stray.model_copy(update={"frame_id": "old"})], np.ones((1, DIM)), np.ones((1, DIM)))
    monkeypatch.setattr(index, "clear_corpus", lambda: None)
    with pytest.raises(RuntimeError, match="old"):
        run(two_sources(tmp_path), index)


def test_two_photos_with_the_same_name_in_one_source_are_refused(tmp_path: Path) -> None:
    both = [*make_images(tmp_path / "a", ["x"]), *make_images(tmp_path / "b", ["x"])]
    index = Index(DIM)
    with pytest.raises(ValueError, match="rdd2022_x"):
        run({"rdd2022": both}, index)
    assert index.count([CORPUS]) == 0


def test_the_manifest_must_match_the_photos_exactly() -> None:
    manifest = {"rdd2022_x": "aa", "rdd2022_y": "bb"}
    assert check_manifest(manifest, manifest) == []
    problems = check_manifest({"rdd2022_x": "ZZ", "rdd2022_z": "cc"}, manifest)
    assert len(problems) == 3  # x changed, z not in the manifest, y missing


def test_the_committed_manifest_matches_the_counts_the_credits_page_promises() -> None:
    manifest = json.loads(index_corpus.MANIFEST_FILE.read_text("utf-8"))
    per_source = {s: sum(k.startswith(s + "_") for k in manifest) for s in expected_counts()}
    assert per_source == expected_counts() and len(manifest) == sum(expected_counts().values())
    assert len(set(manifest.values())) == len(manifest)  # no photo twice under two names
