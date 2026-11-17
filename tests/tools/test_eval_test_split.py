"""tools/eval_test_split.py: the sealed one-time scoring, its guards and what it writes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from common.hashing import file_sha256
from defects.labels import CODES
from defects.tile_index import ImageEntry, TileIndex, TileParams, write_tile_index
from tools import eval_test_split as tool

TEST_IDS = ["test/a", "test/b"]
PARAMS = TileParams(tile_size=4, overlap=0.0, positive_overlap=0.25, negative_overlap=0.05)
THRESHOLDS = {"D00": 0.9, "D10": 0.75, "D20": 0.65, "D40": 0.75}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _entries() -> dict[str, ImageEntry]:
    """Test images a and b, and two train/val images that must never be opened or scored."""
    size = (16, 16)
    return {
        # 2 positives, 1 ignored cell, 13 clean: pooled 2 + 2 clean, all tiles 2 + 13
        "test/a": ImageEntry(
            size, {(0, 0): (1, 0, 0, 0), (0, 1): (0, 1, 0, 0)}, frozenset({(3, 3)})
        ),
        # no positives, 2 ignored cells, 14 clean: pooled 1 clean, all tiles 14
        "test/b": ImageEntry(size, {}, frozenset({(0, 0), (0, 1)})),
        "train/c": ImageEntry(size, {(2, 2): (0, 0, 1, 0)}, frozenset()),
        "train/e": ImageEntry(size, {}, frozenset()),
        "val/d": ImageEntry(size, {(1, 1): (0, 0, 0, 1)}, frozenset({(0, 3)})),
        "val/f": ImageEntry(size, {}, frozenset()),
    }


class Workspace:
    """A temporary git repo holding the committed protocol, plus the sealed inputs beside it."""

    def __init__(self, root: Path) -> None:
        self.repo = root / "repo"
        self.data = root / "data"
        self.repo.mkdir()
        self.data.mkdir()
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "test@example.invalid")
        _git(self.repo, "config", "user.name", "Test")
        self.opened: list[Path] = []
        self.factory_calls: list[Path] = []
        self.batches: list[int] = []
        self.checkpoint = self.data / "best.pt"
        self.checkpoint.write_bytes(b"checkpoint bytes")
        self.checkpoint.with_suffix(".json").write_text(
            json.dumps({"config": {"defects": {"thresholds": {"D00": 0.1}}}})
        )
        self.splits = self.data / "splits.json"
        self.splits.write_text(json.dumps(_splits_raw()))
        self.index = self.data / "tile_index.json"
        write_tile_index(TileIndex(PARAMS, _entries()), self.index)
        self.sample_index = self.data / "sample_index.json"
        self.sample_index.write_text(
            json.dumps(
                [{"id": i, "image_path": f"images/{i.replace('/', '_')}.png"} for i in _entries()]
            )
        )
        self.protocol = self.repo / "eval" / "test_split_protocol.json"
        self.protocol.parent.mkdir()
        self.protocol.write_text(json.dumps(_protocol(self)))
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "protocol")
        self.out = self.repo / "eval" / "test_split_result.json"

    def run(self, **kwargs: Any) -> dict[str, Any]:
        scorer_factory = _fake_loader(self.factory_calls, self.batches, 0.8)
        opener = _fake_opener(self.opened)
        return tool.run(
            protocol_path=self.protocol,
            checkpoint=self.checkpoint,
            splits=self.splits,
            tile_index=self.index,
            sample_index=self.sample_index,
            out=self.out,
            load_scorer=scorer_factory,
            open_image=opener,
            **kwargs,
        )


def _splits_raw() -> dict[str, Any]:
    # (train, val, held out) per fold; each non-test id is held out exactly once.
    return {
        "test": TEST_IDS,
        "folds": [
            [["val/d"], ["val/f"], ["train/c", "train/e"]],
            [["train/c"], ["train/e"], ["val/d", "val/f"]],
        ],
    }


def _protocol(ws: Workspace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_sha256": file_sha256(ws.checkpoint),
        "splits_sha256": file_sha256(ws.splits),
        "tile_index_sha256": file_sha256(ws.index),
        "thresholds": THRESHOLDS,
        "pooled_sampling": {"seed": 42, "clean_per_positive": 1.0, "min_clean_per_image": 1},
        "measurements": ["pooled", "all_tiles"],
        "metrics": ["precision", "recall", "auroc", "flagged", "positives", "tiles"],
        "intervals": "90% Wilson",
        "rule": "Run once. No checkpoint, threshold or config may be chosen or changed.",
        "registered_date": "2026-11-16",
    }


def _fake_loader(calls: list[Path], batches: list[int], value: float) -> Any:
    def load(checkpoint: Path) -> Any:
        calls.append(checkpoint)

        def score(crops: list[Image.Image]) -> np.ndarray:
            batches.append(len(crops))
            return np.full((len(crops), len(CODES)), value)

        return score

    return load


def _fake_opener(opened: list[Path]) -> Any:
    def open_image(path: Path) -> Image.Image:
        opened.append(path)
        return Image.new("RGB", (16, 16), (90, 90, 90))

    return open_image


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path)


def test_a_result_file_that_already_exists_seals_the_split(ws: Workspace) -> None:
    ws.out.parent.mkdir(exist_ok=True)
    ws.out.write_text("{}")
    with pytest.raises(tool.Refusal, match="already exists"):
        ws.run()
    assert ws.factory_calls == [] and ws.opened == []


def test_a_checkpoint_whose_hash_differs_from_the_protocol_is_refused(ws: Workspace) -> None:
    ws.checkpoint.write_bytes(b"other bytes")
    with pytest.raises(tool.Refusal, match="checkpoint"):
        ws.run()
    assert ws.factory_calls == [] and ws.opened == []


def test_a_splits_or_tile_index_with_a_different_hash_is_refused(ws: Workspace) -> None:
    ws.index.write_text(ws.index.read_text() + " ")
    with pytest.raises(tool.Refusal, match="tile_index"):
        ws.run()


def test_an_untracked_protocol_is_refused(ws: Workspace) -> None:
    _git(ws.repo, "rm", "-q", "--cached", str(ws.protocol.relative_to(ws.repo)))
    with pytest.raises(tool.Refusal, match="not tracked"):
        ws.run()


def test_a_protocol_with_uncommitted_changes_is_refused(ws: Workspace) -> None:
    ws.protocol.write_text(ws.protocol.read_text() + "\n")
    with pytest.raises(tool.Refusal, match="uncommitted"):
        ws.run()


def test_a_dirty_tracked_file_elsewhere_in_the_repo_is_refused(ws: Workspace) -> None:
    other = ws.repo / "notes.txt"
    other.write_text("v1")
    _git(ws.repo, "add", "notes.txt")
    _git(ws.repo, "commit", "-q", "-m", "notes")
    other.write_text("v2")
    with pytest.raises(tool.Refusal, match="tracked files"):
        ws.run()


def test_only_test_images_are_opened_and_scored(ws: Workspace) -> None:
    ws.run()
    assert {p.name for p in ws.opened} == {"test_a.png", "test_b.png"}
    assert sum(ws.batches) == 15 + 14  # every non-ignored test tile, scored once


def test_pooled_is_the_training_pool_and_all_tiles_skips_ignored_tiles(ws: Workspace) -> None:
    result = ws.run()
    measures = result["measurements"]
    assert measures["pooled"]["tiles"] == 4 + 1  # positives plus one clean tile per positive, min 1
    assert measures["all_tiles"]["tiles"] == 15 + 14
    assert measures["all_tiles"]["classes"]["D00"]["positives"] == 1
    assert measures["pooled"]["classes"]["D00"]["positives"] == 1


def test_thresholds_come_from_the_protocol_even_when_the_checkpoint_config_says_otherwise(
    ws: Workspace,
) -> None:
    # The sidecar config sets D00 to 0.1 (flags everything); the protocol's 0.9 must win.
    result = ws.run()
    d00 = result["measurements"]["all_tiles"]["classes"]["D00"]
    d10 = result["measurements"]["all_tiles"]["classes"]["D10"]
    assert d00["flagged"] == 0  # scores are 0.8, below 0.9
    assert d10["flagged"] == 29  # 0.8 reaches 0.75
    assert result["provenance"]["thresholds"] == THRESHOLDS


def test_output_holds_numbers_and_hashes_only(ws: Workspace) -> None:
    ws.run()
    text = ws.out.read_text()
    for forbidden in ("/", "\\", ".png", "test_a", "images/"):
        assert forbidden not in text, forbidden
    payload = json.loads(text)
    assert payload["images"] == 2
    assert set(payload["provenance"]["files"]) == {"protocol", "checkpoint", "splits", "tile_index"}


def test_the_output_has_every_metric_and_wilson_intervals_that_bracket_the_rates(
    ws: Workspace,
) -> None:
    result = ws.run()
    for measure in result["measurements"].values():
        for code in CODES:
            row = measure["classes"][code]
            for metric in ("precision", "recall", "auroc", "flagged", "positives", "tiles"):
                assert metric in row
            if row["precision_interval"] is not None:
                low, high = row["precision_interval"]["low90"], row["precision_interval"]["high90"]
                assert low <= row["precision"] <= high


def test_the_scorer_is_called_in_batches_not_once(ws: Workspace) -> None:
    ws.run(batch_size=4)
    assert len(ws.batches) > 1
    assert max(ws.batches) <= 4
    assert sum(ws.batches) == 29


def test_a_second_run_is_refused_once_the_result_exists(ws: Workspace) -> None:
    ws.run()
    with pytest.raises(tool.Refusal, match="already exists"):
        ws.run()


def test_main_exits_nonzero_with_a_message_on_a_refusal(ws: Workspace, capsys: Any) -> None:
    ws.out.parent.mkdir(exist_ok=True)
    ws.out.write_text("{}")
    code = tool.main(
        [
            "--protocol", str(ws.protocol),
            "--checkpoint", str(ws.checkpoint),
            "--splits", str(ws.splits),
            "--tile-index", str(ws.index),
            "--sample-index", str(ws.sample_index),
            "--out", str(ws.out),
        ]
    )  # fmt: skip
    assert code != 0
    assert "already exists" in capsys.readouterr().err


def test_the_command_has_no_force_flag() -> None:
    with pytest.raises(SystemExit):
        tool.main(["--force"])
