"""Training on cached tiles: the dataset, the pools, the image-level score, fit() and tracking."""

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from defects.dataset import Sample, build_dataset_manifest
from defects.splits import Splits, write_splits
from defects.tile_cache import build_tile_cache
from defects.tile_dataset import TileDataset, check_cached, image_scores, tile_pools
from defects.tile_index import (
    TileIndex,
    TileParams,
    entry_from_tiles,
    label_image_tiles,
    training_pool,
    write_tile_index,
)
from defects.train import fit, manifest_path

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("timm")
pytestmark = pytest.mark.ml

ROOT = Path(__file__).resolve().parents[2]
SIDE = 16
PARAMS = TileParams(tile_size=SIDE, overlap=0.0, positive_overlap=0.25, negative_overlap=0.05)
TileCase = tuple[list[Sample], Path, dict[str, Any], TileIndex]


@pytest.fixture
def tile_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TileCase:
    """12 images of 32x32 = four 16 px tiles; even images have a D00 box on their first tile."""
    torch.set_num_threads(1)
    cfg = yaml.safe_load((ROOT / "configs/defects_train.yaml").read_text())
    cache_dir = tmp_path / "cache"
    index_file = tmp_path / "tiles.json"
    cfg.update(
        checkpoint_dir=str(tmp_path / "weights"),
        pretrained=False,
        input_size=SIDE,
        epochs=2,
        patience=2,
        batch_size=2,
        lr=0.01,
        horizontal_flip=0.0,
        color_jitter=0.0,
        head_dropout=0.0,
        drop_path=0.0,
        num_workers=0,
        tile_size=SIDE,
        tile_overlap=0.0,
        tile_index=str(index_file),
        tile_cache_dir=str(cache_dir),
        progress=False,
    )
    samples, images = [], {}
    for i in range(12):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (32, 32), (20 + 15 * i, 40, 200 - 15 * i)).save(path)
        boxes = [("D00", 0, 0, SIDE, SIDE)] if i % 2 == 0 else []
        tiles = label_image_tiles(str(i), (32, 32), boxes, PARAMS)
        images[str(i)] = entry_from_tiles((32, 32), tiles)
        samples.append(Sample(str(i), path, (1 - i % 2, 0, 0, 0)))
    index = TileIndex(PARAMS, images)
    write_tile_index(index, index_file)
    ids = [s.sample_id for s in samples]
    splits_file = tmp_path / "splits.json"
    folds = [(ids[:6], ids[6:8], ids[8:11]), (ids[8:10], ids[10:11], ids[:8])]
    write_splits(Splits([ids[11]], folds), splits_file)
    manifest_path(splits_file).write_text(json.dumps(build_dataset_manifest(samples)))
    pool = training_pool(index, ids, clean_per_positive=1.0, min_clean=1, seed=42)
    paths = {s.sample_id: s.image_path for s in samples}
    build_tile_cache(pool, paths, cache_dir, PARAMS, quality=95)

    def tiny_model(_: dict[str, Any]) -> Any:
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)
        )

    monkeypatch.setattr("defects.train.build_model", tiny_model)
    return samples, splits_file, cfg, index


def test_dataset_item_has_the_model_input_shape_and_the_tile_labels(tile_case: TileCase) -> None:
    _, _, cfg, index = tile_case
    tiles = tile_pools(index, ["0", "1"], ["2"], cfg, seed=42)[0]
    data = TileDataset(tiles, Path(cfg["tile_cache_dir"]), cfg, training=False)
    assert len(data) == len(tiles)
    for k, tile in enumerate(tiles):
        image, labels = data[k]
        assert tuple(image.shape) == (3, SIDE, SIDE)
        assert labels.tolist() == list(tile.labels)
    assert any(sum(t.labels) for t in tiles)  # the positive tile is really in the pool


def test_train_and_validation_pools_never_share_an_image(tile_case: TileCase) -> None:
    _, _, cfg, index = tile_case
    train, val = tile_pools(index, ["0", "1", "2", "3"], ["4", "5"], cfg, seed=42)
    assert {t.image_id for t in train} == {"0", "1", "2", "3"}
    assert {t.image_id for t in val} == {"4", "5"}


def test_pools_keep_every_positive_and_cap_the_clean_tiles(tile_case: TileCase) -> None:
    _, _, cfg, index = tile_case
    train, _ = tile_pools(index, ["0", "1"], ["2"], cfg, seed=42)
    by_image = {i: [t for t in train if t.image_id == i] for i in ("0", "1")}
    assert sum(sum(t.labels) for t in by_image["0"]) == 1  # image 0: its one D00 tile
    assert len(by_image["0"]) == 2  # one positive + one clean (clean_per_positive 1.0)
    assert len(by_image["1"]) == 1  # no defect: min_clean_per_image = 1


def test_image_scores_take_the_highest_tile_probability_per_class(tile_case: TileCase) -> None:
    _, _, cfg, index = tile_case
    tiles = tile_pools(index, ["0", "1"], ["2"], cfg, seed=42)[0]
    probabilities = [[0.1 * (k + 1), 0.0, 0.0, 0.5] for k in range(len(tiles))]
    scores = image_scores(tiles, probabilities)
    first = [k for k, t in enumerate(tiles) if t.image_id == "0"]
    assert scores["0"][0] == pytest.approx(max(probabilities[k][0] for k in first))
    assert scores["0"][3] == 0.5
    assert set(scores) == {"0", "1"}


def test_check_cached_names_how_many_tiles_are_missing(tile_case: TileCase, tmp_path: Path) -> None:
    _, _, cfg, index = tile_case
    tiles = tile_pools(index, ["0"], ["1"], cfg, seed=42)[0]
    check_cached(tiles, Path(cfg["tile_cache_dir"]))
    with pytest.raises(ValueError, match=r"2 tiles are missing"):
        check_cached(tiles, tmp_path / "empty")


def test_fit_on_tiles_trains_and_reports_tile_and_image_scores(tile_case: TileCase) -> None:
    samples, splits_file, cfg, _ = tile_case
    result = fit(samples, splits_file, 0, cfg, run_id="tiles", device="cpu")
    assert len(result.history) == 2
    for epoch in result.history:
        assert {"train_loss", "val_loss", "macro_f1", "pool_image_macro_f1"} <= epoch.keys()
        assert all(math.isfinite(v) for v in epoch.values())
    sidecar = json.loads(result.checkpoint.with_suffix(".json").read_text())
    assert sidecar["input"] == "tiles"
    assert sidecar["tile_params"] == asdict(PARAMS)
    assert len(sidecar["tile_index_sha256"]) == 64


def test_fit_without_a_tile_index_still_trains_on_whole_images(tile_case: TileCase) -> None:
    samples, splits_file, cfg, _ = tile_case
    plain = {k: v for k, v in cfg.items() if k not in ("tile_index", "tile_cache_dir")}
    result = fit(samples, splits_file, 0, plain, run_id="images", device="cpu")
    assert "pool_image_macro_f1" not in result.history[0]
    sidecar = json.loads(result.checkpoint.with_suffix(".json").read_text())
    assert sidecar["input"] == "images"


def test_fit_refuses_an_index_built_with_other_tile_settings(tile_case: TileCase) -> None:
    samples, splits_file, cfg, _ = tile_case
    with pytest.raises(ValueError, match="tile parameters"):
        fit(samples, splits_file, 0, {**cfg, "tile_size": 8}, run_id="mismatch", device="cpu")


def test_fit_refuses_when_the_cache_is_missing_tiles(tile_case: TileCase, tmp_path: Path) -> None:
    samples, splits_file, cfg, _ = tile_case
    with pytest.raises(ValueError, match="missing from the tile cache"):
        fit(
            samples,
            splits_file,
            0,
            {**cfg, "tile_cache_dir": str(tmp_path / "nothing")},
            run_id="nocache",
            device="cpu",
        )


def test_fit_draws_no_bar_when_progress_is_off_even_on_a_terminal(
    tile_case: TileCase, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)  # auto mode would draw a bar
    samples, splits_file, cfg, _ = tile_case
    fit(samples, splits_file, 0, {**cfg, "progress": False}, run_id="quiet", device="cpu")
    assert "epoch 1/2" not in capsys.readouterr().err


def test_fit_draws_a_bar_when_progress_is_on(
    tile_case: TileCase, capsys: pytest.CaptureFixture[str]
) -> None:
    samples, splits_file, cfg, _ = tile_case
    fit(samples, splits_file, 0, {**cfg, "progress": True}, run_id="loud", device="cpu")
    assert "epoch 1/2" in capsys.readouterr().err


def test_fit_records_the_run_in_the_tracking_store(tile_case: TileCase, tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    samples, splits_file, cfg, _ = tile_case
    store = tmp_path / "store"
    store.mkdir()
    cfg = {**cfg, "tracking_db": str(store / "mlflow.db"), "tracking_experiment": "tiles"}
    result = fit(samples, splits_file, 0, cfg, run_id="tracked", device="cpu")

    uri = f"sqlite:///{(store / 'mlflow.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("tiles")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "tracked-fold0"
    history = client.get_metric_history(run.info.run_id, "val_loss")
    assert [m.value for m in history] == pytest.approx([m["val_loss"] for m in result.history])
    assert run.data.params["input"] == "tiles"
    assert [a.path for a in client.list_artifacts(run.info.run_id)] == ["best.json"]


def test_cli_resolves_tile_and_tracking_paths_relative_to_the_config(tmp_path: Path) -> None:
    from defects.train import _resolve_optional_paths

    cfg = {"tile_index": "d/tiles.json", "tile_cache_dir": "d/cache", "tracking_db": None}
    resolved = _resolve_optional_paths(cfg, tmp_path)
    assert resolved["tile_index"] == str((tmp_path / "d" / "tiles.json").resolve())
    assert resolved["tile_cache_dir"] == str((tmp_path / "d" / "cache").resolve())
    assert resolved["tracking_db"] is None  # unset stays unset: tracking off
    assert cfg["tile_index"] == "d/tiles.json"  # the input mapping is not mutated


def test_the_real_fold_splits_keep_train_and_validation_images_apart(tile_case: TileCase) -> None:
    from defects.splits import read_splits

    _, splits_file, cfg, index = tile_case
    for train_ids, val_ids, held_out in read_splits(splits_file).folds:
        train, val = tile_pools(index, train_ids, val_ids, cfg, seed=42)
        assert {t.image_id for t in train}.isdisjoint({t.image_id for t in val})
        assert {t.image_id for t in train}.isdisjoint(held_out)
        assert {t.image_id for t in val}.isdisjoint(held_out)


def test_image_scores_reject_rows_of_the_wrong_length(tile_case: TileCase) -> None:
    _, _, cfg, index = tile_case
    tiles = tile_pools(index, ["0"], ["1"], cfg, seed=42)[0]
    with pytest.raises(ValueError, match="one probability row per tile"):
        image_scores(tiles, [[0.0] * 4])
    with pytest.raises(ValueError, match="4 class probabilities"):
        image_scores(tiles, [[0.0] * 3 for _ in tiles])


def test_epoch_metrics_align_images_and_report_those_without_tiles() -> None:
    from defects.tile_index import TileRecord
    from defects.train import _epoch_metrics, _FoldData, _macro_f1

    def tile(image_id: str, col: int, labels: tuple[int, ...]) -> TileRecord:
        return TileRecord(image_id, 0, col, (0, 0, SIDE, SIDE), "positive", labels)

    clean = (0, 0, 0, 0)
    tiles = [tile("a", 0, clean), tile("a", 1, (1, 0, 0, 0)), tile("b", 0, clean)]
    probabilities = [[0.2, 0, 0, 0], [0.9, 0, 0, 0], [0.1, 0, 0, 0]]
    truth = [[float(v) for v in t.labels] for t in tiles]
    by_id = {
        "a": Sample("a", Path("a"), (1, 0, 0, 0)),
        "b": Sample("b", Path("b"), clean),
        "c": Sample("c", Path("c"), (1, 0, 0, 0)),  # no tiles: cannot be scored by the pool
    }
    data = _FoldData(training=None, validation=None, val_tiles=tiles, sidecar={})
    metrics = _epoch_metrics(1.0, 2.0, probabilities, truth, data, by_id, ["a", "b", "c"], 0.5)
    expected = _macro_f1([[1, 0, 0, 0], [0, 0, 0, 0]], [[0.9, 0, 0, 0], [0.1, 0, 0, 0]], 0.5)
    assert metrics["pool_image_macro_f1"] == pytest.approx(expected)
    assert metrics["val_images_unscored"] == 1


def test_fit_refuses_tile_mode_without_a_cache_dir(tile_case: TileCase) -> None:
    samples, splits_file, cfg, _ = tile_case
    with pytest.raises(ValueError, match="Set tile_cache_dir"):
        fit(samples, splits_file, 0, {**cfg, "tile_cache_dir": None}, run_id="nodir", device="cpu")


def test_an_empty_tile_pool_is_a_clear_error(tile_case: TileCase) -> None:
    from defects.train import _fold_data

    samples, _, cfg, _ = tile_case
    by_id = {s.sample_id: s for s in samples}
    with pytest.raises(ValueError, match="empty"):
        _fold_data(by_id, ["0", "1"], [], cfg, 42)


def test_same_seed_gives_identical_tile_training_histories(tile_case: TileCase) -> None:
    samples, splits_file, cfg, _ = tile_case
    first = fit(samples, splits_file, 0, cfg, run_id="same-a", device="cpu")
    second = fit(samples, splits_file, 0, cfg, run_id="same-b", device="cpu")
    assert first.history == second.history


def test_out_of_fold_in_tile_mode_scores_images_tile_by_tile(
    tile_case: TileCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import defects.train as train

    samples, splits_file, cfg, _ = tile_case
    calls: list[tuple[int, int, float]] = []
    real = train.predict_tiled

    def spy(model: Any, held: Any, cfg_: Any, **kwargs: Any) -> Any:
        calls.append((len(held), kwargs["tile_size"], kwargs["overlap"]))
        return real(model, held, cfg_, **kwargs)

    def whole_image_scoring_must_not_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("tile checkpoints must not be scored on resized whole images")

    monkeypatch.setattr(train, "predict_tiled", spy)
    monkeypatch.setattr(train, "predict", whole_image_scoring_must_not_run)
    result = train.out_of_fold(samples, splits_file, cfg, run_id="oof", device="cpu")
    assert set(result) == {s.sample_id for s in samples} - {"11"}
    assert all(len(row) == 4 for row in result.values())
    assert calls == [(3, SIDE, 0.0), (8, SIDE, 0.0)]  # the two folds' held-out sets


def test_a_failure_mid_training_marks_the_tracked_run_failed(
    tile_case: TileCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    samples, splits_file, cfg, _ = tile_case

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("validation exploded")

    monkeypatch.setattr("defects.train._validate", explode)
    store = tmp_path / "store"
    store.mkdir()
    cfg = {**cfg, "tracking_db": str(store / "mlflow.db"), "tracking_experiment": "boom"}
    with pytest.raises(RuntimeError, match="exploded"):
        fit(samples, splits_file, 0, cfg, run_id="boom", device="cpu")
    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{(store / 'mlflow.db').as_posix()}")
    experiment = client.get_experiment_by_name("boom")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FAILED"


def test_progress_bars_are_closed_when_an_epoch_fails() -> None:
    from tqdm import tqdm

    from defects.train import _train_epoch

    class Broken(torch.nn.Module):  # type: ignore[misc,name-defined]
        def forward(self, x: Any) -> Any:
            raise RuntimeError("bad batch")

    loader = [(torch.zeros(2, 3, 4, 4), torch.zeros(2, 4))] * 3
    model = Broken()
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    cfg = {"progress": True, "amp": "none"}
    with pytest.raises(RuntimeError, match="bad batch"):
        _train_epoch(model, loader, optimizer, torch.nn.BCEWithLogitsLoss(), "cpu", cfg, "t")
    assert not list(tqdm._instances)


def test_predict_images_scores_a_tile_checkpoint_tile_by_tile(
    tile_case: TileCase, tmp_path: Path
) -> None:
    from defects.evaluation import read_predictions
    from tools.predict_images import main as predict_images

    samples, splits_file, cfg, _ = tile_case
    checkpoint = fit(samples, splits_file, 0, cfg, run_id="cli", device="cpu").checkpoint
    index = tmp_path / "sample_index.json"
    rows = [
        {"id": s.sample_id, "image_path": s.image_path.name, "labels": list(s.labels)}
        for s in samples
    ]
    index.write_text(json.dumps(rows))
    config = tmp_path / "cfg.yaml"
    config.write_text(
        yaml.safe_dump({**cfg, "tile_size": 8, "tile_overlap": 0.5})
    )  # stale on purpose
    out = tmp_path / "predictions.npz"
    argv = ["--config", str(config), "--checkpoint", str(checkpoint)]
    argv += ["--sample-index", str(index), "--splits", str(splits_file), "--fold", "0"]
    assert predict_images([*argv, "--part", "heldout", "--out", str(out)]) == 0

    result = read_predictions(out)
    assert list(result.ids) == ["8", "9", "10"]
    assert result.meta["unit"] == "image"
    assert result.meta["aggregation"] == "max-over-tiles"
    # The geometry comes from the checkpoint, not from the (here deliberately wrong) config.
    assert (result.meta["tile_size"], result.meta["tile_overlap"]) == (SIDE, 0.0)
