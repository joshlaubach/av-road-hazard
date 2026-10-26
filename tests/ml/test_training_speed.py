"""Config-gated training speed options on a tiny CPU model and generated images."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from defects.dataset import Sample, build_dataset_manifest
from defects.splits import Splits, write_splits
from defects.train import _loader, fit, load_checkpoint, manifest_path, setup_determinism

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("timm")
pytestmark = pytest.mark.ml

SPEED_OFF = {"amp": "none", "channels_last": False, "fused_optimizer": False, "deterministic": True}
SPEED_ON = {"amp": "bf16", "channels_last": True, "fused_optimizer": True, "deterministic": True}


@pytest.fixture
def case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Sample], Path, dict[str, Any]]:
    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path("configs/defects_train.yaml").read_text())
    cfg.update(
        checkpoint_dir=str(tmp_path / "weights"),
        pretrained=False,
        input_size=16,
        epochs=3,
        patience=3,
        batch_size=4,
        lr=0.01,
        horizontal_flip=0.0,
        color_jitter=0.0,
        head_dropout=0.0,
        drop_path=0.0,
        num_workers=0,
    )
    samples = []
    for i in range(24):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (16, 16), (20 + 8 * i, 40, 200 - 8 * i)).save(path)
        samples.append(Sample(str(i), path, (i % 2, 0, 1 - i % 2, 0)))
    ids = [s.sample_id for s in samples]
    splits = Splits(
        [ids[23]],
        [(ids[:12], ids[12:16], ids[16:23]), (ids[16:21], ids[21:23], ids[:16])],
    )
    splits_file = tmp_path / "splits.json"
    write_splits(splits, splits_file)
    manifest_path(splits_file).write_text(json.dumps(build_dataset_manifest(samples)))

    def tiny_model(_: dict[str, Any]) -> Any:
        # No convolution: oneDNN on some CPUs cannot run a bf16 convolution backward.
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)
        )

    monkeypatch.setattr("defects.train.build_model", tiny_model)
    monkeypatch.setattr("defects.model.build_model", tiny_model)
    return samples, splits_file, cfg


def run(case: tuple[list[Sample], Path, dict[str, Any]], run_id: str, **options: Any) -> Any:
    samples, splits_file, cfg = case
    return fit(samples, splits_file, 0, {**cfg, **options}, run_id=run_id, device="cpu")


def test_defaults_without_the_new_keys_match_the_explicit_off_settings(
    case: tuple[list[Sample], Path, dict[str, Any]],
) -> None:
    """Existing configs and runs must keep their exact behaviour."""
    legacy = {k: v for k, v in case[2].items() if k not in SPEED_OFF}
    plain = fit(case[0], case[1], 0, legacy, run_id="plain", device="cpu")
    explicit = run(case, "explicit", **SPEED_OFF)
    assert plain.history == explicit.history


def test_mixed_precision_channels_last_and_fused_options_train_and_stay_close(
    case: tuple[list[Sample], Path, dict[str, Any]],
) -> None:
    base = run(case, "fp32", **SPEED_OFF)
    fast = run(case, "bf16", **SPEED_ON)
    assert all(abs(v) < 1e9 and v == v for m in fast.history for v in m.values())
    assert fast.history != base.history  # autocast really ran: bf16 changes the numbers a little
    assert fast.history[-1]["val_loss"] == pytest.approx(base.history[-1]["val_loss"], rel=0.05)
    # The checkpoint is plain fp32 weights that the normal loader accepts.
    model = load_checkpoint(fast.checkpoint, {**case[2], **SPEED_OFF})
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.parametrize("bad", ["fp16", "BF16", True, 16])
def test_unsupported_amp_values_are_rejected(
    case: tuple[list[Sample], Path, dict[str, Any]], bad: Any
) -> None:
    with pytest.raises(ValueError, match="amp"):
        run(case, "bad", amp=bad)


def test_determinism_flag_controls_the_global_algorithm_setting() -> None:
    try:
        setup_determinism(1)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        setup_determinism(1, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark
    finally:
        setup_determinism(1)


def test_persistent_workers_are_opt_in_and_ignored_without_workers(
    case: tuple[list[Sample], Path, dict[str, Any]],
) -> None:
    samples, _, cfg = case
    assert not _loader(samples, {**cfg, "num_workers": 1}, training=True).persistent_workers
    on = {**cfg, "num_workers": 1, "persistent_workers": True}
    assert _loader(samples, on, training=True).persistent_workers
    assert not _loader(samples, {**on, "num_workers": 0}, training=True).persistent_workers
