"""Tile-by-tile scoring of whole images, on a hand-built model whose output is predictable."""

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from defects.dataset import Sample
from defects.tile_predict import predict_tiled

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytestmark = pytest.mark.ml

SIDE = 16


def red_detector() -> Any:
    """Class 0 fires on a bright red channel: ~1 on a white tile, ~0 on a black tile."""
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)
    )
    with torch.no_grad():
        model[2].weight.zero_()
        model[2].weight[0, 0] = 10.0
        model[2].bias.zero_()
    return model


def picture(path: Path, size: tuple[int, int], white: tuple[int, int, int, int] | None) -> Sample:
    image = Image.new("RGB", size, (0, 0, 0))
    if white is not None:
        image.paste((255, 255, 255), white)
    image.save(path)
    return Sample(path.stem, path, (0, 0, 0, 0))


@pytest.fixture
def cfg() -> dict[str, Any]:
    return {
        "input_size": SIDE,
        "batch_size": 3,
        "num_workers": 0,
        "amp": "none",
        "channels_last": False,
    }


def score(samples: list[Sample], cfg: dict[str, Any], model: Any = None, **kw: Any) -> Any:
    return predict_tiled(
        model or red_detector(), samples, cfg, tile_size=SIDE, overlap=0.0, device="cpu", **kw
    )


def test_an_image_scores_high_when_any_one_tile_is_defective(
    tmp_path: Path, cfg: dict[str, Any]
) -> None:
    only_last = picture(tmp_path / "a.png", (32, 32), (16, 16, 32, 32))  # last tile of four
    clean = picture(tmp_path / "b.png", (32, 32), None)
    all_defect = picture(tmp_path / "c.png", (32, 32), (0, 0, 32, 32))
    a, b, c = score([only_last, clean, all_defect], cfg)
    assert a[0] > 0.99 and b[0] < 0.01 and c[0] > 0.99
    assert a[1:] == pytest.approx([0.5, 0.5, 0.5])  # the other classes carry no signal


def test_results_follow_the_order_of_the_samples(tmp_path: Path, cfg: dict[str, Any]) -> None:
    first = picture(tmp_path / "w.png", (32, 32), (0, 0, 32, 32))
    second = picture(tmp_path / "k.png", (32, 32), None)
    forward = score([first, second], cfg)
    backward = score([second, first], cfg)
    assert forward == backward[::-1]


def test_non_square_images_are_covered_by_a_full_grid(tmp_path: Path, cfg: dict[str, Any]) -> None:
    wide = picture(tmp_path / "wide.png", (48, 16), (32, 0, 48, 16))  # defect in the third tile
    assert score([wide], cfg)[0][0] > 0.99


def test_every_tile_is_scored_exactly_once_in_chunks_of_batch_size(
    tmp_path: Path, cfg: dict[str, Any]
) -> None:
    seen: list[int] = []

    class Recorder(torch.nn.Module):  # type: ignore[misc,name-defined]
        def forward(self, x: Any) -> Any:
            seen.append(x.shape[0])
            return torch.zeros(x.shape[0], 4)

    score([picture(tmp_path / "r.png", (32, 32), None)], cfg, Recorder())
    assert seen == [3, 1]  # four tiles, batch_size 3


def test_the_model_is_left_in_eval_mode(tmp_path: Path, cfg: dict[str, Any]) -> None:
    model = red_detector().train()
    score([picture(tmp_path / "e.png", (32, 32), None)], cfg, model)
    assert not model.training
