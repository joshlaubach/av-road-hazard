"""finetune_nuscenes: targets and masks, masked loss, freezing, the round sampler, scene folds."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tools.finetune_nuscenes import (
    NewTile,
    RoundSampler,
    check_pins,
    freeze_all_but,
    load_new_tiles,
    masked_bce,
    pick_round,
    pooled_metrics,
    scene_folds,
    set_train_mode,
    targets_for,
)

KEEP = ("blocks.1", "bn2", "classifier")


class Fake(nn.Module):
    """The same parameter-name prefixes as the timm EfficientNet, at toy size."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)) for _ in range(2)]
        )
        self.bn2 = nn.BatchNorm1d(2)
        self.classifier = nn.Linear(2, 4)


def test_a_clean_tile_has_all_four_classes_as_no() -> None:
    assert targets_for("No crack", "") == ((0, 0, 0, 0), (1, 1, 1, 1))
    assert targets_for("Not road", "") == ((0, 0, 0, 0), (1, 1, 1, 1))


def test_a_crack_counts_only_its_named_kinds() -> None:
    assert targets_for("Crack", "longitudinal") == ((1, 0, 0, 0), (1, 0, 0, 0))
    assert targets_for("Crack", "transverse, alligator") == ((0, 1, 1, 0), (0, 1, 1, 0))


@pytest.mark.parametrize(
    ("call", "kinds"), [("Crack", ""), ("Crack", "diagonal"), ("Maybe", ""), ("", "")]
)
def test_bad_labels_fail_loudly(call: str, kinds: str) -> None:
    with pytest.raises(ValueError):
        targets_for(call, kinds)


def write_batch(folder: Path, rows: list[tuple[str, str, str]]) -> tuple[Path, Path]:
    key = {
        f"t{n}": {"frame": f"f{n}.jpg", "box": [0, 0, 8, 8], "classes": {}}
        for n in range(len(rows))
    }
    (folder / "key.json").write_text(json.dumps(key))
    lines = ["tile_id,call,certainty,location,type,notes"]
    lines += [f"t{n},{call},sure,,{kinds}," for n, (call, kinds, _) in enumerate(rows)]
    (folder / "labels.csv").write_text("\n".join(lines) + "\n")
    return folder / "labels.csv", folder / "key.json"


def test_loader_drops_ambiguous_and_attaches_scenes(tmp_path: Path) -> None:
    csv_path, key_path = write_batch(
        tmp_path, [("No crack", "", "s1"), ("Ambiguous", "", "s1"), ("Crack", "pothole", "s2")]
    )
    scenes = {"f0.jpg": "s1", "f1.jpg": "s1", "f2.jpg": "s2"}

    tiles = load_new_tiles(csv_path, key_path, scenes)

    assert [(t.frame, t.scene, t.call) for t in tiles] == [
        ("f0.jpg", "s1", "No crack"),
        ("f2.jpg", "s2", "Crack"),
    ]
    assert tiles[1].target == (0, 0, 0, 1)


def test_loader_rejects_a_blank_call(tmp_path: Path) -> None:
    csv_path, key_path = write_batch(tmp_path, [("", "", "s1")])

    with pytest.raises(ValueError):
        load_new_tiles(csv_path, key_path, {"f0.jpg": "s1"})


def test_masked_loss_ignores_masked_classes_and_matches_plain_bce_when_all_counted() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
    target = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    plain = nn.BCEWithLogitsLoss()(logits, target)

    assert torch.isclose(masked_bce(logits, target, torch.ones(1, 4)), plain)
    flipped = target.clone()
    flipped[0, 3] = 0.0  # a masked class's target must not matter
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    assert torch.isclose(masked_bce(logits, target, mask), masked_bce(logits, flipped, mask))
    assert masked_bce(logits, target, torch.zeros(1, 4)) == 0.0


def test_only_the_kept_prefixes_train_and_frozen_batchnorm_stays_in_eval() -> None:
    model = Fake()

    trainable = freeze_all_but(model, KEEP)
    set_train_mode(model, KEEP)

    assert trainable == {
        n for n, _ in model.named_parameters() if n.startswith(("blocks.1.", "bn2.", "classifier."))
    }
    assert not any(p.requires_grad for n, p in model.named_parameters() if n not in trainable)
    assert model.training  # the head dropout in timm reads the root flag
    assert not model.get_submodule("stem.1").training
    assert not model.get_submodule("blocks.0.1").training
    assert model.get_submodule("blocks.1.1").training
    assert model.bn2.training


def test_an_unknown_prefix_is_an_error() -> None:
    with pytest.raises(ValueError):
        freeze_all_but(Fake(), ("blocks.9",))


def test_a_round_has_every_original_once_and_spreads_the_new_slots_evenly() -> None:
    sampler = RoundSampler(n_orig=1000, n_tiles=7, share=0.01, seed=3)

    order = list(sampler)

    assert len(order) == len(sampler) == 1000 + sampler.n_new
    assert sampler.n_new == round(0.01 * 1000 / 0.99)
    assert sorted(i for i in order if i < 1000) == list(range(1000))
    views = Counter(i for i in order if i >= 1000)
    assert set(views) == set(range(1000, 1007))
    assert max(views.values()) - min(views.values()) <= 1
    assert order != sorted(order)  # shuffled


def test_rounds_are_repeatable_for_a_seed_and_differ_between_rounds() -> None:
    a, b = RoundSampler(100, 5, 0.05, seed=1), RoundSampler(100, 5, 0.05, seed=1)
    first = list(a)
    a.set_round(1)

    assert list(b) == first
    assert list(a) != first


class Indices(Dataset[int]):
    def __len__(self) -> int:
        return 1000 + 7

    def __getitem__(self, i: int) -> int:
        return i


def test_the_loader_consumes_the_sampler_order_including_the_last_partial_batch() -> None:
    sampler = RoundSampler(n_orig=1000, n_tiles=7, share=0.01, seed=3)
    loader = DataLoader(Indices(), batch_size=16, sampler=sampler, shuffle=False, drop_last=False)

    seen = [int(i) for batch in loader for i in batch]

    assert seen == list(sampler)
    assert len(seen) % 16 != 0  # a partial last batch exists and is kept


def test_scene_folds_keep_a_scene_together_and_are_repeatable() -> None:
    scenes = ["a", "a", "b", "c", "c", "c", "d", "e"]

    folds = scene_folds(scenes, k=3, seed=0)

    assert scene_folds(scenes, k=3, seed=0) == folds
    for scene in set(scenes):
        assert len({f for s, f in zip(scenes, folds, strict=True) if s == scene}) == 1
    assert set(folds) == {0, 1, 2}


def test_the_round_is_the_highest_one_that_passes_in_every_fold_or_none() -> None:
    assert pick_round([[True, True, True], [True, True, False]]) == 2
    assert pick_round([[True, False, True], [True, True, True]]) == 3  # highest, not first failure
    assert pick_round([[True, False, False], [True, True, False]]) == 1
    assert pick_round([[False, False, False], [True, True, True]]) is None


def test_a_changed_input_file_stops_the_run(tmp_path: Path) -> None:
    extra = tmp_path / "extra.json"
    extra.write_text("[]")
    index = tmp_path / "index.json"
    index.write_text("{}")
    import hashlib

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    recorded = {"extra_negatives_sha256": sha(extra), "tile_index_sha256": sha(index)}

    check_pins(recorded, extra, index)  # unchanged: passes
    extra.write_text("[1]")
    with pytest.raises(ValueError, match="extra"):
        check_pins(recorded, extra, index)


def tile(call: str, kinds: str = "") -> NewTile:
    target, mask = targets_for(call, kinds)
    return NewTile("f.jpg", (0, 0, 1, 1), "s", call, target, mask)


def test_pooled_metrics_count_false_alarms_on_clean_tiles_and_recall_on_named_classes() -> None:
    thresholds = (0.9, 0.75, 0.65, 0.75)
    tiles = [
        tile("No crack"),
        tile("Not road"),
        tile("Crack", "transverse"),
        tile("Crack", "longitudinal"),
    ]
    baseline = np.array(
        [[0.1, 0.8, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.8, 0.1, 0.1], [0.95, 0.1, 0.1, 0.1]]
    )
    tuned = np.array(
        [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1], [0.5, 0.1, 0.1, 0.1]]
    )

    result = pooled_metrics(tiles, baseline, tuned, thresholds)

    assert result["clean"]["tiles"] == 2
    assert (
        result["clean"]["flagged_baseline"]["count"],
        result["clean"]["flagged_finetuned"]["count"],
    ) == (1, 0)
    assert (
        result["D10"]["found_baseline"]["count"],
        result["D10"]["found_finetuned"]["count"],
    ) == (1, 1)
    assert (
        result["D00"]["found_baseline"]["count"],
        result["D00"]["found_finetuned"]["count"],
    ) == (1, 0)
    assert result["D20"]["positives"] == 0 and result["D40"]["positives"] == 0


def test_every_sample_has_float32_targets_and_masks_so_mixed_batches_stack(tmp_path: Path) -> None:
    """Worker processes allocate the batch from the first sample's dtype; an int target failed."""
    from PIL import Image

    from tools.finetune_nuscenes import MixedDataset

    Image.new("RGB", (16, 16), (200, 200, 200)).save(tmp_path / "f.jpg")
    cfg = {"input_size": 8, "horizontal_flip": 0.0, "color_jitter": 0.0}
    original = [(torch.zeros(3, 8, 8), torch.tensor([0.0, 1.0, 0.0, 0.0]))]
    crack = NewTile("f.jpg", (0, 0, 16, 16), "s", "Crack", *targets_for("Crack", "transverse"))
    data = MixedDataset(original, [tile("No crack"), crack], tmp_path, cfg)

    for index in range(3):  # one original tile, a clean new tile, a crack new tile
        image, target, mask = data[index]
        assert image.shape == (3, 8, 8)
        assert target.dtype == mask.dtype == torch.float32
    assert data[0][2].tolist() == [1.0, 1.0, 1.0, 1.0]  # an original tile counts every class
    assert data[2][2].tolist() == [0.0, 1.0, 0.0, 0.0]  # a Crack counts only what it names


def test_a_replay_only_round_has_every_original_tile_once_and_no_new_slots() -> None:
    sampler = RoundSampler(n_orig=500, n_tiles=0, share=0.005, seed=1)

    order = list(sampler)

    assert sampler.n_new == 0
    assert sorted(order) == list(range(500))
