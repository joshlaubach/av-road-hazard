"""Data-free checks for the frozen classifier work order."""

from collections import Counter
from pathlib import Path

import pytest

from defects.labels import CODES, NAMES, class_index
from defects.mining import rank_hard_examples
from defects.splits import make_splits, read_splits, write_splits


def test_codes_and_verified_names() -> None:
    assert CODES == ("D00", "D10", "D20", "D40")
    assert [NAMES[code] for code in CODES] == [
        "longitudinal crack",
        "transverse crack",
        "alligator crack",
        "pothole",
    ]
    assert class_index("D40") == 3
    with pytest.raises(ValueError):
        class_index("unknown")


def test_grouped_splits(tmp_path: Path) -> None:
    items = [f"{g}/{i}" for g in range(40) for i in range(3)]
    splits = make_splits(items, lambda x: x.split("/")[0], seed=42)
    assert splits == make_splits(list(reversed(items)), lambda x: x.split("/")[0], seed=42)
    assert abs(len(splits.test) / len(items) - 0.15) < 0.05
    held: Counter[str] = Counter()
    for train, val, out in splits.folds:
        parts = [set(train), set(val), set(out), set(splits.test)]
        assert all(parts)
        assert set.union(*parts) == set(items)
        groups = [{i.split("/")[0] for i in p} for p in parts]
        for a in range(4):
            for b in range(a):
                assert not parts[a] & parts[b]
                assert not groups[a] & groups[b]
        held.update(out)
    assert held == Counter(set(items) - set(splits.test))
    path = tmp_path / "splits.json"
    write_splits(splits, path)
    assert read_splits(path) == splits


def test_invalid_splits() -> None:
    with pytest.raises(ValueError):
        make_splits(["a"] * 20, lambda x: x, seed=42)
    with pytest.raises(ValueError):
        make_splits(["a", "b"], lambda x: x, seed=42)


def test_mining() -> None:
    ids = ["a", "b", "c", "d"]
    true = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    prob = [[0.2, 0, 0, 0], [0.1, 0, 0, 0], [0.8, 0, 0, 0], [0.9, 0, 0, 0]]
    ranked = rank_hard_examples(ids, true, prob, 0.5, allowed_ids=set(ids), test_ids={"final-test"})
    assert [r.sample_id for r in ranked] == ["b", "a", "d", "c"]
    for allowed, test in [(set(ids[1:]), {"final-test"}), (set(ids), {"a"})]:
        with pytest.raises(ValueError):
            rank_hard_examples(ids, true, prob, 0.5, allowed_ids=allowed, test_ids=test)


def test_split_cli_and_required_groups(tmp_path: Path) -> None:
    import json

    import yaml
    from PIL import Image

    from defects.dataset import Sample, verify_dataset_manifest
    from defects.train import main, manifest_path

    cfg = yaml.safe_load(Path("configs/defects_train.yaml").read_text())
    cfg.update(sample_index="samples.json", splits_file="saved/splits.json")
    config_file = tmp_path / "train.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    image = tmp_path / "fake.png"
    Image.new("RGB", (8, 8)).save(image)
    rows = [{"id": str(i), "image_path": "fake.png", "labels": [0, 0, 0, 0]} for i in range(40)]
    (tmp_path / "samples.json").write_text(json.dumps(rows))
    with pytest.raises(SystemExit):
        main(["splits", "--config", str(config_file)])
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({row["id"]: row["id"] for row in rows}))
    args = ["splits", "--config", str(config_file), "--groups", str(groups)]
    assert main(args) == 0
    split_file = tmp_path / "saved/splits.json"
    read_splits(split_file).validate([str(row["id"]) for row in rows])
    manifest = json.loads(manifest_path(split_file).read_text())
    assert (
        verify_dataset_manifest([Sample(str(i), image, (0, 0, 0, 0)) for i in range(40)], manifest)
        == []
    )
    with pytest.raises(ValueError, match="overwrite"):
        main(args)


def test_reject_corrupt_split_membership(tmp_path: Path) -> None:
    from defects.splits import Splits

    bad = Splits(["test"], [(["train"], ["val"], ["test"]), (["val"], ["train"], ["other"])])
    path = tmp_path / "bad.json"
    write_splits(bad, path)
    with pytest.raises(ValueError):
        read_splits(path)


def test_mining_requires_final_test_ids() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        rank_hard_examples(
            ["a"], [[1, 0, 0, 0]], [[0.1, 0, 0, 0]], 0.5, allowed_ids={"a"}, test_ids=set()
        )
