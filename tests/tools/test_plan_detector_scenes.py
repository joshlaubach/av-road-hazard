"""tools/plan_detector_scenes.py: the rule that picks the test scenes, set before any training."""

from __future__ import annotations

from tools.plan_detector_scenes import choose_scenes, plan_text

COUNTS = {
    "scene-0001": (50, 0),  # barriers only
    "scene-0002": (0, 30),  # cones only
    "scene-0003": (5, 5),  # some, but under the limit
    "scene-0004": (0, 0),  # plain road
    "scene-0005": (100, 100),  # the richest
    "scene-0006": (0, 0),  # plain road
    "scene-0007": (0, 0),  # plain road
}


def test_scenes_with_enough_barriers_or_cones_are_chosen_and_the_others_are_not() -> None:
    chosen = choose_scenes(COUNTS, limit=20, plain=0)
    assert set(chosen) == {"scene-0001", "scene-0002", "scene-0005"}


def test_the_chosen_scenes_are_ranked_richest_first_then_by_name() -> None:
    assert choose_scenes(COUNTS, limit=20, plain=0) == ["scene-0005", "scene-0001", "scene-0002"]


def test_plain_road_scenes_are_the_first_ones_by_name_not_picked_by_hand() -> None:
    chosen = choose_scenes(COUNTS, limit=20, plain=2)
    assert chosen[-2:] == ["scene-0004", "scene-0006"] and "scene-0007" not in chosen


def test_a_scene_under_the_limit_is_never_added_as_plain_road() -> None:
    assert "scene-0003" not in choose_scenes(COUNTS, limit=20, plain=10)


def test_the_plan_text_is_a_yaml_table_that_load_fold_plan_accepts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from tools.finetune_detector import load_fold_plan

    path = tmp_path / "plan.yaml"
    path.write_text(plan_text(["scene-0005", "scene-0001", "scene-0002"], 2, {}), encoding="utf-8")
    assert load_fold_plan(path) == {"scene-0005": 0, "scene-0001": 1, "scene-0002": 0}
