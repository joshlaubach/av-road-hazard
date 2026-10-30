"""tools/trim_nuscenes.py: keep only the scenes whose images are on disk, and nothing dangling."""

from __future__ import annotations

from typing import Any

from tools.trim_nuscenes import scenes_with_images, trim_tables


def _tables() -> dict[str, list[dict[str, Any]]]:
    return {
        "scene": [
            {
                "token": "sc-a",
                "name": "scene-a",
                "log_token": "log-a",
                "first_sample_token": "s-a1",
            },
            {
                "token": "sc-b",
                "name": "scene-b",
                "log_token": "log-b",
                "first_sample_token": "s-b1",
            },
        ],
        "log": [{"token": "log-a"}, {"token": "log-b"}],
        "map": [
            {"token": "m1", "log_tokens": ["log-a", "log-b"]},
            {"token": "m2", "log_tokens": ["log-b"]},
        ],
        "sample": [
            {"token": "s-a1", "scene_token": "sc-a"},
            {"token": "s-b1", "scene_token": "sc-b"},
        ],
        "sample_data": [
            {
                "token": "d-a1",
                "sample_token": "s-a1",
                "ego_pose_token": "e-a",
                "calibrated_sensor_token": "c-a",
                "filename": "samples/CAM_FRONT/a1.jpg",
                "is_key_frame": True,
            },
            {
                "token": "d-b1",
                "sample_token": "s-b1",
                "ego_pose_token": "e-b",
                "calibrated_sensor_token": "c-b",
                "filename": "samples/CAM_FRONT/b1.jpg",
                "is_key_frame": True,
            },
        ],
        "ego_pose": [{"token": "e-a"}, {"token": "e-b"}],
        "calibrated_sensor": [
            {"token": "c-a", "sensor_token": "sen"},
            {"token": "c-b", "sensor_token": "sen"},
        ],
        "sensor": [{"token": "sen"}],
        "sample_annotation": [
            {"token": "n-a", "sample_token": "s-a1", "instance_token": "i-a"},
            {"token": "n-b", "sample_token": "s-b1", "instance_token": "i-b"},
        ],
        "instance": [{"token": "i-a"}, {"token": "i-b"}],
        "category": [{"token": "cat"}],
        "attribute": [{"token": "att"}],
        "visibility": [{"token": "1"}],
    }


def _tokens(rows: list[dict[str, Any]]) -> set[str]:
    return {r["token"] for r in rows}


def test_only_the_wanted_scene_and_what_hangs_off_it_is_kept() -> None:
    out = trim_tables(_tables(), {"scene-a"})
    assert _tokens(out["scene"]) == {"sc-a"} and _tokens(out["sample"]) == {"s-a1"}
    assert _tokens(out["sample_data"]) == {"d-a1"} and _tokens(out["sample_annotation"]) == {"n-a"}
    assert _tokens(out["ego_pose"]) == {"e-a"} and _tokens(out["calibrated_sensor"]) == {"c-a"}
    assert _tokens(out["instance"]) == {"i-a"} and _tokens(out["log"]) == {"log-a"}


def test_shared_lookup_tables_are_kept_whole() -> None:
    out = trim_tables(_tables(), {"scene-a"})
    for name in ("sensor", "category", "attribute", "visibility"):
        assert out[name] == _tables()[name]


def test_maps_lose_the_logs_that_were_dropped_and_vanish_if_none_are_left() -> None:
    out = trim_tables(_tables(), {"scene-a"})
    assert out["map"] == [{"token": "m1", "log_tokens": ["log-a"]}]


def test_the_input_tables_are_not_changed() -> None:
    tables = _tables()
    trim_tables(tables, {"scene-a"})
    assert tables == _tables()


def test_a_scene_counts_as_on_disk_only_if_its_first_front_image_exists(tmp_path: Any) -> None:
    (tmp_path / "samples" / "CAM_FRONT").mkdir(parents=True)
    (tmp_path / "samples" / "CAM_FRONT" / "a1.jpg").write_bytes(b"x")
    assert scenes_with_images(_tables(), tmp_path) == {"scene-a"}
