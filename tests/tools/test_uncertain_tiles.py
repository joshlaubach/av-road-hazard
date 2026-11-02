"""uncertain_tiles: which tiles are near a class threshold, and a balanced, scene-capped pick."""

from collections import Counter

from eval.classifier import Alert
from tools.uncertain_tiles import WINDOW, Candidate, in_window, pick, side


def candidate(
    scene: str, n: int, name: str = "D10", score: float = 0.8, threshold: float = 0.75
) -> Candidate:
    box = (n, n, n + 1, n + 1)
    return Candidate(
        scene, Alert(f"{scene}/f.jpg", box, name, score, False, 1.0), side(score, threshold)
    )


def test_window_is_closed_below_and_open_above() -> None:
    assert in_window(0.75 - WINDOW, 0.75)
    assert in_window(0.75, 0.75)
    assert not in_window(0.75 + WINDOW, 0.75)
    assert not in_window(0.75 - WINDOW - 0.01, 0.75)


def test_side_is_below_until_the_threshold() -> None:
    assert side(0.74, 0.75) == "below"
    assert side(0.75, 0.75) == "above"


def test_pick_caps_tiles_per_scene() -> None:
    pool = [candidate("s1", n) for n in range(10)] + [candidate("s2", n) for n in range(10, 14)]

    chosen = pick(pool, total=20, per_scene=3, seed=0)

    per_scene = Counter(c.scene for c in chosen)
    assert per_scene == {"s1": 3, "s2": 3}


def test_pick_mixes_classes_and_sides_instead_of_filling_from_the_biggest_group() -> None:
    pool = (
        [candidate(f"a{n}", n, "D10", 0.8) for n in range(30)]
        + [candidate(f"b{n}", n + 100, "D10", 0.6) for n in range(30)]
        + [candidate(f"c{n}", n + 200, "D20", 0.7, 0.65) for n in range(3)]
    )

    chosen = pick(pool, total=9, per_scene=1, seed=0)

    groups = Counter((c.alert.name, c.side) for c in chosen)
    assert groups[("D10", "above")] >= 2
    assert groups[("D10", "below")] >= 2
    assert groups[("D20", "above")] >= 2


def test_pick_never_returns_a_tile_twice() -> None:
    same_tile = [candidate("s1", 1, "D10"), candidate("s1", 1, "D20", 0.7, 0.65)]

    assert len(pick(same_tile, total=5, per_scene=5, seed=0)) == 1


def test_pick_is_repeatable_for_a_seed() -> None:
    pool = [candidate(f"s{n}", n) for n in range(40)]

    assert pick(pool, total=10, per_scene=1, seed=3) == pick(pool, total=10, per_scene=1, seed=3)
