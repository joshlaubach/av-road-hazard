"""What the tracker will and will not log. Needs no MLflow: the guards run with tracking off too."""

from pathlib import Path

import pytest
from PIL import Image

from common.tracking import NullTracker, open_tracker


def test_no_uri_gives_a_tracker_that_does_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With tracking off every call is accepted and nothing is written to disk."""
    sidecar = tmp_path / "best.json"
    sidecar.write_text("{}")
    monkeypatch.chdir(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    with open_tracker(None, experiment="e", run_name="r") as tracker:
        assert isinstance(tracker, NullTracker)
        tracker.log_params({"lr": 0.1})
        tracker.log_metrics({"loss": 1.0}, step=0)
        tracker.log_file(sidecar)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


@pytest.mark.parametrize("suffix", [".json", ".YAML", ".txt", ".csv", ".md", ".log", ".yml"])
def test_plain_text_files_are_allowed(tmp_path: Path, suffix: str) -> None:
    """The allowlisted suffixes pass, whatever their case."""
    path = tmp_path / f"run{suffix}"
    path.write_text("ok")
    NullTracker().log_file(path)


@pytest.mark.parametrize(
    "name", ["tile.jpg", "frame.PNG", "photo.jpeg", "x.webp", "y.bmp", "z.heic", "w.dng", "frame"]
)
def test_images_and_unknown_types_are_refused(tmp_path: Path, name: str) -> None:
    """Anything not on the allowlist is refused, including image formats nobody listed."""
    path = tmp_path / name
    path.write_bytes(b"bytes")
    with pytest.raises(ValueError, match="refusing to log"):
        NullTracker().log_file(path)


def test_image_content_under_a_text_name_is_refused(tmp_path: Path) -> None:
    """A real picture saved as run.json is caught by its content."""
    path = tmp_path / "run.json"
    Image.new("RGB", (4, 4)).save(path, format="PNG")
    with pytest.raises(ValueError, match="content is an image"):
        NullTracker().log_file(path)


def test_a_symlink_is_judged_by_its_target(tmp_path: Path) -> None:
    """best.json -> tile.jpg must not slip through under the link's innocent name."""
    target = tmp_path / "tile.jpg"
    target.write_bytes(b"bytes")
    link = tmp_path / "best.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks need privileges on this machine")
    with pytest.raises(ValueError, match="refusing to log"):
        NullTracker().log_file(link)


def test_directories_and_missing_files_are_refused(tmp_path: Path) -> None:
    """Only existing regular files can be logged."""
    folder = tmp_path / "runs.json"
    folder.mkdir()
    with pytest.raises(ValueError, match="only files"):
        NullTracker().log_file(folder)
    with pytest.raises(FileNotFoundError):
        NullTracker().log_file(tmp_path / "absent.json")
