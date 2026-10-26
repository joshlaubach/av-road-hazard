"""tools/fetch_detector.py: fetch a pinned .pth, verify it, convert it to safetensors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from perception.detector import DETECTOR_SUBDIR, DetectorSpec
from tools import fetch_detector as fetch

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")


def pth_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "src.pth"
    torch.save(torch.nn.Linear(2, 2).state_dict(), path)
    return path.read_bytes()


@pytest.fixture
def payload(tmp_path: Path) -> bytes:
    return pth_bytes(tmp_path)


def spec_for(payload: bytes, tmp_path: Path) -> DetectorSpec:
    """A spec whose hashes are right for payload, including the converted safetensors file."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "ref.pth"
    source.write_bytes(payload)
    converted = tmp_path / "ref.safetensors"
    fetch.convert_to_safetensors(source, converted)
    return DetectorSpec(
        "tiny",
        "https://download.pytorch.org/models/tiny-00000000.pth",
        hashlib.sha256(payload).hexdigest(),
        hashlib.sha256(converted.read_bytes()).hexdigest(),
    )


class FakeDownloader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.calls: list[str] = []

    def __call__(self, url: str, dest: Path) -> None:
        self.calls.append(url)
        dest.write_bytes(self.data)


def target(root: Path, spec: DetectorSpec) -> Path:
    return root / DETECTOR_SUBDIR / spec.filename


def leftovers(root: Path) -> list[str]:
    return [p.name for p in (root / DETECTOR_SUBDIR).iterdir() if p.name.startswith(".fetch_")]


def test_fetch_verifies_converts_and_cleans_up(tmp_path: Path, payload: bytes) -> None:
    spec = spec_for(payload, tmp_path / "ref")
    root = tmp_path / "ckpt"
    entry = fetch.fetch_detector(root, spec, FakeDownloader(payload))
    assert hashlib.sha256(target(root, spec).read_bytes()).hexdigest() == spec.sha256
    assert entry["sha256"] == spec.sha256 and entry["path"] == f"{DETECTOR_SUBDIR}/tiny.safetensors"
    assert leftovers(root) == []


def test_a_present_verified_file_is_not_downloaded_again(tmp_path: Path, payload: bytes) -> None:
    spec = spec_for(payload, tmp_path / "ref")
    root = tmp_path / "ckpt"
    fetch.fetch_detector(root, spec, FakeDownloader(payload))
    again = FakeDownloader(payload)
    fetch.fetch_detector(root, spec, again)
    assert again.calls == []


def test_a_tampered_download_is_rejected_and_nothing_is_installed(
    tmp_path: Path, payload: bytes
) -> None:
    spec = spec_for(payload, tmp_path / "ref")
    root = tmp_path / "ckpt"
    with pytest.raises(fetch.FetchError, match="sha256"):
        fetch.fetch_detector(root, spec, FakeDownloader(payload + b"tamper"))
    assert not target(root, spec).exists() and leftovers(root) == []


def test_a_conversion_that_does_not_match_the_pin_is_rejected(
    tmp_path: Path, payload: bytes
) -> None:
    good = spec_for(payload, tmp_path / "ref")
    bad = DetectorSpec(good.name, good.url, good.source_sha256, "0" * 64)
    root = tmp_path / "ckpt"
    with pytest.raises(fetch.FetchError, match="converted"):
        fetch.fetch_detector(root, bad, FakeDownloader(payload))
    assert not target(root, bad).exists() and leftovers(root) == []


def test_a_download_error_is_reported_as_a_fetch_error(tmp_path: Path, payload: bytes) -> None:
    def boom(url: str, dest: Path) -> None:
        raise OSError("offline")

    with pytest.raises(fetch.FetchError, match="offline"):
        fetch.fetch_detector(tmp_path / "ckpt", spec_for(payload, tmp_path / "ref"), boom)


def test_main_exit_codes_and_manifest_output(
    tmp_path: Path, payload: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = spec_for(payload, tmp_path / "ref")
    specs = {"tiny": spec}
    args = ["--model", "tiny", "--checkpoint-dir", str(tmp_path / "ckpt")]
    assert fetch.main(args, FakeDownloader(payload), specs) == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{") :])["sha256"] == spec.sha256
    assert (
        fetch.main(
            args,
            FakeDownloader(b"junk"),
            {"tiny": DetectorSpec("tiny2", spec.url, "1" * 64, "2" * 64)},
        )
        == 1
    )
    assert fetch.main(["--model", "nope"], FakeDownloader(payload), specs) == 2


def test_dry_run_fetches_nothing(tmp_path: Path, payload: bytes) -> None:
    spec = spec_for(payload, tmp_path / "ref")
    fake = FakeDownloader(payload)
    args = ["--model", "tiny", "--checkpoint-dir", str(tmp_path / "ckpt"), "--dry-run"]
    assert fetch.main(args, fake, {"tiny": spec}) == 0
    assert fake.calls == [] and not (tmp_path / "ckpt").exists()


def test_only_https_urls_are_fetched(tmp_path: Path, payload: bytes) -> None:
    good = spec_for(payload, tmp_path / "ref")
    http = DetectorSpec(
        good.name, "http://example.com/x-00000000.pth", good.source_sha256, good.sha256
    )
    fake = FakeDownloader(payload)
    with pytest.raises(ValueError, match="https"):
        fetch.fetch_detector(tmp_path / "ckpt", http, fake)
    assert fake.calls == []
