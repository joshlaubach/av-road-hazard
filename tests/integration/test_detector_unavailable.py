"""When the detector is unavailable the API fails loudly (503) instead of returning lower scores."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import pipeline as pipeline_module
from api.main import app
from perception.detector import DetectorUnavailable

client = TestClient(app)
SECRET = "/secret/path/detector.safetensors: expected sha256 abc, got def"


@pytest.fixture(autouse=True)
def detector_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(frame: np.ndarray) -> Any:
        raise DetectorUnavailable(SECRET)

    monkeypatch.setattr(pipeline_module, "detect", down)


def test_analyze_returns_503_without_leaking_internal_detail(frame_bytes: bytes) -> None:
    response = client.post("/analyze", files={"file": ("f.png", frame_bytes, "image/png")})
    assert response.status_code == 503
    assert response.json() == {"detail": "object detector unavailable"}
    assert "secret" not in response.text


def test_the_full_reason_is_logged(frame_bytes: bytes, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="api"):
        client.post("/analyze", files={"file": ("f.png", frame_bytes, "image/png")})
    assert any(SECRET in record.getMessage() for record in caplog.records)


def test_index_also_returns_503_and_stores_nothing(frame_bytes: bytes) -> None:
    response = client.post("/index", files=[("files", ("f.png", frame_bytes, "image/png"))])
    assert response.status_code == 503
