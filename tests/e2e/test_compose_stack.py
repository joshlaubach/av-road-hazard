"""Runs against a live stack: docker compose -f docker/docker-compose.yml up -d --build api.

Excluded by default. Run with: E2E_BASE_URL=http://localhost:8000 pytest -m e2e
"""

import os

import httpx
import pytest

pytestmark = pytest.mark.e2e
BASE = os.environ.get("E2E_BASE_URL", "http://localhost:8000")


def test_health_and_analyze(frame_bytes: bytes) -> None:
    assert httpx.get(f"{BASE}/health", timeout=10).json()["status"] == "ok"
    r = httpx.post(
        f"{BASE}/analyze", files={"file": ("f.png", frame_bytes, "image/png")}, timeout=30
    )
    assert r.status_code == 200
    assert r.json()["image_size"] == [640, 360]
