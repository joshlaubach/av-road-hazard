from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def frame_bytes() -> bytes:
    """The bytes of a small generated 640x360 PNG, shared by tests that need an image upload."""
    return (FIXTURES / "synthetic_frame.png").read_bytes()
