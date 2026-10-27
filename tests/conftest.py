from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def frame_bytes() -> bytes:
    """The bytes of a small generated 640x360 PNG, shared by tests that need an image upload."""
    return (FIXTURES / "synthetic_frame.png").read_bytes()


@pytest.fixture(autouse=True)
def fresh_rate_limits() -> None:
    """Tests share one client address; start each with empty request counts."""
    import sys

    if "api.main" not in sys.modules:
        return  # a test that never touches the API needs nothing reset
    app = sys.modules["api.main"].app
    service = app.state.rag()
    service.query_limiter.reset()
    service.index_limiter.reset()
