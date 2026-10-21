"""Write the API's OpenAPI document to openapi.json. The contract test checks it stays current."""

from __future__ import annotations

import json
from pathlib import Path

from api.main import app

TARGET = Path(__file__).resolve().parent.parent / "openapi.json"


def render() -> str:
    """The app's OpenAPI document as stable text: sorted keys and a trailing newline, so the file
    only changes when the API changes and diffs stay readable."""
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


# Run as `python -m tools.export_openapi` to regenerate openapi.json after changing the API.
if __name__ == "__main__":
    TARGET.write_text(render())
