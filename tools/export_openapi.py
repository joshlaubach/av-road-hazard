"""Write the API's OpenAPI document to openapi.json. The contract test checks it stays current."""

from __future__ import annotations

import json
from pathlib import Path

from api.main import app

TARGET = Path(__file__).resolve().parent.parent / "openapi.json"


def render() -> str:
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    TARGET.write_text(render())
