"""Write the API's OpenAPI document to openapi.json (or --check against the committed copy)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from api.main import app

TARGET = Path(__file__).resolve().parent.parent / "openapi.json"


def render() -> str:
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if openapi.json is out of date")
    args = parser.parse_args()
    current = render()
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != current:
            print("openapi.json is out of date: run `python -m tools.export_openapi`")
            return 1
        return 0
    TARGET.write_text(current)
    return 0


if __name__ == "__main__":
    sys.exit(main())
