# Convenience targets; the underlying commands are listed in AGENTS.md section 4.
.PHONY: lint format typecheck test e2e api openapi lock
lint:
	ruff check . && ruff format --check .
format:
	ruff check . --fix && ruff format .
typecheck:
	python -m mypy
test:
	python -m pytest
e2e:
	E2E_BASE_URL=http://localhost:8000 python -m pytest -m e2e -o addopts=-q
api:
	uvicorn api.main:app --reload --port 8000
openapi:
	python -m tools.export_openapi
lock:
	pip-compile --strip-extras -o requirements.lock pyproject.toml && pip-compile --extra dev --strip-extras -o requirements-dev.lock pyproject.toml
