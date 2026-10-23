"""Experiment tracking with a local MLflow store, off unless a tracking URI is configured.

Datasets here (nuScenes, RDD2022) must stay on this machine and out of git, so this module fails
closed: the store and its artifacts must live outside the repository, and log_file accepts only
plain text files (an allowlist, checked on the resolved path and by content), never images.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

REPO_ROOT = Path(__file__).resolve().parents[1]
SQLITE_PREFIX = "sqlite:///"
ALLOWED_SUFFIXES = frozenset({".json", ".yaml", ".yml", ".txt", ".csv", ".md", ".log"})
MAX_PARAM_CHARS = 5000  # MLflow rejects longer param values


class Tracker(Protocol):
    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None: ...
    def log_file(self, path: Path) -> None: ...


def _check_loggable(path: Path) -> Path:
    """The resolved path of a plain text file that may be logged; anything else is an error."""
    resolved = path.resolve(strict=True)  # follows symlinks, so a link to an image is judged by it
    if not resolved.is_file():
        raise ValueError(f"only files can be logged, not {path.name}")
    if resolved.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"refusing to log {path.name}: only {sorted(ALLOWED_SUFFIXES)} are allowed"
        )
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(resolved):
            pass
    except (UnidentifiedImageError, OSError):
        return resolved
    raise ValueError(f"refusing to log {path.name}: its content is an image")


class NullTracker:
    """Accepts every call and records nothing; used when tracking is off."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        return None

    def log_file(self, path: Path) -> None:
        _check_loggable(path)


class MlflowTracker:
    """One MLflow run, written through an explicit client (no process-wide MLflow state)."""

    def __init__(self, client: Any, run_id: str) -> None:
        self._client = client
        self._run_id = run_id

    def log_params(self, params: Mapping[str, Any]) -> None:
        for key, value in params.items():
            self._client.log_param(self._run_id, key, str(value)[:MAX_PARAM_CHARS])

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        for key, value in metrics.items():
            self._client.log_metric(self._run_id, key, float(value), step=step)

    def log_file(self, path: Path) -> None:
        self._client.log_artifact(self._run_id, str(_check_loggable(path)))


def _require_outside_repo(path: Path, what: str) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"{what} must be outside the repository, got {resolved}")
    return resolved


def _location_to_path(location: str) -> Path | None:
    """A local filesystem path for a file: URI or a bare path; None for remote locations."""
    parsed = urlparse(location)
    if parsed.scheme == "file":
        return Path(url2pathname(unquote(parsed.path)))
    if parsed.scheme in ("",) or len(parsed.scheme) == 1:  # bare path, or a Windows drive letter
        return Path(location)
    return None


def _validated_root(uri: str, artifact_root: Path | None) -> Path | None:
    """Check the store location and return where artifacts go (None for a remote server)."""
    if uri.startswith(SQLITE_PREFIX):
        database = _require_outside_repo(
            Path(uri.removeprefix(SQLITE_PREFIX)), "the MLflow database"
        )
        root = artifact_root if artifact_root is not None else database.parent / "artifacts"
        return _require_outside_repo(root, "the artifact root")
    if urlparse(uri).scheme in ("http", "https"):
        if artifact_root is not None:
            return _require_outside_repo(artifact_root, "the artifact root")
        return None
    raise ValueError(
        f"tracking uri must be sqlite:///<path outside the repo> or http(s), got {uri}"
    )


def _experiment_id(client: Any, name: str, root: Path | None) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        location = root.as_uri() if root is not None else None
        return str(client.create_experiment(name, artifact_location=location))
    local = _location_to_path(str(experiment.artifact_location))
    if local is not None:
        _require_outside_repo(local, f"the artifact location of experiment {name!r}")
    return str(experiment.experiment_id)


@contextmanager
def open_tracker(
    uri: str | None,
    *,
    experiment: str,
    run_name: str,
    artifact_root: Path | None = None,
) -> Iterator[Tracker]:
    """Yield a tracker for one run.

    The run ends FINISHED, KILLED on Ctrl-C, or FAILED on any other exception, which propagates.
    A store or artifact location inside the repository is refused before anything is created.
    """
    if uri is None:
        yield NullTracker()
        return
    import mlflow

    root = _validated_root(uri, artifact_root)
    client = mlflow.MlflowClient(tracking_uri=uri)
    run = client.create_run(_experiment_id(client, experiment, root), run_name=run_name)
    run_id = run.info.run_id
    status = "FINISHED"
    try:
        yield MlflowTracker(client, run_id)
    except KeyboardInterrupt:
        status = "KILLED"
        raise
    except BaseException:
        status = "FAILED"
        raise
    finally:
        # Never hide the real error behind a bookkeeping failure.
        with suppress(Exception):
            client.set_terminated(run_id, status=status)
