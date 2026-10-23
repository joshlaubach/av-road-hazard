"""The MLflow wrapper against a real local store: what it records and where it may write."""

from pathlib import Path
from typing import Any

import pytest

mlflow = pytest.importorskip("mlflow")

from common.tracking import REPO_ROOT, open_tracker  # noqa: E402


def sqlite_uri(folder: Path) -> str:
    return f"sqlite:///{(folder / 'mlflow.db').as_posix()}"


def only_run(uri: str, experiment: str = "exp") -> Any:
    client = mlflow.MlflowClient(tracking_uri=uri)
    found = client.get_experiment_by_name(experiment)
    assert found is not None
    (run,) = client.search_runs([found.experiment_id])
    return run


def test_metrics_and_params_reach_the_store(tmp_path: Path) -> None:
    """A run writes its params and a metric history, and ends FINISHED."""
    uri = sqlite_uri(tmp_path)
    with open_tracker(uri, experiment="exp", run_name="run1") as tracker:
        tracker.log_params({"lr": 0.001, "model_name": "efficientnet_b4"})
        tracker.log_metrics({"val_loss": 0.9}, step=0)
        tracker.log_metrics({"val_loss": 0.7}, step=1)

    run = only_run(uri)
    assert run.info.status == "FINISHED"
    assert run.data.params["lr"] == "0.001"
    history = mlflow.MlflowClient(tracking_uri=uri).get_metric_history(run.info.run_id, "val_loss")
    assert [m.value for m in history] == [0.9, 0.7]


def test_long_params_are_truncated(tmp_path: Path) -> None:
    """A huge value (a long path list) is cut to fit MLflow's limit instead of failing the run."""
    uri = sqlite_uri(tmp_path)
    with open_tracker(uri, experiment="exp", run_name="r") as tracker:
        tracker.log_params({"paths": "x" * 20000})
    assert len(only_run(uri).data.params["paths"]) == 5000


def test_default_artifacts_go_next_to_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an explicit root, artifacts land in <db folder>/artifacts and nothing in the cwd."""
    store, cwd = tmp_path / "store", tmp_path / "cwd"
    store.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    sidecar = tmp_path / "best.json"
    sidecar.write_text("{}")
    with open_tracker(sqlite_uri(store), experiment="exp", run_name="r") as tracker:
        tracker.log_file(sidecar)
    assert list((store / "artifacts").rglob("best.json"))
    assert list(cwd.iterdir()) == []


def test_run_file_is_listed_on_the_run(tmp_path: Path) -> None:
    """A checkpoint sidecar JSON attached to the run is visible through the client."""
    uri = sqlite_uri(tmp_path)
    sidecar = tmp_path / "best.json"
    sidecar.write_text("{}")
    with open_tracker(uri, experiment="exp", run_name="r") as tracker:
        tracker.log_file(sidecar)
    run = only_run(uri)
    artifacts = mlflow.MlflowClient(tracking_uri=uri).list_artifacts(run.info.run_id)
    assert [a.path for a in artifacts] == ["best.json"]


def test_images_are_not_stored(tmp_path: Path) -> None:
    """A refused image leaves the run without artifacts."""
    uri = sqlite_uri(tmp_path)
    image = tmp_path / "tile.jpg"
    image.write_bytes(b"bytes")
    with open_tracker(uri, experiment="exp", run_name="r") as tracker, pytest.raises(ValueError):
        tracker.log_file(image)
    run = only_run(uri)
    assert mlflow.MlflowClient(tracking_uri=uri).list_artifacts(run.info.run_id) == []


def test_a_failing_run_is_marked_failed(tmp_path: Path) -> None:
    """An exception ends the run FAILED and still propagates."""
    uri = sqlite_uri(tmp_path)
    with pytest.raises(RuntimeError), open_tracker(uri, experiment="exp", run_name="r"):
        raise RuntimeError("boom")
    assert only_run(uri).info.status == "FAILED"


def test_ctrl_c_is_marked_killed(tmp_path: Path) -> None:
    """An interrupt is not a failure of the run."""
    uri = sqlite_uri(tmp_path)
    with pytest.raises(KeyboardInterrupt), open_tracker(uri, experiment="exp", run_name="r"):
        raise KeyboardInterrupt
    assert only_run(uri).info.status == "KILLED"


def test_a_bookkeeping_failure_does_not_hide_the_real_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If closing the run fails, the training error is still the one the caller sees."""

    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("store locked")

    monkeypatch.setattr(mlflow.MlflowClient, "set_terminated", broken)
    with (
        pytest.raises(RuntimeError, match="boom"),
        open_tracker(sqlite_uri(tmp_path), experiment="exp", run_name="r"),
    ):
        raise RuntimeError("boom")


def test_a_relative_database_in_the_repo_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """sqlite:///mlflow.db resolves against the cwd; inside the repo it is rejected up front."""
    monkeypatch.chdir(REPO_ROOT)
    with (
        pytest.raises(ValueError, match="outside the repository"),
        open_tracker("sqlite:///mlflow.db", experiment="exp", run_name="r"),
    ):
        pass
    assert not (REPO_ROOT / "mlflow.db").exists()


def test_an_artifact_root_in_the_repo_is_refused(tmp_path: Path) -> None:
    """An explicit root inside the repository is rejected even when the database is elsewhere."""
    with (
        pytest.raises(ValueError, match="outside the repository"),
        open_tracker(
            sqlite_uri(tmp_path),
            experiment="exp",
            run_name="r",
            artifact_root=REPO_ROOT / "artifacts",
        ),
    ):
        pass


def test_an_existing_experiment_located_in_the_repo_is_refused(tmp_path: Path) -> None:
    """An older experiment created with an in-repo artifact location cannot be reused."""
    uri = sqlite_uri(tmp_path)
    mlflow.MlflowClient(tracking_uri=uri).create_experiment(
        "exp", artifact_location=(REPO_ROOT / "stray").as_uri()
    )
    with (
        pytest.raises(ValueError, match="outside the repository"),
        open_tracker(uri, experiment="exp", run_name="r"),
    ):
        pass
    assert not (REPO_ROOT / "stray").exists()


@pytest.mark.parametrize("uri", ["file:./mlruns", "mlruns", "ftp://host/x"])
def test_other_store_kinds_are_refused(uri: str) -> None:
    """Only sqlite (checked) and http(s) servers are accepted; file stores default into the cwd."""
    with (
        pytest.raises(ValueError, match="tracking uri must be"),
        open_tracker(uri, experiment="exp", run_name="r"),
    ):
        pass
