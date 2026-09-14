from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from bearing_predictive_maintenance import data


def test_build_download_command_uses_argument_vector(tmp_path: Path) -> None:
    command = data.build_download_command("/venv/bin/kaggle", tmp_path, force=True)

    assert command == [
        "/venv/bin/kaggle",
        "datasets",
        "download",
        "vinayak123tyagi/bearing-dataset",
        "--path",
        str(tmp_path),
        "--unzip",
        "--force",
    ]


def test_download_dataset_runs_kaggle_without_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "bearing_predictive_maintenance.data.shutil.which", lambda _: "/venv/bin/kaggle"
    )

    def fake_run(command: list[str], *, check: bool, text: bool) -> None:
        captured.update(command=command, check=check, text=text)

    monkeypatch.setattr("bearing_predictive_maintenance.data.subprocess.run", fake_run)

    result = data.download_dataset(tmp_path)

    assert result == tmp_path.resolve()
    assert captured == {
        "command": data.build_download_command("/venv/bin/kaggle", tmp_path.resolve()),
        "check": True,
        "text": True,
    }


def test_download_dataset_reports_missing_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("bearing_predictive_maintenance.data.shutil.which", lambda _: None)

    with pytest.raises(data.DatasetDownloadError, match="Kaggle CLI is unavailable"):
        data.download_dataset(tmp_path)


def test_download_dataset_translates_kaggle_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "bearing_predictive_maintenance.data.shutil.which", lambda _: "/venv/bin/kaggle"
    )

    def fail(command: list[str], *, check: bool, text: bool) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr("bearing_predictive_maintenance.data.subprocess.run", fail)

    with pytest.raises(data.DatasetDownloadError, match="Verify authentication"):
        data.download_dataset(tmp_path)
