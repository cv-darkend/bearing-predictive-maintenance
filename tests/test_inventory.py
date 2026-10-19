from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from bearing_predictive_maintenance import ims, inventory


def _write_snapshot(path: Path, rows: list[list[str]]) -> None:
    content = "\r\n".join("\t".join(row) for row in rows) + "\r\n"
    path.write_bytes(content.encode("ascii"))


@pytest.fixture
def synthetic_test2_dir(tmp_path: Path) -> Path:
    """Build a synthetic raw_root with a few test-2 snapshots (4 channels each)."""
    test_dir = tmp_path / "2nd_test" / "2nd_test"
    test_dir.mkdir(parents=True)
    # first snapshot: 3 well-formed rows
    _write_snapshot(
        test_dir / "2004.02.12.10.32.39",
        [["0.1", "0.2", "0.3", "0.4"] for _ in range(3)],
    )
    # second snapshot, 10 minutes later: one unparsable token
    _write_snapshot(
        test_dir / "2004.02.12.10.42.39",
        [["0.1", "0.2", "0.3", "0.4"], ["bad", "0.2", "0.3", "0.4"]],
    )
    # third snapshot, an unusually large gap later (20 minutes instead of 10)
    _write_snapshot(
        test_dir / "2004.02.12.11.02.39",
        [["0.1", "0.2", "0.3", "0.4"] for _ in range(3)],
    )
    return tmp_path


def test_build_inventory_reports_one_row_per_file(synthetic_test2_dir: Path) -> None:
    result = inventory.build_inventory(synthetic_test2_dir, "2")

    assert list(result["file"]) == [
        "2004.02.12.10.32.39",
        "2004.02.12.10.42.39",
        "2004.02.12.11.02.39",
    ]
    assert (result["test_id"] == "2").all()
    assert list(result["n_rows"]) == [3, 2, 3]
    assert list(result["n_cols"]) == [4, 4, 4]
    assert list(result["n_non_finite"]) == [0, 1, 0]


def test_build_inventory_computes_gap_and_elapsed_time(synthetic_test2_dir: Path) -> None:
    result = inventory.build_inventory(synthetic_test2_dir, "2")

    assert pd.isna(result["gap_seconds"].iloc[0])
    assert result["gap_seconds"].iloc[1] == pytest.approx(600.0)
    assert result["gap_seconds"].iloc[2] == pytest.approx(1200.0)

    assert result["elapsed_hours"].iloc[0] == pytest.approx(0.0)
    assert result["elapsed_hours"].iloc[2] == pytest.approx(1800.0 / 3600.0)


def test_build_inventory_flags_unreadable_file_without_raising(tmp_path: Path) -> None:
    test_dir = tmp_path / "2nd_test" / "2nd_test"
    test_dir.mkdir(parents=True)
    _write_snapshot(test_dir / "2004.02.12.10.32.39", [["0.1", "0.2", "0.3", "0.4"]])
    # a genuinely unreadable file: a directory sharing a valid snapshot name would break
    # iteration, so instead simulate an unreadable file via permission removal is
    # platform-dependent; use an empty file which pandas cannot parse into any columns.
    empty_path = test_dir / "2004.02.12.10.42.39"
    empty_path.write_bytes(b"")

    result = inventory.build_inventory(tmp_path, "2")

    broken_row = result[result["file"] == "2004.02.12.10.42.39"].iloc[0]
    assert broken_row["read_error"] is not None
    assert broken_row["n_rows"] == 0


def test_summarize_inventory_compares_documented_and_observed(synthetic_test2_dir: Path) -> None:
    spec = ims.TEST_REGISTRY["2"]
    inv = inventory.build_inventory(synthetic_test2_dir, "2")

    summary = inventory.summarize_inventory(inv, spec)

    assert summary.test_id == "2"
    assert summary.documented_file_count == 984
    assert summary.observed_file_count == 3
    assert summary.documented_start == datetime(2004, 2, 12, 10, 32, 39)
    assert summary.observed_start == datetime(2004, 2, 12, 10, 32, 39)
    assert summary.observed_end == datetime(2004, 2, 12, 11, 2, 39)
    assert summary.documented_channel_count == 4
    assert summary.observed_channel_count == 4
    assert summary.median_gap_seconds == pytest.approx(900.0)
    assert summary.max_gap_seconds == pytest.approx(1200.0)
    assert summary.gaps_over_1_5x_median == 0
    assert summary.duplicate_timestamp_count == 0


def test_summarize_inventory_counts_duplicate_timestamps(synthetic_test2_dir: Path) -> None:
    spec = ims.TEST_REGISTRY["2"]
    inv = inventory.build_inventory(synthetic_test2_dir, "2")
    duplicated = pd.concat([inv, inv.iloc[[0]]], ignore_index=True)

    summary = inventory.summarize_inventory(duplicated, spec)

    assert summary.duplicate_timestamp_count == 1
