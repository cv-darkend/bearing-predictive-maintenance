from __future__ import annotations

import hashlib
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
    good_row = result[result["file"] == "2004.02.12.10.32.39"].iloc[0]
    assert isinstance(broken_row["read_error"], str) and broken_row["read_error"]
    assert pd.isna(good_row["read_error"])
    assert broken_row["n_rows"] == 0

    summary = inventory.summarize_inventory(result, ims.TEST_REGISTRY["2"])
    assert summary.read_error_count == 1


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


def test_summarize_inventory_reports_integrity_counts(synthetic_test2_dir: Path) -> None:
    test_dir = synthetic_test2_dir / "2nd_test" / "2nd_test"
    # a readable file with the wrong channel count (3 instead of the documented 4)
    _write_snapshot(test_dir / "2004.02.12.11.12.39", [["0.1", "0.2", "0.3"] for _ in range(3)])
    inv = inventory.build_inventory(synthetic_test2_dir, "2")

    summary = inventory.summarize_inventory(inv, ims.TEST_REGISTRY["2"])

    assert summary.read_error_count == 0
    assert summary.non_finite_file_count == 1
    assert summary.non_finite_value_count == 1
    assert summary.channel_count_mismatch_count == 1
    assert summary.rows_per_file_consistent is False


def test_summarize_inventory_counts_gaps_over_threshold(tmp_path: Path) -> None:
    test_dir = tmp_path / "2nd_test" / "2nd_test"
    test_dir.mkdir(parents=True)
    row = [["0.1", "0.2", "0.3", "0.4"]]
    for name in (
        "2004.02.12.10.00.00",
        "2004.02.12.10.10.00",
        "2004.02.12.10.20.00",
        "2004.02.12.11.20.00",
        "2004.02.12.11.30.00",
    ):
        _write_snapshot(test_dir / name, row)
    inv = inventory.build_inventory(tmp_path, "2")

    summary = inventory.summarize_inventory(inv, ims.TEST_REGISTRY["2"])

    # gaps: 600, 600, 3600, 600 -> median 600, one gap above 1.5x median
    assert summary.median_gap_seconds == pytest.approx(600.0)
    assert summary.gaps_over_1_5x_median == 1
    assert summary.rows_per_file_consistent is True


def test_inventory_counts_samples_at_the_acquisition_limit(tmp_path: Path) -> None:
    test_dir = tmp_path / "2nd_test" / "2nd_test"
    test_dir.mkdir(parents=True)
    _write_snapshot(test_dir / "2004.02.12.10.32.39", [["0.1", "0.2", "0.3", "0.4"]] * 3)
    _write_snapshot(
        test_dir / "2004.02.12.10.42.39",
        [["4.999", "0.2", "0.3", "0.4"], ["0.1", "-4.999", "0.3", "4.98"], ["0.1"] * 4],
    )
    inv = inventory.build_inventory(tmp_path, "2")

    summary = inventory.summarize_inventory(inv, ims.TEST_REGISTRY["2"])

    # 4.98 is below the limit; only the two +/-4.999 samples are saturated.
    assert list(inv["n_saturated"]) == [0, 2]
    assert summary.saturated_file_count == 1
    assert summary.saturated_value_count == 2


def test_inventory_hashes_file_content_and_counts_duplicates(tmp_path: Path) -> None:
    test_dir = tmp_path / "2nd_test" / "2nd_test"
    test_dir.mkdir(parents=True)
    rows = [["0.1", "0.2", "0.3", "0.4"]] * 3
    _write_snapshot(test_dir / "2004.02.12.10.32.39", rows)
    _write_snapshot(test_dir / "2004.02.12.10.42.39", [["0.5", "0.2", "0.3", "0.4"]] * 3)
    # same bytes as the first file under a different name
    _write_snapshot(test_dir / "2004.02.12.10.52.39", rows)
    inv = inventory.build_inventory(tmp_path, "2")

    summary = inventory.summarize_inventory(inv, ims.TEST_REGISTRY["2"])

    expected = hashlib.sha256((test_dir / "2004.02.12.10.32.39").read_bytes()).hexdigest()
    assert inv["content_sha256"].iloc[0] == expected
    assert inv["content_sha256"].iloc[2] == expected
    assert inv["content_sha256"].iloc[1] != expected
    assert summary.duplicate_content_count == 1


def test_inventory_reports_no_content_duplicates_for_distinct_files(tmp_path: Path) -> None:
    test_dir = tmp_path / "2nd_test" / "2nd_test"
    test_dir.mkdir(parents=True)
    _write_snapshot(test_dir / "2004.02.12.10.32.39", [["0.1", "0.2", "0.3", "0.4"]] * 3)
    _write_snapshot(test_dir / "2004.02.12.10.42.39", [["0.2", "0.2", "0.3", "0.4"]] * 3)
    inv = inventory.build_inventory(tmp_path, "2")

    summary = inventory.summarize_inventory(inv, ims.TEST_REGISTRY["2"])

    assert inv["content_sha256"].str.len().eq(64).all()
    assert summary.duplicate_content_count == 0


def test_inventory_unreadable_files_get_empty_hash_and_are_not_duplicates(
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "2nd_test" / "2nd_test"
    test_dir.mkdir(parents=True)
    _write_snapshot(test_dir / "2004.02.12.10.32.39", [["0.1", "0.2", "0.3", "0.4"]])
    (test_dir / "2004.02.12.10.42.39").write_bytes(b"")
    (test_dir / "2004.02.12.10.52.39").write_bytes(b"")
    inv = inventory.build_inventory(tmp_path, "2")

    summary = inventory.summarize_inventory(inv, ims.TEST_REGISTRY["2"])

    assert list(inv["content_sha256"].iloc[1:]) == ["", ""]
    assert summary.read_error_count == 2
    assert summary.duplicate_content_count == 0


def test_inventory_columns_place_content_hash_after_saturation_count() -> None:
    columns = list(inventory.INVENTORY_COLUMNS)

    assert columns.index("content_sha256") == columns.index("n_saturated") + 1


def test_summarize_empty_inventory_reports_no_content_duplicates() -> None:
    empty = pd.DataFrame(columns=list(inventory.INVENTORY_COLUMNS))

    summary = inventory.summarize_inventory(empty, ims.TEST_REGISTRY["2"])

    assert summary.duplicate_content_count == 0
