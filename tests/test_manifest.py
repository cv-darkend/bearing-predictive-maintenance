from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from bearing_predictive_maintenance import ims, manifest


def _timestamps(start: datetime, gaps_s: list[float]) -> list[datetime]:
    """Build timestamps starting at ``start`` separated by the given gaps (seconds)."""
    values = [start]
    for gap in gaps_s:
        values.append(values[-1] + timedelta(seconds=gap))
    return values


def _frames(
    test_id: str,
    timestamps: list[datetime],
    rms_per_file: list[float],
    *,
    n_saturated: list[int] | None = None,
    read_error: list[str | None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a synthetic (inventory, indicators) pair for one 4-channel test.

    Channel ``c`` maps to bearing ``c``; the max-channel RMS of file ``i`` is
    ``rms_per_file[i]`` (carried by channel 1, the other channels are smaller). Files
    with a ``read_error`` get no indicator rows, as in ``indicators.build_indicator_dataset``.
    """
    n_files = len(timestamps)
    saturated = n_saturated if n_saturated is not None else [0] * n_files
    errors = read_error if read_error is not None else [None] * n_files
    start = timestamps[0]
    files = [ts.strftime("%Y.%m.%d.%H.%M.%S") for ts in timestamps]

    inventory = pd.DataFrame(
        {
            "test_id": test_id,
            "file": files,
            "timestamp": timestamps,
            "n_saturated": saturated,
            "read_error": errors,
        }
    )
    rows = []
    for file, ts, rms, error in zip(files, timestamps, rms_per_file, errors, strict=True):
        if error is not None:
            continue
        for channel in range(1, 5):
            rows.append(
                {
                    "test_id": test_id,
                    "file": file,
                    "timestamp": ts,
                    "elapsed_hours": (ts - start).total_seconds() / 3600.0,
                    "channel": channel,
                    "bearing": channel,
                    "axis": "single",
                    "rms": rms if channel == 1 else rms / 10.0,
                }
            )
    return inventory, pd.DataFrame(rows)


def _per_file(result: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-channel manifest to one row per file (channel 1)."""
    return result[result["channel"] == 1].reset_index(drop=True)


def test_operating_hours_caps_each_gap_at_nominal_cadence_and_handles_unsorted_input() -> None:
    start = datetime(2004, 2, 12, 10, 0, 0)
    # Gaps: 600 s, a 2 h stop (counted as 600 s), then 300 s.
    sorted_ts = _timestamps(start, [600.0, 7200.0, 300.0])
    shuffled = pd.Series(
        [sorted_ts[2], sorted_ts[0], sorted_ts[3], sorted_ts[1]], index=[7, 3, 9, 5]
    )

    result = manifest.operating_hours(shuffled)

    expected_by_ts = {
        sorted_ts[0]: 0.0,
        sorted_ts[1]: 600.0 / 3600.0,
        sorted_ts[2]: 1200.0 / 3600.0,
        sorted_ts[3]: 1500.0 / 3600.0,
    }
    # Result is aligned to the input index (not reordered).
    assert list(result.index) == [7, 3, 9, 5]
    for idx, ts in shuffled.items():
        assert result[idx] == pytest.approx(expected_by_ts[ts])

    custom = manifest.operating_hours(pd.Series(sorted_ts), nominal_cadence_s=300.0)
    assert list(custom) == pytest.approx([0.0, 300 / 3600, 600 / 3600, 900 / 3600])


def test_post_shutdown_flag_valid_and_hours_to_end_relative_to_last_valid_capture() -> None:
    start = datetime(2004, 2, 12, 10, 0, 0)
    n_files = 105
    # Uniform 600 s cadence except one 1 h stop before the last running file.
    gaps = [600.0] * (n_files - 1)
    gaps[101] = 3600.0
    timestamps = _timestamps(start, gaps)
    # The last three files are near-zero amplitude: rig stopped (post shutdown).
    rms = [1.0] * (n_files - 3) + [0.05, 0.01, 0.0]
    inventory, indicators = _frames("2", timestamps, rms)

    result = manifest.build_manifest(inventory, indicators)
    files = _per_file(result)

    assert manifest.POST_SHUTDOWN_BASELINE_N == 100
    assert manifest.POST_SHUTDOWN_FRACTION_OF_BASELINE_MEDIAN == 0.1
    assert len(result) == n_files * 4
    assert "label" not in result.columns
    assert list(files["post_shutdown"]) == [False] * (n_files - 3) + [True] * 3
    assert list(files["valid"]) == [True] * (n_files - 3) + [False] * 3
    # The per-file flag is broadcast to every channel row of the file.
    assert result.groupby("file")["post_shutdown"].nunique().max() == 1

    last_valid = files.iloc[n_files - 4]
    assert last_valid["hours_to_end_clock"] == pytest.approx(0.0)
    assert last_valid["hours_to_end_operating"] == pytest.approx(0.0)
    first = files.iloc[0]
    assert first["clock_hours"] == pytest.approx(0.0)
    assert first["operating_hours"] == pytest.approx(0.0)
    # The 1 h stop (gap index 101 -> file 102) counts fully on the clock axis but only
    # one 600 s step on the operating axis.
    assert first["hours_to_end_clock"] == pytest.approx(101 * 600 / 3600)
    assert first["hours_to_end_operating"] == pytest.approx(101 * 600 / 3600)
    after = files.iloc[n_files - 3]  # file 102, right after the 1 h stop
    assert after["clock_hours"] == pytest.approx((101 * 600 + 3600) / 3600)
    assert after["operating_hours"] == pytest.approx(102 * 600 / 3600)
    assert after["hours_to_end_clock"] == pytest.approx(-1.0)
    assert after["hours_to_end_operating"] == pytest.approx(-600 / 3600)
    assert (files.loc[n_files - 3 :, "hours_to_end_clock"] < 0).all()


def test_saturated_file_flag_follows_inventory_n_saturated() -> None:
    timestamps = _timestamps(datetime(2004, 2, 12, 10, 0, 0), [600.0, 600.0])
    inventory, indicators = _frames("2", timestamps, [1.0, 1.0, 1.0], n_saturated=[0, 3, 0])

    files = _per_file(manifest.build_manifest(inventory, indicators))

    assert list(files["saturated_file"]) == [False, True, False]
    # Saturated files are kept (flagged), not excluded.
    assert list(files["valid"]) == [True, True, True]


def test_after_documented_end_flags_captures_past_the_registry_end() -> None:
    end = ims.TEST_REGISTRY["3"].documented_end
    timestamps = [end - timedelta(seconds=600), end, end + timedelta(seconds=600)]
    inventory, indicators = _frames("3", timestamps, [1.0, 1.0, 1.0])

    files = _per_file(manifest.build_manifest(inventory, indicators))

    assert list(files["after_documented_end"]) == [False, False, True]
    assert list(files["valid"]) == [True, True, True]
    assert files.iloc[-1]["hours_to_end_clock"] == pytest.approx(0.0)


def test_read_error_rows_are_flagged_invalid_and_excluded_from_last_valid_capture() -> None:
    timestamps = _timestamps(datetime(2004, 2, 12, 10, 0, 0), [600.0, 600.0])
    inventory, indicators = _frames("2", timestamps, [1.0, 1.0, 1.0])
    # A file that parsed for indicators but carries an inventory read error.
    inventory.loc[2, "read_error"] = "could not convert string to float"

    result = manifest.build_manifest(inventory, indicators)
    files = _per_file(result)

    assert list(files["read_error"]) == [False, False, True]
    assert list(files["valid"]) == [True, True, False]
    assert files.iloc[1]["hours_to_end_clock"] == pytest.approx(0.0)
    assert files.iloc[2]["hours_to_end_clock"] == pytest.approx(-600 / 3600)
    assert result["read_error"].dtype == bool


def test_documented_failure_and_failure_mode_per_bearing() -> None:
    timestamps = _timestamps(datetime(2003, 10, 22, 12, 0, 0), [600.0])
    inventory, indicators = _frames("1", timestamps, [1.0, 1.0])

    result = manifest.build_manifest(inventory, indicators)
    by_bearing = result[result["file"] == result["file"].iloc[0]].set_index("bearing")

    assert by_bearing.loc[3, "documented_failure"]
    assert by_bearing.loc[3, "failure_mode"] == "inner race defect"
    assert by_bearing.loc[4, "documented_failure"]
    assert by_bearing.loc[4, "failure_mode"] == "roller element defect"
    assert not by_bearing.loc[1, "documented_failure"]
    assert by_bearing.loc[1, "failure_mode"] == ""


def test_manifest_is_sorted_and_accepts_integer_test_ids_from_csv() -> None:
    inv_a, ind_a = _frames("3", _timestamps(datetime(2004, 3, 4, 10, 0, 0), [600.0]), [1.0, 1.0])
    inv_b, ind_b = _frames("1", _timestamps(datetime(2003, 10, 22, 12, 0, 0), [600.0]), [1.0, 1.0])
    inventory = pd.concat([inv_a, inv_b], ignore_index=True)
    indicators = pd.concat([ind_a, ind_b], ignore_index=True).iloc[::-1]
    # Round-tripping through CSV turns the test ids into integers.
    inventory["test_id"] = inventory["test_id"].astype(int)
    indicators["test_id"] = indicators["test_id"].astype(int)

    result = manifest.build_manifest(inventory, indicators)

    assert list(result.columns[: len(manifest.MANIFEST_COLUMNS)]) == list(manifest.MANIFEST_COLUMNS)
    assert list(result["test_id"].unique()) == ["1", "3"]
    sorted_view = result.sort_values(["test_id", "timestamp", "channel"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, sorted_view)


@pytest.mark.parametrize(
    ("table", "column"),
    [("inventory", "n_saturated"), ("inventory", "read_error"), ("indicators", "rms")],
)
def test_missing_required_columns_raise_a_clear_error(table: str, column: str) -> None:
    timestamps = _timestamps(datetime(2004, 2, 12, 10, 0, 0), [600.0])
    inventory, indicators = _frames("2", timestamps, [1.0, 1.0])
    if table == "inventory":
        inventory = inventory.drop(columns=[column])
    else:
        indicators = indicators.drop(columns=[column])

    with pytest.raises(ValueError, match=rf"{table}.*{column}.*make indicators"):
        manifest.build_manifest(inventory, indicators)


def test_indicator_file_without_inventory_row_raises_a_clear_error() -> None:
    timestamps = _timestamps(datetime(2004, 2, 12, 10, 0, 0), [600.0])
    inventory, indicators = _frames("2", timestamps, [1.0, 1.0])
    inventory = inventory.iloc[:1]

    with pytest.raises(
        ValueError, match=r"1 indicator file\(s\) have no inventory row.*make indicators"
    ):
        manifest.build_manifest(inventory, indicators)


def test_cli_writes_manifest_csv_and_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    timestamps = _timestamps(datetime(2004, 2, 12, 10, 0, 0), [600.0, 600.0])
    inventory, indicators = _frames("2", timestamps, [1.0, 1.0, 1.0], n_saturated=[0, 2, 0])
    inventory_path = tmp_path / "inventory.csv"
    indicators_path = tmp_path / "indicators.csv"
    inventory.to_csv(inventory_path, index=False)
    indicators.to_csv(indicators_path, index=False)
    output = tmp_path / "processed" / "manifest.csv"

    manifest.main(
        [
            "--inventory",
            str(inventory_path),
            "--indicators",
            str(indicators_path),
            "--output",
            str(output),
        ]
    )

    written = pd.read_csv(output, dtype={"test_id": str, "file": str})
    assert len(written) == 12
    assert set(manifest.MANIFEST_COLUMNS) <= set(written.columns)
    assert written["saturated_file"].sum() == 4
    out = capsys.readouterr().out.strip()
    assert out.count("\n") == 0
    assert str(output) in out
    assert "12 rows" in out
    assert "12 valid" in out
    assert "0 post-shutdown" in out
    assert "4 saturated-file" in out
    assert "0 after-documented-end" in out


def test_cli_defaults() -> None:
    args = manifest.build_parser().parse_args([])

    assert args.inventory == Path("data/interim/ims_inventory.csv")
    assert args.indicators == Path("data/interim/ims_snapshot_indicators.csv")
    assert args.output == Path("data/processed/ims_snapshot_manifest.csv")
