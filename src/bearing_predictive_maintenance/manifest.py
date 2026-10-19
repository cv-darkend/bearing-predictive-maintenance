"""Per-channel snapshot manifest: cleaning flags and time axes, and the CLI to build it.

Joins the interim per-file inventory (``ims_inventory.csv``) with the per-channel
indicator table (``ims_snapshot_indicators.csv``), both produced by ``make indicators``,
into one row per (test, file, channel). No raw file is re-read. Each row carries:

- cleaning flags: post-shutdown capture (excluded), read error (excluded), saturated
  file (kept, flagged) and capture after the documented test end (kept, flagged);
- time axes: clock hours, operating hours (``operating_hours``, the single shared
  definition) and hours to the test's last valid capture on both axes;
- the documented failure flag and failure mode of the row's bearing.

There is deliberately no label column: the prediction horizon is chosen in Phase V
and derived there from the hours-to-end columns.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from bearing_predictive_maintenance import ims
from bearing_predictive_maintenance.spectral import flag_post_shutdown

# Nominal acquisition cadence of the IMS rig: one snapshot every 10 minutes.
NOMINAL_CADENCE_S = 600.0

# Phase II post-shutdown rule (notebook 01, "Capturas posteriores al apagado"): a file
# is post-shutdown when its max-channel RMS is below this fraction of the median of the
# first ``POST_SHUTDOWN_BASELINE_N`` files of its test.
POST_SHUTDOWN_BASELINE_N = 100
POST_SHUTDOWN_FRACTION_OF_BASELINE_MEDIAN = 0.1

# Columns required from each input table (names follow ``inventory.INVENTORY_COLUMNS``
# and ``indicators.DATASET_COLUMNS``).
REQUIRED_INVENTORY_COLUMNS = ("test_id", "file", "n_saturated", "read_error")
REQUIRED_INDICATOR_COLUMNS = (
    "test_id",
    "file",
    "timestamp",
    "elapsed_hours",
    "channel",
    "bearing",
    "axis",
    "rms",
)

MANIFEST_COLUMNS = (
    "test_id",
    "file",
    "timestamp",
    "channel",
    "bearing",
    "axis",
    "clock_hours",
    "operating_hours",
    "post_shutdown",
    "read_error",
    "saturated_file",
    "after_documented_end",
    "valid",
    "hours_to_end_clock",
    "hours_to_end_operating",
    "documented_failure",
    "failure_mode",
)

DEFAULT_INVENTORY = Path("data/interim/ims_inventory.csv")
DEFAULT_INDICATORS = Path("data/interim/ims_snapshot_indicators.csv")
DEFAULT_OUTPUT = Path("data/processed/ims_snapshot_manifest.csv")


def operating_hours(
    timestamps: pd.Series, nominal_cadence_s: float = NOMINAL_CADENCE_S
) -> pd.Series:
    """Cumulative operating hours of each capture, aligned to the input index.

    Timestamps are ordered chronologically and each gap between consecutive captures
    counts at most ``nominal_cadence_s`` seconds, so rig stops (long gaps between
    recordings) do not add running time. The earliest capture is at 0 hours. This is
    the single shared definition of operating hours for the project.
    """
    parsed = pd.to_datetime(timestamps)
    ordered = parsed.sort_values(kind="stable")
    gaps = ordered.diff().dt.total_seconds().fillna(0.0)
    hours = gaps.clip(upper=nominal_cadence_s).cumsum() / 3600.0
    return hours.reindex(timestamps.index).astype(np.float64)


def _require_columns(frame: pd.DataFrame, required: Sequence[str], table: str) -> None:
    """Raise a ``ValueError`` naming the required columns missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"The {table} table is missing required column(s) {', '.join(missing)}; "
            "regenerate the interim tables with `make indicators`."
        )


def _has_read_error(values: pd.Series) -> pd.Series:
    """True where ``read_error`` holds an error message (not missing or blank)."""
    return values.notna() & (values.astype(str).str.strip() != "")


def _file_flags(files: pd.DataFrame) -> pd.DataFrame:
    """Compute per-file flags and time axes for the files of one test.

    ``files`` holds one row per file, sorted by timestamp, with ``max_rms``,
    ``read_error`` and ``clock_hours``.
    """
    files = files.copy()
    files["post_shutdown"] = flag_post_shutdown(
        files["max_rms"],
        baseline_n=POST_SHUTDOWN_BASELINE_N,
        fraction_of_baseline_median=POST_SHUTDOWN_FRACTION_OF_BASELINE_MEDIAN,
    ).to_numpy(dtype=bool)
    files["operating_hours"] = operating_hours(files["timestamp"])
    files["valid"] = ~files["post_shutdown"] & ~files["read_error"]

    valid = files[files["valid"]]
    if valid.empty:
        files["hours_to_end_clock"] = np.nan
        files["hours_to_end_operating"] = np.nan
    else:
        last = valid.loc[valid["timestamp"].idxmax()]
        files["hours_to_end_clock"] = last["clock_hours"] - files["clock_hours"]
        files["hours_to_end_operating"] = last["operating_hours"] - files["operating_hours"]
    return files


def build_manifest(inventory: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    """Build the per-channel snapshot manifest from the interim inventory and indicators.

    One row per (test_id, file, channel) of ``indicators``, sorted by test_id,
    timestamp and channel, with the columns in ``MANIFEST_COLUMNS``:

    - ``post_shutdown``: ``spectral.flag_post_shutdown`` on the per-file max-channel
      ``rms`` of each test in timestamp order (the Phase II rule);
    - ``read_error``: the inventory reports a read error for the file;
    - ``saturated_file``: the inventory's ``n_saturated`` (samples at or beyond
      ``inventory.SATURATION_LEVEL``) is positive; kept, only flagged;
    - ``after_documented_end``: timestamp later than the test's documented end in
      ``ims.TEST_REGISTRY``; kept, only flagged;
    - ``valid``: neither post-shutdown nor read error;
    - ``clock_hours`` (the indicator ``elapsed_hours``) and ``operating_hours``;
    - ``hours_to_end_clock`` / ``hours_to_end_operating``: hours until the test's last
      valid capture on each axis, negative for captures after it;
    - ``documented_failure`` / ``failure_mode``: the documented failure of the row's
      bearing, ``False`` / ``""`` when it did not fail.

    Raises ``ValueError`` when either table lacks a required column, or when an
    indicator file has no matching inventory row.
    """
    _require_columns(inventory, REQUIRED_INVENTORY_COLUMNS, "inventory")
    _require_columns(indicators, REQUIRED_INDICATOR_COLUMNS, "indicators")

    rows = indicators[list(REQUIRED_INDICATOR_COLUMNS)].copy()
    rows["test_id"] = rows["test_id"].astype(str)
    rows["file"] = rows["file"].astype(str)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"])
    rows = rows.rename(columns={"elapsed_hours": "clock_hours"})

    file_info = inventory[list(REQUIRED_INVENTORY_COLUMNS)].copy()
    file_info["test_id"] = file_info["test_id"].astype(str)
    file_info["file"] = file_info["file"].astype(str)
    file_info["read_error"] = _has_read_error(file_info["read_error"])
    file_info["saturated_file"] = file_info["n_saturated"].fillna(0) > 0
    file_info = file_info.drop(columns=["n_saturated"])

    files = (
        rows.groupby(["test_id", "file"], as_index=False)
        .agg(timestamp=("timestamp", "first"), clock_hours=("clock_hours", "first"))
        .merge(
            rows.groupby(["test_id", "file"], as_index=False)["rms"]
            .max()
            .rename(columns={"rms": "max_rms"}),
            on=["test_id", "file"],
        )
        .merge(file_info, on=["test_id", "file"], how="left", validate="one_to_one")
    )
    unmatched = files["read_error"].isna()
    if unmatched.any():
        examples = ", ".join(
            f"test {t} file {f}"
            for t, f in files.loc[unmatched, ["test_id", "file"]].head(3).values
        )
        raise ValueError(
            f"{int(unmatched.sum())} indicator file(s) have no inventory row (e.g. {examples}); "
            "regenerate the interim tables with `make indicators`."
        )
    files["read_error"] = files["read_error"].astype(bool)
    files["saturated_file"] = files["saturated_file"].astype(bool)

    per_test = []
    for test_id, group in files.groupby("test_id", sort=True):
        flagged = _file_flags(group.sort_values("timestamp", kind="stable"))
        end = ims.TEST_REGISTRY[str(test_id)].documented_end
        flagged["after_documented_end"] = flagged["timestamp"] > pd.Timestamp(end)
        per_test.append(flagged)
    file_flags = pd.concat(per_test, ignore_index=True).drop(
        columns=["timestamp", "clock_hours", "max_rms"]
    )

    result = rows.drop(columns=["rms"]).merge(
        file_flags, on=["test_id", "file"], how="left", validate="many_to_one"
    )

    failures = {
        (test_id, failure.bearing): failure.description
        for test_id, spec in ims.TEST_REGISTRY.items()
        for failure in spec.documented_failures
    }
    failure_mode = [
        failures.get((test_id, int(bearing)), "") if pd.notna(bearing) else ""
        for test_id, bearing in zip(result["test_id"], result["bearing"], strict=True)
    ]
    result["failure_mode"] = failure_mode
    result["documented_failure"] = result["failure_mode"] != ""

    result = result.sort_values(["test_id", "timestamp", "channel"], kind="stable")
    return result[list(MANIFEST_COLUMNS)].reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for the manifest-building CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the per-channel snapshot manifest (cleaning flags and time axes) for "
            "the NASA IMS bearing dataset from the interim inventory and indicator tables."
        )
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
        help=f"Per-file inventory CSV from `make indicators` (default: {DEFAULT_INVENTORY}).",
    )
    parser.add_argument(
        "--indicators",
        type=Path,
        default=DEFAULT_INDICATORS,
        help=f"Per-channel indicator CSV from `make indicators` (default: {DEFAULT_INDICATORS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination CSV for the manifest (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Build and write the snapshot manifest CSV and print a one-line summary."""
    args = build_parser().parse_args(argv)
    id_dtypes = {"test_id": str, "file": str}
    inventory = pd.read_csv(args.inventory, dtype=id_dtypes)
    indicators = pd.read_csv(args.indicators, dtype=id_dtypes)

    result = build_manifest(inventory, indicators)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"Wrote manifest to {args.output}: {len(result)} rows, "
        f"{int(result['valid'].sum())} valid, "
        f"{int(result['post_shutdown'].sum())} post-shutdown, "
        f"{int(result['read_error'].sum())} read-error, "
        f"{int(result['saturated_file'].sum())} saturated-file, "
        f"{int(result['after_documented_end'].sum())} after-documented-end"
    )
