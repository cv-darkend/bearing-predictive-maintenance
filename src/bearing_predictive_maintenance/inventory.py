"""Per-file inventory and integrity checks for a NASA IMS test's raw snapshots.

Builds a per-file table (one row per snapshot) recording shape, non-finite-value
counts, and cadence, then compares the observed values against the documented test
metadata from ``ims.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bearing_predictive_maintenance import ims

# Samples at or beyond this magnitude sit at the acquisition limit: the IMS recordings
# clip at the rails -5.000 and +4.998 (a +/-5 V range), so their true amplitude is
# unknown. Saturation counts use |x| >= SATURATION_LEVEL.
SATURATION_LEVEL = 4.99

INVENTORY_COLUMNS = (
    "test_id",
    "file",
    "timestamp",
    "n_rows",
    "n_cols",
    "n_non_finite",
    "n_saturated",
    "content_sha256",
    "gap_seconds",
    "elapsed_hours",
    "read_error",
)


def _inventory_row(test_id: str, path: Path, timestamp: datetime) -> dict[str, object]:
    """Build one inventory row for a single snapshot file, never raising on bad data.

    ``content_sha256`` is the SHA-256 of the raw file bytes, or an empty string when the
    file cannot be read.
    """
    try:
        array = ims.load_snapshot(path)
    except Exception as error:  # noqa: BLE001 - any parse failure must be flagged, not raised
        return {
            "test_id": test_id,
            "file": path.name,
            "timestamp": timestamp,
            "n_rows": 0,
            "n_cols": 0,
            "n_non_finite": 0,
            "n_saturated": 0,
            "content_sha256": "",
            "read_error": str(error),
        }
    return {
        "test_id": test_id,
        "file": path.name,
        "timestamp": timestamp,
        "n_rows": array.shape[0],
        "n_cols": array.shape[1] if array.ndim == 2 else 0,
        "n_non_finite": int(np.count_nonzero(~np.isfinite(array))),
        "n_saturated": int(np.count_nonzero(np.abs(array) >= SATURATION_LEVEL)),
        "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "read_error": None,
    }


def build_inventory(raw_root: Path, test_id: str) -> pd.DataFrame:
    """Build a per-file inventory table for one test's raw snapshots.

    One row per snapshot file, sorted by timestamp, with shape, non-finite-value
    count, gap to the previous snapshot (seconds, ``NaN`` for the first file), and
    elapsed time since the first snapshot (hours). A file that cannot be parsed at
    all is flagged via ``read_error`` instead of raising.
    """
    paths = ims.list_snapshots(raw_root, test_id)
    rows = [
        _inventory_row(test_id, path, ims.parse_snapshot_timestamp(path.name)) for path in paths
    ]
    frame = pd.DataFrame(rows, columns=list(INVENTORY_COLUMNS[:-2]) + ["read_error"])

    timestamps = pd.to_datetime(frame["timestamp"])
    gap_seconds = timestamps.diff().dt.total_seconds()
    elapsed_hours = (
        (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
        if len(frame)
        else pd.Series(dtype=float)
    )

    frame["gap_seconds"] = gap_seconds
    frame["elapsed_hours"] = elapsed_hours
    return frame[list(INVENTORY_COLUMNS)]


@dataclass(frozen=True, slots=True)
class InventorySummary:
    """Documented-vs-observed comparison for one test's inventory."""

    test_id: str
    documented_file_count: int
    observed_file_count: int
    documented_start: datetime
    observed_start: datetime | None
    documented_end: datetime
    observed_end: datetime | None
    documented_channel_count: int
    observed_channel_count: int | None
    rows_per_file_consistent: bool
    rows_per_file_mode: int | None
    median_gap_seconds: float | None
    gaps_over_1_5x_median: int
    max_gap_seconds: float | None
    duplicate_timestamp_count: int
    read_error_count: int
    non_finite_file_count: int
    non_finite_value_count: int
    channel_count_mismatch_count: int
    saturated_file_count: int
    saturated_value_count: int
    duplicate_content_count: int


def summarize_inventory(inventory: pd.DataFrame, spec: ims.TestSpec) -> InventorySummary:
    """Summarize an inventory table into a documented-vs-observed comparison.

    Cadence gaps are computed relative to the inventory's own observed median gap
    (never a hardcoded cadence), so the "greater than 1.5x median" threshold adapts to
    each test's documented cadence.

    Integrity counts: ``read_error_count`` counts files that could not be parsed,
    ``non_finite_file_count``/``non_finite_value_count`` count files and values that are
    NaN or non-numeric, and ``channel_count_mismatch_count`` counts readable files whose
    column count differs from the documented channel map. ``saturated_file_count``/
    ``saturated_value_count`` count files and samples at the acquisition limit
    (``SATURATION_LEVEL``). ``duplicate_content_count`` counts files whose
    ``content_sha256`` repeats an earlier file's hash (empty hashes of unreadable files
    are ignored).
    """
    observed_file_count = len(inventory)
    if observed_file_count == 0:
        return InventorySummary(
            test_id=spec.test_id,
            documented_file_count=spec.documented_file_count,
            observed_file_count=0,
            documented_start=spec.documented_start,
            observed_start=None,
            documented_end=spec.documented_end,
            observed_end=None,
            documented_channel_count=len(spec.channel_map),
            observed_channel_count=None,
            rows_per_file_consistent=True,
            rows_per_file_mode=None,
            median_gap_seconds=None,
            gaps_over_1_5x_median=0,
            max_gap_seconds=None,
            duplicate_timestamp_count=0,
            read_error_count=0,
            non_finite_file_count=0,
            non_finite_value_count=0,
            channel_count_mismatch_count=0,
            saturated_file_count=0,
            saturated_value_count=0,
            duplicate_content_count=0,
        )

    timestamps = pd.to_datetime(inventory["timestamp"])
    n_cols_mode = inventory["n_cols"][inventory["n_cols"] > 0].mode()
    n_rows_mode = inventory["n_rows"][inventory["n_rows"] > 0].mode()
    rows_per_file_mode = int(n_rows_mode.iloc[0]) if not n_rows_mode.empty else None
    rows_per_file_consistent = bool(
        (inventory.loc[inventory["n_rows"] > 0, "n_rows"] == rows_per_file_mode).all()
        if rows_per_file_mode is not None
        else True
    )

    gaps = inventory["gap_seconds"].dropna()
    median_gap = float(gaps.median()) if not gaps.empty else None
    max_gap = float(gaps.max()) if not gaps.empty else None
    gaps_over_threshold = (
        int((gaps > 1.5 * median_gap).sum()) if median_gap is not None and median_gap > 0 else 0
    )
    duplicate_count = int(timestamps.duplicated().sum())
    readable = inventory["read_error"].isna()
    channel_mismatches = int((readable & (inventory["n_cols"] != len(spec.channel_map))).sum())
    hashes = inventory["content_sha256"].fillna("")
    hashes = hashes[hashes != ""]
    duplicate_content = int(hashes.duplicated().sum())

    return InventorySummary(
        test_id=spec.test_id,
        documented_file_count=spec.documented_file_count,
        observed_file_count=observed_file_count,
        documented_start=spec.documented_start,
        observed_start=timestamps.min().to_pydatetime(),
        documented_end=spec.documented_end,
        observed_end=timestamps.max().to_pydatetime(),
        documented_channel_count=len(spec.channel_map),
        observed_channel_count=int(n_cols_mode.iloc[0]) if not n_cols_mode.empty else None,
        rows_per_file_consistent=rows_per_file_consistent,
        rows_per_file_mode=rows_per_file_mode,
        median_gap_seconds=median_gap,
        gaps_over_1_5x_median=gaps_over_threshold,
        max_gap_seconds=max_gap,
        duplicate_timestamp_count=duplicate_count,
        read_error_count=int((~readable).sum()),
        non_finite_file_count=int((inventory["n_non_finite"] > 0).sum()),
        non_finite_value_count=int(inventory["n_non_finite"].sum()),
        channel_count_mismatch_count=channel_mismatches,
        saturated_file_count=int((inventory["n_saturated"] > 0).sum()),
        saturated_value_count=int(inventory["n_saturated"].sum()),
        duplicate_content_count=duplicate_content,
    )
