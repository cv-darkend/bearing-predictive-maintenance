"""Signal-level cleaning and segmentation of NASA IMS snapshots ("Mixto" windowing).

The Phase III windowing decision is mixed: the full snapshot
(``ims.EXPECTED_ROWS_PER_SNAPSHOT`` = 20480 samples, about 1 s at 20 kHz) is the unit
of analysis and the window for spectral features, while time-domain features are
computed on its ``N_SEGMENTS`` contiguous, non-overlapping segments of
``SEGMENT_LENGTH`` samples and then summarized back to one row per snapshot with
robust statistics. Signal-level cleaning is per-channel mean (DC) removal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bearing_predictive_maintenance import ims
from bearing_predictive_maintenance.indicators import compute_time_domain_indicators

SEGMENT_LENGTH = 2048
N_SEGMENTS = ims.EXPECTED_ROWS_PER_SNAPSHOT // SEGMENT_LENGTH

if N_SEGMENTS * SEGMENT_LENGTH != ims.EXPECTED_ROWS_PER_SNAPSHOT:  # pragma: no cover
    raise RuntimeError("SEGMENT_LENGTH must tile ims.EXPECTED_ROWS_PER_SNAPSHOT exactly.")

# Robust per-snapshot summaries of the segment table. ``median`` is the default
# central value; ``p90`` (linear-interpolated 90th percentile) and ``max`` capture
# transient, impulsive activity concentrated in a few segments.
SUMMARY_STATISTICS = ("median", "p90", "max")


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """Return ``signal`` with its per-channel mean (DC level) subtracted.

    Accepts a 1D signal or a 2D ``(rows, channels)`` array; for 2D input the mean of
    each column is removed. The input is not modified.
    """
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError(f"Expected a 1D or 2D (rows, channels) array, got shape {array.shape}.")
    result: np.ndarray = array - np.mean(array, axis=0)
    return result


def split_segments(signal: np.ndarray, segment_length: int = SEGMENT_LENGTH) -> np.ndarray:
    """Split a 1D signal into contiguous, non-overlapping segments.

    Returns an array of shape ``(n_segments, segment_length)`` whose row ``k`` holds
    samples ``k * segment_length`` to ``(k + 1) * segment_length - 1``. The signal
    length must be an exact multiple of ``segment_length``: no sample is dropped or
    padded.
    """
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {array.shape}.")
    if segment_length <= 0:
        raise ValueError(f"segment_length must be positive, got {segment_length}.")
    if array.size % segment_length != 0:
        raise ValueError(
            f"Signal length {array.size} is not a multiple of segment_length {segment_length}."
        )
    return array.reshape(-1, segment_length)


def segment_indicators(signal: np.ndarray, segment_length: int = SEGMENT_LENGTH) -> pd.DataFrame:
    """Compute time-domain indicators on each segment of a 1D signal.

    Reuses ``indicators.compute_time_domain_indicators`` by passing the segments as
    columns, so every segment gets exactly the snapshot-level definitions (amplitude
    indicators on the mean-removed segment, population std, Pearson kurtosis).
    Returns one row per segment, indexed by 1-based ``segment`` number.
    """
    segments = split_segments(signal, segment_length)
    table = compute_time_domain_indicators(segments.T)
    table.index = pd.Index(range(1, len(table) + 1), name="segment")
    return table


def summarize_segments(segment_table: pd.DataFrame, statistic: str = "median") -> pd.Series:
    """Reduce a per-segment indicator table to one row of per-snapshot summaries.

    ``statistic`` is one of ``SUMMARY_STATISTICS``: ``median`` (robust central
    value), ``p90`` (90th percentile, linear interpolation) or ``max``. The result is
    indexed by ``<indicator>_<statistic>``, in the table's column order; NaN segment
    values are skipped.
    """
    if statistic == "median":
        values = segment_table.median(axis=0)
    elif statistic == "p90":
        values = segment_table.quantile(0.9, axis=0)
    elif statistic == "max":
        values = segment_table.max(axis=0)
    else:
        raise ValueError(f"Unknown statistic {statistic!r}; expected one of {SUMMARY_STATISTICS}.")
    summary = pd.Series(values, dtype=np.float64)
    summary.index = pd.Index([f"{column}_{statistic}" for column in segment_table.columns])
    return summary
