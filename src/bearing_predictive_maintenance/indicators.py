"""Time-domain indicators per snapshot and channel, and the CLI to build them.

Computes classic time-domain condition-monitoring indicators (mean, RMS, kurtosis,
crest factor, ...) for every channel of every snapshot across all three NASA IMS
tests, producing a tidy (long-format) table plus a companion per-file inventory, both
written to ``data/interim/``.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bearing_predictive_maintenance import ims
from bearing_predictive_maintenance.inventory import build_inventory

TIME_DOMAIN_INDICATOR_NAMES = (
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "shape_factor",
    "energy",
)

DATASET_COLUMNS = (
    "test_id",
    "file",
    "timestamp",
    "elapsed_hours",
    "channel",
    "bearing",
    "axis",
    "n_non_finite",
    *TIME_DOMAIN_INDICATOR_NAMES,
)

DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_OUTPUT = Path("data/interim/ims_snapshot_indicators.csv")
DEFAULT_INVENTORY_OUTPUT = Path("data/interim/ims_inventory.csv")


def compute_time_domain_indicators(signal: np.ndarray) -> pd.DataFrame:
    """Compute per-channel time-domain indicators for a 2D ``(rows, channels)`` signal.

    Returns a DataFrame indexed by 1-based channel (matching raw file column order)
    with one column per indicator in ``TIME_DOMAIN_INDICATOR_NAMES``.

    ``mean`` is the DC level of the channel. Every amplitude indicator (``rms``,
    ``peak``, ``peak_to_peak``, ``energy`` and the crest, impulse and shape factors) is
    computed on the mean-removed signal, so a sensor DC offset does not mask the
    vibration: test 1 channels carry an offset of the same size as their vibration. As a
    result ``rms`` equals ``std``.

    ``std`` is the population standard deviation (``ddof=0``): each snapshot is a
    complete, fixed-size population, not a sample. ``kurtosis`` uses Pearson's
    definition, where a normal distribution has kurtosis 3 (not Fisher's "excess"
    kurtosis, which would be 0 for a normal distribution).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError(f"Expected a 2D (rows, channels) array, got shape {signal.shape}.")

    n_channels = signal.shape[1]
    mean = np.mean(signal, axis=0)
    centered = signal - mean
    abs_centered = np.abs(centered)
    mean_abs = np.mean(abs_centered, axis=0)
    std = np.std(signal, axis=0, ddof=0)
    rms = np.sqrt(np.mean(centered**2, axis=0))
    peak = np.max(abs_centered, axis=0)
    peak_to_peak = np.max(signal, axis=0) - np.min(signal, axis=0)
    energy = np.sum(centered**2, axis=0)

    with warnings.catch_warnings():
        # A near-constant channel makes scipy's moment calculation lose precision;
        # the resulting NaN is the correct, expected value for those channels.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        skewness = np.atleast_1d(stats.skew(signal, axis=0, bias=True))
        kurtosis = np.atleast_1d(stats.kurtosis(signal, axis=0, fisher=False, bias=True))

    nan_fill = np.full(n_channels, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        crest_factor = np.divide(peak, rms, out=nan_fill.copy(), where=rms != 0)
        impulse_factor = np.divide(peak, mean_abs, out=nan_fill.copy(), where=mean_abs != 0)
        shape_factor = np.divide(rms, mean_abs, out=nan_fill.copy(), where=mean_abs != 0)

    return pd.DataFrame(
        {
            "mean": mean,
            "std": std,
            "rms": rms,
            "peak": peak,
            "peak_to_peak": peak_to_peak,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "crest_factor": crest_factor,
            "impulse_factor": impulse_factor,
            "shape_factor": shape_factor,
            "energy": energy,
        },
        index=pd.Index(range(1, n_channels + 1), name="channel"),
    )


def _process_file(test_id: str, path: Path) -> pd.DataFrame | None:
    """Compute the tidy indicator rows for one snapshot file.

    Returns ``None`` (rather than raising) when the file cannot be parsed at all, so a
    single corrupt file never aborts the whole dataset build.
    """
    try:
        array = ims.load_snapshot(path)
    except Exception:  # noqa: BLE001 - a corrupt file must be skipped, not crash the run
        return None

    spec = ims.TEST_REGISTRY[test_id]
    channel_lookup = {mapping.channel: mapping for mapping in spec.channel_map}
    channel_indicators = compute_time_domain_indicators(array)
    non_finite_per_channel = np.count_nonzero(~np.isfinite(array), axis=0)
    timestamp = ims.parse_snapshot_timestamp(path.name)

    rows = []
    for channel in channel_indicators.index:
        mapping = channel_lookup.get(channel)
        row: dict[str, object] = {
            "test_id": test_id,
            "file": path.name,
            "timestamp": timestamp,
            "channel": channel,
            "bearing": mapping.bearing if mapping is not None else None,
            "axis": mapping.axis if mapping is not None else None,
            "n_non_finite": int(non_finite_per_channel[channel - 1]),
        }
        row.update(channel_indicators.loc[channel].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def build_indicator_dataset(
    raw_root: Path, test_id: str, max_workers: int | None = None
) -> pd.DataFrame:
    """Build the tidy (long-format) time-domain indicator dataset for one test.

    One row per (file, channel) pair. Files that cannot be parsed are skipped (their
    integrity is separately captured by ``inventory.build_inventory``). Set
    ``max_workers`` above 1 to parallelize across files with ``ProcessPoolExecutor``.
    """
    paths = ims.list_snapshots(raw_root, test_id)
    if not paths:
        return pd.DataFrame(columns=list(DATASET_COLUMNS))

    start = ims.parse_snapshot_timestamp(paths[0].name)

    if max_workers is not None and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(partial(_process_file, test_id), paths))
    else:
        results = [_process_file(test_id, path) for path in paths]

    frames = [frame for frame in results if frame is not None]
    if not frames:
        return pd.DataFrame(columns=list(DATASET_COLUMNS))

    combined = pd.concat(frames, ignore_index=True)
    combined["elapsed_hours"] = (
        pd.to_datetime(combined["timestamp"]) - start
    ).dt.total_seconds() / 3600.0
    return combined[list(DATASET_COLUMNS)]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for the indicator-building CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the per-file inventory and per-channel time-domain indicator tables "
            "for the NASA IMS bearing dataset."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Root directory holding the raw NASA IMS tests (default: {DEFAULT_RAW_ROOT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination CSV for the indicator dataset (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=DEFAULT_INVENTORY_OUTPUT,
        help=f"Destination CSV for the file inventory (default: {DEFAULT_INVENTORY_OUTPUT}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of worker processes for indicator computation (default: sequential).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Build and write the inventory and indicator CSVs for all documented tests."""
    args = build_parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_output.parent.mkdir(parents=True, exist_ok=True)

    inventory_frames = []
    indicator_frames = []
    for test_id in sorted(ims.TEST_REGISTRY):
        print(f"Processing test {test_id}...")
        inventory_frames.append(build_inventory(args.raw_root, test_id))
        indicator_frames.append(
            build_indicator_dataset(args.raw_root, test_id, max_workers=args.max_workers)
        )

    pd.concat(inventory_frames, ignore_index=True).to_csv(args.inventory_output, index=False)
    pd.concat(indicator_frames, ignore_index=True).to_csv(args.output, index=False)
    print(f"Wrote inventory to {args.inventory_output} and indicators to {args.output}")
