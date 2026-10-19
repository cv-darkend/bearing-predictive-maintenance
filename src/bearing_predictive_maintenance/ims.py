"""Metadata registry and snapshot loading for the NASA IMS bearing dataset.

The dataset ships as three run-to-failure "tests", each a directory of tab-separated
snapshot files named after the timestamp at which they were recorded
(``YYYY.MM.DD.HH.MM.SS``, no extension). This module encodes the documented metadata
from the NASA IMS readme (test layout, channel-to-bearing mapping, documented failures,
bearing geometry) and provides the primitives to parse snapshot timestamps, enumerate a
test's snapshots, and load one snapshot as an array.

Documented values (from the NASA IMS readme) are recorded as *documented*, never as
observed truth: they must be cross-checked against the actual raw files (see
``inventory.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_TIMESTAMP_FORMAT = "%Y.%m.%d.%H.%M.%S"

SAMPLING_RATE_HZ = 20_000.0
SHAFT_SPEED_RPM = 2000.0
RADIAL_LOAD_LBF = 6000.0
EXPECTED_ROWS_PER_SNAPSHOT = 20_480


class InvalidSnapshotNameError(ValueError):
    """Raised when a file name does not match the documented snapshot timestamp format."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Invalid snapshot file name {name!r}: expected format "
            f"'YYYY.MM.DD.HH.MM.SS' (documented NASA IMS naming convention)."
        )


@dataclass(frozen=True, slots=True)
class BearingGeometry:
    """Rolling-element bearing geometry used to derive characteristic fault frequencies."""

    n_rollers: int
    pitch_diameter_in: float
    roller_diameter_in: float
    contact_angle_deg: float


REXNORD_ZA2115 = BearingGeometry(
    n_rollers=16,
    pitch_diameter_in=2.815,
    roller_diameter_in=0.331,
    contact_angle_deg=15.17,
)


@dataclass(frozen=True, slots=True)
class CharacteristicFrequencies:
    """Characteristic bearing fault frequencies, in Hz."""

    ftf: float
    bpfo: float
    bpfi: float
    bsf: float


def characteristic_fault_frequencies(
    shaft_speed_rpm: float = SHAFT_SPEED_RPM,
    geometry: BearingGeometry = REXNORD_ZA2115,
) -> CharacteristicFrequencies:
    """Compute the characteristic fault frequencies for a rolling-element bearing.

    Uses the standard rolling-element bearing formulas for the fundamental train
    frequency (FTF), ball/roller pass frequency outer race (BPFO), ball/roller pass
    frequency inner race (BPFI), and ball/roller spin frequency (BSF). At the documented
    shaft speed of 2000 RPM and the Rexnord ZA-2115 geometry, these evaluate to
    approximately FTF 14.8 Hz, BPFO 236.4 Hz, BPFI 296.9 Hz, BSF 139.9 Hz.
    """
    shaft_freq_hz = shaft_speed_rpm / 60.0
    ratio = (geometry.roller_diameter_in / geometry.pitch_diameter_in) * math.cos(
        math.radians(geometry.contact_angle_deg)
    )

    ftf = (shaft_freq_hz / 2.0) * (1.0 - ratio)
    bpfo = geometry.n_rollers * (shaft_freq_hz / 2.0) * (1.0 - ratio)
    bpfi = geometry.n_rollers * (shaft_freq_hz / 2.0) * (1.0 + ratio)
    bsf = (
        (geometry.pitch_diameter_in / (2.0 * geometry.roller_diameter_in))
        * shaft_freq_hz
        * (1.0 - ratio**2)
    )
    return CharacteristicFrequencies(ftf=ftf, bpfo=bpfo, bpfi=bpfi, bsf=bsf)


@dataclass(frozen=True, slots=True)
class ChannelMapping:
    """Maps one raw data column (1-based, in file column order) to a bearing and axis."""

    channel: int
    bearing: int
    axis: str


@dataclass(frozen=True, slots=True)
class BearingFailure:
    """A documented end-of-test failure mode observed on one bearing."""

    bearing: int
    description: str


@dataclass(frozen=True, slots=True)
class TestSpec:
    """Documented metadata for one of the three NASA IMS run-to-failure tests."""

    test_id: str
    raw_relative_path: Path
    channel_map: tuple[ChannelMapping, ...]
    documented_file_count: int
    documented_start: datetime
    documented_end: datetime
    documented_failures: tuple[BearingFailure, ...]


def _bearing_bidirectional_map(test_id: str, raw_relative_path: str) -> tuple[ChannelMapping, ...]:
    """Build the 8-channel map for a test where each bearing has an x and a y channel."""
    mapping = []
    channel = 1
    for bearing in range(1, 5):
        mapping.append(ChannelMapping(channel=channel, bearing=bearing, axis="x"))
        mapping.append(ChannelMapping(channel=channel + 1, bearing=bearing, axis="y"))
        channel += 2
    return tuple(mapping)


def _bearing_single_axis_map() -> tuple[ChannelMapping, ...]:
    """Build the 4-channel map for a test with a single channel per bearing."""
    return tuple(
        ChannelMapping(channel=bearing, bearing=bearing, axis="single") for bearing in range(1, 5)
    )


TEST_REGISTRY: dict[str, TestSpec] = {
    "1": TestSpec(
        test_id="1",
        raw_relative_path=Path("1st_test/1st_test"),
        channel_map=_bearing_bidirectional_map("1", "1st_test/1st_test"),
        documented_file_count=2156,
        documented_start=datetime(2003, 10, 22, 12, 6, 24),
        documented_end=datetime(2003, 11, 25, 23, 39, 56),
        documented_failures=(
            BearingFailure(bearing=3, description="inner race defect"),
            BearingFailure(bearing=4, description="roller element defect"),
        ),
    ),
    "2": TestSpec(
        test_id="2",
        raw_relative_path=Path("2nd_test/2nd_test"),
        channel_map=_bearing_single_axis_map(),
        documented_file_count=984,
        documented_start=datetime(2004, 2, 12, 10, 32, 39),
        documented_end=datetime(2004, 2, 19, 6, 22, 39),
        documented_failures=(BearingFailure(bearing=1, description="outer race failure"),),
    ),
    "3": TestSpec(
        test_id="3",
        raw_relative_path=Path("3rd_test/4th_test/txt"),
        channel_map=_bearing_single_axis_map(),
        documented_file_count=4448,
        documented_start=datetime(2004, 3, 4, 9, 27, 46),
        documented_end=datetime(2004, 4, 4, 19, 1, 57),
        documented_failures=(BearingFailure(bearing=3, description="outer race failure"),),
    ),
}


def parse_snapshot_timestamp(name: str) -> datetime:
    """Parse a snapshot file name into its recorded timestamp.

    Raises ``InvalidSnapshotNameError`` when ``name`` does not match the documented
    ``YYYY.MM.DD.HH.MM.SS`` format.
    """
    try:
        return datetime.strptime(name, SNAPSHOT_TIMESTAMP_FORMAT)
    except ValueError as error:
        raise InvalidSnapshotNameError(name) from error


def list_snapshots(raw_root: Path, test_id: str) -> list[Path]:
    """List a test's snapshot files, sorted by their parsed timestamp.

    Raises ``KeyError`` for an unknown ``test_id``.
    """
    spec = TEST_REGISTRY[test_id]
    test_dir = Path(raw_root) / spec.raw_relative_path
    paths = [path for path in test_dir.iterdir() if path.is_file()]
    return sorted(paths, key=lambda path: parse_snapshot_timestamp(path.name))


def load_snapshot(path: Path) -> np.ndarray:
    """Load one snapshot file as a float64 array of shape ``(rows, channels)``.

    Uses pandas' C parser for speed and is tolerant of CRLF line endings (handled
    transparently by pandas). Tokens that cannot be parsed as floats become ``NaN``
    instead of raising, so the caller can inspect the array's finiteness rather than
    have the whole file rejected.
    """
    frame = pd.read_csv(path, sep="\t", header=None, engine="c")
    numeric = frame.apply(lambda column: pd.to_numeric(column, errors="coerce"))
    return np.asarray(numeric.to_numpy(dtype=np.float64), dtype=np.float64)
