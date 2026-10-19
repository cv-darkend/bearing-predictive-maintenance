from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from bearing_predictive_maintenance import ims


def _write_snapshot(path: Path, rows: list[list[str]]) -> None:
    """Write a synthetic snapshot file mimicking the real IMS format: tab-separated,
    CRLF line endings, no header.
    """
    content = "\r\n".join("\t".join(row) for row in rows) + "\r\n"
    path.write_bytes(content.encode("ascii"))


def test_parse_snapshot_timestamp_parses_documented_format() -> None:
    assert ims.parse_snapshot_timestamp("2003.10.22.14.39.13") == datetime(2003, 10, 22, 14, 39, 13)


def test_parse_snapshot_timestamp_rejects_invalid_name() -> None:
    with pytest.raises(ims.InvalidSnapshotNameError, match="not-a-timestamp"):
        ims.parse_snapshot_timestamp("not-a-timestamp")


def test_list_snapshots_returns_paths_sorted_by_timestamp(tmp_path: Path) -> None:
    raw_root = tmp_path
    test_dir = raw_root / "1st_test" / "1st_test"
    test_dir.mkdir(parents=True)
    names = ["2003.10.22.14.39.13", "2003.10.22.12.06.24", "2003.10.22.13.00.00"]
    for name in names:
        _write_snapshot(test_dir / name, [["0.0"] * 8])

    result = ims.list_snapshots(raw_root, "1")

    assert result == [
        test_dir / "2003.10.22.12.06.24",
        test_dir / "2003.10.22.13.00.00",
        test_dir / "2003.10.22.14.39.13",
    ]


def test_list_snapshots_rejects_unknown_test_id(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        ims.list_snapshots(tmp_path, "unknown")


def test_load_snapshot_returns_float64_array_with_expected_shape(tmp_path: Path) -> None:
    path = tmp_path / "2003.10.22.14.39.13"
    _write_snapshot(
        path,
        [
            ["-0.110", "-0.066"],
            ["-0.129", "-0.024"],
            ["-0.146", "-0.151"],
        ],
    )

    array = ims.load_snapshot(path)

    assert array.dtype == np.float64
    assert array.shape == (3, 2)
    np.testing.assert_allclose(
        array,
        [[-0.110, -0.066], [-0.129, -0.024], [-0.146, -0.151]],
    )


def test_load_snapshot_coerces_unparsable_tokens_to_nan(tmp_path: Path) -> None:
    path = tmp_path / "2003.10.22.14.39.13"
    _write_snapshot(
        path,
        [
            ["0.1", "0.2"],
            ["garbage", "0.4"],
        ],
    )

    array = ims.load_snapshot(path)

    assert array.shape == (2, 2)
    assert math.isnan(array[1, 0])
    assert array[1, 1] == pytest.approx(0.4)


def test_registry_documents_channel_counts_and_file_counts() -> None:
    assert len(ims.TEST_REGISTRY["1"].channel_map) == 8
    assert len(ims.TEST_REGISTRY["2"].channel_map) == 4
    assert len(ims.TEST_REGISTRY["3"].channel_map) == 4
    assert ims.TEST_REGISTRY["1"].documented_file_count == 2156
    assert ims.TEST_REGISTRY["2"].documented_file_count == 984
    assert ims.TEST_REGISTRY["3"].documented_file_count == 4448
    assert ims.TEST_REGISTRY["1"].raw_relative_path == Path("1st_test/1st_test")
    assert ims.TEST_REGISTRY["2"].raw_relative_path == Path("2nd_test/2nd_test")
    assert ims.TEST_REGISTRY["3"].raw_relative_path == Path("3rd_test/4th_test/txt")


def test_registry_channel_map_matches_documented_bearing_layout() -> None:
    test1 = ims.TEST_REGISTRY["1"]
    mapping = {(m.channel): (m.bearing, m.axis) for m in test1.channel_map}
    assert mapping[1] == (1, "x")
    assert mapping[2] == (1, "y")
    assert mapping[3] == (2, "x")
    assert mapping[4] == (2, "y")
    assert mapping[5] == (3, "x")
    assert mapping[6] == (3, "y")
    assert mapping[7] == (4, "x")
    assert mapping[8] == (4, "y")

    test2 = ims.TEST_REGISTRY["2"]
    mapping2 = {(m.channel): (m.bearing, m.axis) for m in test2.channel_map}
    assert mapping2[1] == (1, "single")
    assert mapping2[4] == (4, "single")


def test_characteristic_fault_frequencies_match_documented_values() -> None:
    frequencies = ims.characteristic_fault_frequencies(shaft_speed_rpm=2000.0)

    assert frequencies.ftf == pytest.approx(14.8, abs=0.2)
    assert frequencies.bpfo == pytest.approx(236.4, abs=0.5)
    assert frequencies.bpfi == pytest.approx(296.9, abs=0.5)
    assert frequencies.bsf == pytest.approx(139.9, abs=0.5)
