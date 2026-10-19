from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bearing_predictive_maintenance import ims, indicators


def _write_snapshot(path: Path, rows: list[list[str]]) -> None:
    content = "\r\n".join("\t".join(row) for row in rows) + "\r\n"
    path.write_bytes(content.encode("ascii"))


def test_compute_time_domain_indicators_on_constant_signal() -> None:
    signal = np.full((100, 2), 5.0)

    result = indicators.compute_time_domain_indicators(signal)

    assert list(result.index) == [1, 2]
    for channel in (1, 2):
        row = result.loc[channel]
        assert row["mean"] == pytest.approx(5.0)
        assert row["std"] == pytest.approx(0.0)
        assert row["rms"] == pytest.approx(5.0)
        assert row["peak"] == pytest.approx(5.0)
        assert row["peak_to_peak"] == pytest.approx(0.0)
        assert row["crest_factor"] == pytest.approx(1.0)
        assert row["impulse_factor"] == pytest.approx(1.0)
        assert row["shape_factor"] == pytest.approx(1.0)
        assert row["energy"] == pytest.approx(100 * 25.0)


def test_compute_time_domain_indicators_on_sine_wave_matches_theory() -> None:
    amplitude = 2.0
    n_samples = 10_000
    n_cycles = 10
    t = np.linspace(0.0, n_cycles, n_samples, endpoint=False)
    signal = (amplitude * np.sin(2 * math.pi * t)).reshape(-1, 1)

    result = indicators.compute_time_domain_indicators(signal)
    row = result.loc[1]

    assert row["mean"] == pytest.approx(0.0, abs=1e-9)
    assert row["rms"] == pytest.approx(amplitude / math.sqrt(2), rel=1e-3)
    assert row["peak"] == pytest.approx(amplitude, rel=1e-3)
    assert row["crest_factor"] == pytest.approx(math.sqrt(2), rel=1e-2)


def test_compute_time_domain_indicators_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        indicators.compute_time_domain_indicators(np.array([1.0, 2.0, 3.0]))


def test_build_indicator_dataset_produces_tidy_frame(tmp_path: Path) -> None:
    test_dir = tmp_path / "2nd_test" / "2nd_test"
    test_dir.mkdir(parents=True)
    _write_snapshot(
        test_dir / "2004.02.12.10.32.39",
        [["0.1", "0.2", "0.3", "0.4"] for _ in range(5)],
    )
    _write_snapshot(
        test_dir / "2004.02.12.10.42.39",
        [["0.5", "0.6", "0.7", "0.8"] for _ in range(5)],
    )

    frame = indicators.build_indicator_dataset(tmp_path, "2")

    assert set(frame["file"]) == {"2004.02.12.10.32.39", "2004.02.12.10.42.39"}
    assert set(frame["channel"]) == {1, 2, 3, 4}
    assert len(frame) == 2 * 4
    assert set(frame["bearing"]) == {1, 2, 3, 4}
    first_file_channel1 = frame[
        (frame["file"] == "2004.02.12.10.32.39") & (frame["channel"] == 1)
    ].iloc[0]
    assert first_file_channel1["mean"] == pytest.approx(0.1)
    assert first_file_channel1["elapsed_hours"] == pytest.approx(0.0)
    assert first_file_channel1["n_non_finite"] == 0


def test_build_indicator_dataset_skips_unreadable_file(tmp_path: Path) -> None:
    test_dir = tmp_path / "2nd_test" / "2nd_test"
    test_dir.mkdir(parents=True)
    _write_snapshot(
        test_dir / "2004.02.12.10.32.39",
        [["0.1", "0.2", "0.3", "0.4"] for _ in range(5)],
    )
    (test_dir / "2004.02.12.10.42.39").write_bytes(b"")

    frame = indicators.build_indicator_dataset(tmp_path, "2")

    assert set(frame["file"]) == {"2004.02.12.10.32.39"}


def test_build_parser_defaults() -> None:
    args = indicators.build_parser().parse_args([])

    assert args.raw_root == Path("data/raw")
    assert args.output == Path("data/interim/ims_snapshot_indicators.csv")
    assert args.inventory_output == Path("data/interim/ims_inventory.csv")


def test_main_writes_both_csv_outputs(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    for spec in ims.TEST_REGISTRY.values():
        test_dir = raw_root / spec.raw_relative_path
        test_dir.mkdir(parents=True)
        n_channels = len(spec.channel_map)
        _write_snapshot(
            test_dir / spec.documented_start.strftime(ims.SNAPSHOT_TIMESTAMP_FORMAT),
            [["0.1"] * n_channels for _ in range(4)],
        )

    output = tmp_path / "out" / "indicators.csv"
    inventory_output = tmp_path / "out" / "inventory.csv"

    indicators.main(
        [
            "--raw-root",
            str(raw_root),
            "--output",
            str(output),
            "--inventory-output",
            str(inventory_output),
        ]
    )

    assert output.exists()
    assert inventory_output.exists()
    indicator_frame = pd.read_csv(output, dtype={"test_id": str})
    inventory_frame = pd.read_csv(inventory_output, dtype={"test_id": str})
    assert set(inventory_frame["test_id"]) == {"1", "2", "3"}
    assert set(indicator_frame["test_id"]) == {"1", "2", "3"}
