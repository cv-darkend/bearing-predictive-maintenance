from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bearing_predictive_maintenance import ims, indicators, segmentation


def test_segment_constants_tile_one_snapshot_exactly() -> None:
    assert segmentation.SEGMENT_LENGTH == 2048
    assert segmentation.N_SEGMENTS == 10
    assert segmentation.SEGMENT_LENGTH * segmentation.N_SEGMENTS == (ims.EXPECTED_ROWS_PER_SNAPSHOT)


def test_remove_dc_on_1d_signal() -> None:
    signal = np.array([1.0, 2.0, 3.0, 6.0])

    result = segmentation.remove_dc(signal)

    np.testing.assert_allclose(result, [-2.0, -1.0, 0.0, 3.0])
    # The input is not modified in place.
    np.testing.assert_allclose(signal, [1.0, 2.0, 3.0, 6.0])


def test_remove_dc_on_2d_signal_is_per_channel() -> None:
    signal = np.array([[1.0, 10.0], [3.0, 30.0]])

    result = segmentation.remove_dc(signal)

    np.testing.assert_allclose(result, [[-1.0, -10.0], [1.0, 10.0]])
    np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0])


def test_remove_dc_rejects_3d_input() -> None:
    with pytest.raises(ValueError, match="1D or 2D"):
        segmentation.remove_dc(np.zeros((2, 2, 2)))


def test_split_segments_shape_order_and_contiguity() -> None:
    signal = np.arange(12, dtype=np.float64)

    segments = segmentation.split_segments(signal, segment_length=4)

    assert segments.shape == (3, 4)
    np.testing.assert_array_equal(segments[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(segments[1], [4, 5, 6, 7])
    np.testing.assert_array_equal(segments[2], [8, 9, 10, 11])
    # Contiguous and non-overlapping: concatenating the segments restores the signal.
    np.testing.assert_array_equal(segments.reshape(-1), signal)


def test_split_segments_default_gives_ten_segments_per_snapshot() -> None:
    signal = np.zeros(ims.EXPECTED_ROWS_PER_SNAPSHOT)

    segments = segmentation.split_segments(signal)

    assert segments.shape == (10, 2048)


def test_split_segments_rejects_non_multiple_length() -> None:
    with pytest.raises(ValueError, match="multiple"):
        segmentation.split_segments(np.zeros(10), segment_length=4)


def test_split_segments_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1D"):
        segmentation.split_segments(np.zeros((8, 2)), segment_length=4)


def test_split_segments_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        segmentation.split_segments(np.zeros(8), segment_length=0)


def test_segment_indicators_match_indicators_on_each_segment() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(0.5, 1.0, size=40)

    table = segmentation.segment_indicators(signal, segment_length=8)

    assert list(table.index) == [1, 2, 3, 4, 5]
    assert table.index.name == "segment"
    assert list(table.columns) == list(indicators.TIME_DOMAIN_INDICATOR_NAMES)
    for number in range(1, 6):
        piece = signal[(number - 1) * 8 : number * 8]
        expected = indicators.compute_time_domain_indicators(piece[:, np.newaxis]).loc[1]
        pd.testing.assert_series_equal(
            table.loc[number], expected, check_names=False, check_exact=False
        )


def test_summarize_segments_median() -> None:
    table = pd.DataFrame({"rms": [1.0, 2.0, 10.0], "kurtosis": [3.0, 4.0, 5.0]})

    summary = segmentation.summarize_segments(table)

    assert summary["rms_median"] == pytest.approx(2.0)
    assert summary["kurtosis_median"] == pytest.approx(4.0)
    assert list(summary.index) == ["rms_median", "kurtosis_median"]


def test_summarize_segments_p90_and_max() -> None:
    table = pd.DataFrame({"rms": np.arange(1.0, 12.0)})  # 1..11

    p90 = segmentation.summarize_segments(table, statistic="p90")
    maximum = segmentation.summarize_segments(table, statistic="max")

    assert p90["rms_p90"] == pytest.approx(10.0)
    assert maximum["rms_max"] == pytest.approx(11.0)


def test_summarize_segments_rejects_unknown_statistic() -> None:
    with pytest.raises(ValueError, match="statistic"):
        segmentation.summarize_segments(pd.DataFrame({"rms": [1.0]}), statistic="mean")
