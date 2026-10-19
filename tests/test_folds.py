from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bearing_predictive_maintenance import folds


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "test_id": ["1", "1", "2", "2", "2", "3"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        index=[10, 11, 12, 13, 14, 15],
    )


def test_leave_one_test_out_gives_one_fold_per_test() -> None:
    frame = _frame()

    result = folds.leave_one_test_out(frame)

    assert [fold.held_out for fold in result] == ["1", "2", "3"]
    for fold in result:
        train_groups = set(frame.loc[fold.train_index, "test_id"])
        test_groups = set(frame.loc[fold.test_index, "test_id"])
        assert test_groups == {fold.held_out}
        assert fold.held_out not in train_groups
        assert train_groups.isdisjoint(test_groups)
        # Every row is in exactly one side of the split.
        assert sorted([*fold.train_index, *fold.test_index]) == sorted(frame.index)


def test_leave_one_test_out_uses_frame_index_labels() -> None:
    result = folds.leave_one_test_out(_frame())

    first = result[0]
    assert list(first.test_index) == [10, 11]
    assert list(first.train_index) == [12, 13, 14, 15]


def test_leave_one_test_out_custom_group_column() -> None:
    frame = pd.DataFrame({"group": ["a", "b", "a"]})

    result = folds.leave_one_test_out(frame, group_column="group")

    assert [fold.held_out for fold in result] == ["a", "b"]
    assert list(result[0].test_index) == [0, 2]


def test_leave_one_test_out_requires_two_groups() -> None:
    with pytest.raises(ValueError, match="at least two"):
        folds.leave_one_test_out(pd.DataFrame({"test_id": ["1", "1"]}))


def test_leave_one_test_out_rejects_missing_group_values() -> None:
    with pytest.raises(ValueError, match="missing"):
        folds.leave_one_test_out(pd.DataFrame({"test_id": ["1", None, "2"]}))


def test_standardize_by_fold_uses_training_statistics() -> None:
    features = pd.DataFrame({"x": [1.0, 3.0, 100.0]})
    train_mask = np.array([True, True, False])

    result = folds.standardize_by_fold(features, train_mask)

    # Train mean 2, population std 1.
    np.testing.assert_allclose(result["x"], [-1.0, 1.0, 98.0])


def test_standardize_by_fold_has_no_leakage_from_test_rows() -> None:
    features = pd.DataFrame({"x": [1.0, 2.0, 4.0, 7.0], "y": [5.0, 6.0, 9.0, -3.0]})
    train_mask = pd.Series([True, True, True, False])
    changed = features.copy()
    changed.loc[3] = [1e6, -1e6]

    original = folds.standardize_by_fold(features, train_mask)
    perturbed = folds.standardize_by_fold(changed, train_mask)

    # Training rows (and therefore the fitted mean/std) are unchanged.
    pd.testing.assert_frame_equal(original.iloc[:3], perturbed.iloc[:3])
    scaler = folds.fit_standardizer(features, train_mask)
    scaler_perturbed = folds.fit_standardizer(changed, train_mask)
    pd.testing.assert_series_equal(scaler.mean, scaler_perturbed.mean)
    pd.testing.assert_series_equal(scaler.scale, scaler_perturbed.scale)


def test_standardize_by_fold_zero_std_column_is_centered_with_unit_scale() -> None:
    features = pd.DataFrame({"flat": [5.0, 5.0, 8.0]})
    train_mask = np.array([True, True, False])

    result = folds.standardize_by_fold(features, train_mask)

    np.testing.assert_allclose(result["flat"], [0.0, 0.0, 3.0])
    assert all(math.isfinite(value) for value in result["flat"])


def test_standardize_by_fold_rejects_misaligned_mask() -> None:
    with pytest.raises(ValueError, match="train_mask"):
        folds.standardize_by_fold(pd.DataFrame({"x": [1.0, 2.0]}), np.array([True]))


def test_standardize_by_fold_rejects_empty_training_set() -> None:
    with pytest.raises(ValueError, match="training"):
        folds.standardize_by_fold(pd.DataFrame({"x": [1.0, 2.0]}), np.array([False, False]))
