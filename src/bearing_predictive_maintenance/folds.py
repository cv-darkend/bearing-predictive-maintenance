"""Leakage-safe validation: leave-one-test-out folds and train-only standardization.

The three NASA IMS run-to-failure tests are the independent units: snapshots of the
same test are strongly autocorrelated, so a random row split would leak. Each fold
holds out one whole test and trains on the others, and feature standardization is
fitted on the training rows of the fold only, then applied to every row.

The split is implemented directly (it is equivalent to scikit-learn's
``LeaveOneGroupOut``) so it returns pandas index labels and stays typed under mypy
strict without scikit-learn stubs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One leave-one-test-out split, expressed as labels of the source frame's index."""

    held_out: object
    train_index: pd.Index
    test_index: pd.Index


@dataclass(frozen=True)
class Standardizer:
    """Per-column mean and scale fitted on training rows only."""

    mean: pd.Series
    scale: pd.Series

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply ``(x - mean) / scale`` column-wise to ``features``."""
        return (features - self.mean) / self.scale


def leave_one_test_out(frame: pd.DataFrame, group_column: str = "test_id") -> list[Fold]:
    """Return one fold per distinct value of ``group_column``, in sorted order.

    In each fold every row of the held-out group is in ``test_index`` and every other
    row in ``train_index``, so no group is ever shared between train and test. At least
    two groups are required, and missing group values are rejected.
    """
    groups = frame[group_column]
    if groups.isna().any():
        raise ValueError(f"Column {group_column!r} has missing values.")
    unique = sorted(groups.unique())
    if len(unique) < 2:
        raise ValueError(f"Leave-one-out needs at least two groups in {group_column!r}.")
    result = []
    for value in unique:
        is_test = (groups == value).to_numpy()
        result.append(
            Fold(
                held_out=value,
                train_index=frame.index[~is_test],
                test_index=frame.index[is_test],
            )
        )
    return result


def _as_mask(features: pd.DataFrame, train_mask: pd.Series | np.ndarray) -> np.ndarray:
    # The mask is positional (one boolean per row, in row order); a Series index is
    # not aligned against ``features.index``.
    mask = np.asarray(train_mask, dtype=bool)
    if mask.shape != (len(features),):
        raise ValueError(
            f"train_mask has shape {mask.shape}; expected ({len(features)},) matching features."
        )
    if not mask.any():
        raise ValueError("train_mask selects no training rows.")
    return mask


def fit_standardizer(features: pd.DataFrame, train_mask: pd.Series | np.ndarray) -> Standardizer:
    """Fit per-column mean and population std (``ddof=0``) on the training rows only.

    A column whose training std is zero (or not finite) gets scale 1: it is centered
    on the training mean but not rescaled, so it never produces inf.
    """
    train = features.loc[_as_mask(features, train_mask)]
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=0)
    scale = std.where(np.isfinite(std) & (std > 0), 1.0)
    return Standardizer(mean=mean, scale=scale)


def standardize_by_fold(features: pd.DataFrame, train_mask: pd.Series | np.ndarray) -> pd.DataFrame:
    """Standardize all rows with mean/std fitted on the rows where ``train_mask`` holds.

    Test rows never influence the fitted parameters; zero-std training columns are
    centered with scale 1 (see ``fit_standardizer``).
    """
    return fit_standardizer(features, train_mask).transform(features)
