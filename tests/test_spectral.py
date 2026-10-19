from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bearing_predictive_maintenance import spectral


def test_amplitude_spectrum_of_pure_sine_peaks_near_amplitude() -> None:
    fs = 20_000.0
    f0 = 500.0
    amplitude = 1.5
    n_samples = 20_480
    t = np.arange(n_samples) / fs
    signal = amplitude * np.sin(2 * math.pi * f0 * t)

    freqs, amps = spectral.amplitude_spectrum(signal, fs)

    assert freqs[0] == pytest.approx(0.0)
    assert freqs[-1] == pytest.approx(fs / 2.0, rel=1e-3)
    peak_index = int(np.argmax(amps))
    assert freqs[peak_index] == pytest.approx(f0, abs=fs / n_samples * 2)
    # Hann-window amplitude correction should recover close to the true amplitude,
    # allowing for the window's inherent scalloping/leakage loss.
    assert amps[peak_index] == pytest.approx(amplitude, rel=0.15)


def test_amplitude_spectrum_is_single_sided_and_non_negative() -> None:
    fs = 1000.0
    n_samples = 1024
    t = np.arange(n_samples) / fs
    signal = np.sin(2 * math.pi * 50.0 * t)

    freqs, amps = spectral.amplitude_spectrum(signal, fs)

    assert len(freqs) == len(amps)
    assert len(freqs) == n_samples // 2 + 1
    assert np.all(amps >= 0.0)
    assert np.all(np.diff(freqs) > 0)


def test_amplitude_spectrum_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1D"):
        spectral.amplitude_spectrum(np.zeros((10, 2)), fs=1000.0)


def test_envelope_spectrum_recovers_modulation_frequency() -> None:
    fs = 20_000.0
    n_samples = 20_480
    t = np.arange(n_samples) / fs
    carrier_freq = 3000.0
    modulation_freq = 120.0
    modulation_depth = 0.8

    carrier = np.sin(2 * math.pi * carrier_freq * t)
    envelope = 1.0 + modulation_depth * np.sin(2 * math.pi * modulation_freq * t)
    signal = envelope * carrier

    freqs, amps = spectral.envelope_spectrum(signal, fs, band=(1000.0, 5000.0))

    # Ignore the DC bin; the modulation frequency should be the dominant AC peak.
    ac_mask = freqs > 10.0
    peak_index = int(np.argmax(amps[ac_mask]))
    peak_freq = freqs[ac_mask][peak_index]
    assert peak_freq == pytest.approx(modulation_freq, abs=5.0)


def test_envelope_spectrum_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1D"):
        spectral.envelope_spectrum(np.zeros((10, 2)), fs=1000.0, band=(100.0, 200.0))


def test_band_amplitude_reads_peak_near_target_frequency() -> None:
    freqs = np.linspace(0.0, 1000.0, 2001)
    amps = np.zeros_like(freqs)
    target_index = int(np.argmin(np.abs(freqs - 236.4)))
    amps[target_index] = 0.05
    amps[target_index - 1] = 0.02

    result = spectral.band_amplitude(freqs, amps, target=236.4, tolerance=2.0)

    assert result == pytest.approx(0.05)


def test_band_amplitude_returns_nan_when_nothing_in_tolerance() -> None:
    freqs = np.linspace(0.0, 1000.0, 2001)
    amps = np.ones_like(freqs)

    result = spectral.band_amplitude(freqs, amps, target=5000.0, tolerance=1.0)

    assert math.isnan(result)


def test_fault_peak_finds_peak_shifted_by_bearing_slip() -> None:
    freqs = np.linspace(0.0, 1000.0, 4001)
    amps = np.full_like(freqs, 0.001)
    # Observed outer-race peak sits ~2.2% below the kinematic BPFO (231.2 vs 236.4 Hz),
    # outside a fixed +/-5 Hz window but within a 3% relative tolerance.
    observed_index = int(np.argmin(np.abs(freqs - 231.2)))
    amps[observed_index] = 0.2

    peak_freq, peak_amp = spectral.fault_peak(freqs, amps, target=236.4, rel_tolerance=0.03)

    assert peak_freq == pytest.approx(freqs[observed_index])
    assert peak_amp == pytest.approx(0.2)
    # The fixed +/-5 Hz window misses the peak and reads only the floor.
    assert spectral.band_amplitude(freqs, amps, target=236.4, tolerance=5.0) == pytest.approx(0.001)


def test_fault_peak_returns_nan_pair_when_no_bin_in_window() -> None:
    freqs = np.linspace(0.0, 100.0, 101)
    amps = np.ones_like(freqs)

    peak_freq, peak_amp = spectral.fault_peak(freqs, amps, target=5000.0, rel_tolerance=0.03)

    assert math.isnan(peak_freq)
    assert math.isnan(peak_amp)


@pytest.mark.parametrize("rel_tolerance", [0.0, -0.1, 1.0])
def test_fault_peak_rejects_invalid_tolerance(rel_tolerance: float) -> None:
    freqs = np.linspace(0.0, 1000.0, 2001)
    with pytest.raises(ValueError, match="rel_tolerance"):
        spectral.fault_peak(freqs, np.ones_like(freqs), target=236.4, rel_tolerance=rel_tolerance)


def test_detect_onset_finds_first_persistent_excursion() -> None:
    baseline_n = 50
    persistence = 5
    baseline = np.random.default_rng(0).normal(loc=0.0, scale=1.0, size=baseline_n)
    # A clean step: flat baseline, then a jump that stays elevated.
    series = pd.Series(
        np.concatenate([baseline, np.full(30, 20.0)]),
    )

    onset_index = spectral.detect_onset(
        series, baseline_n=baseline_n, n_sigma=3.0, persistence=persistence
    )

    assert onset_index == baseline_n


def test_detect_onset_ignores_isolated_spikes_shorter_than_persistence() -> None:
    baseline_n = 50
    persistence = 5
    rng = np.random.default_rng(1)
    baseline = rng.normal(loc=0.0, scale=1.0, size=baseline_n)
    tail = np.zeros(20)
    tail[3] = 50.0  # isolated single-point spike, shorter than persistence
    series = pd.Series(np.concatenate([baseline, tail]))

    onset_index = spectral.detect_onset(
        series, baseline_n=baseline_n, n_sigma=3.0, persistence=persistence
    )

    assert onset_index is None


def test_detect_onset_returns_none_when_series_never_exceeds_threshold() -> None:
    rng = np.random.default_rng(2)
    series = pd.Series(rng.normal(loc=0.0, scale=1.0, size=100))

    onset_index = spectral.detect_onset(series, baseline_n=50, n_sigma=6.0, persistence=5)

    assert onset_index is None


def test_flag_post_shutdown_snapshots_detects_dead_trailing_rows() -> None:
    frame = pd.DataFrame(
        {
            "file": ["a", "b", "c", "d", "e"],
            "max_rms": [0.05, 0.06, 0.055, 0.058, 0.001],
        }
    )

    flags = spectral.flag_post_shutdown(
        frame["max_rms"], baseline_n=4, fraction_of_baseline_median=0.1
    )

    assert list(flags) == [False, False, False, False, True]


def test_flag_post_shutdown_keeps_normal_snapshots_unflagged() -> None:
    frame = pd.DataFrame(
        {
            "file": ["a", "b", "c", "d"],
            "max_rms": [0.05, 0.06, 0.055, 0.058],
        }
    )

    flags = spectral.flag_post_shutdown(
        frame["max_rms"], baseline_n=4, fraction_of_baseline_median=0.1
    )

    assert list(flags) == [False, False, False, False]


def test_detect_onset_ignores_excursions_inside_the_baseline_window() -> None:
    baseline_n = 50
    values = np.random.default_rng(3).normal(loc=0.0, scale=1.0, size=baseline_n + 30)
    values[:3] = 12.0  # run-in transient inside the baseline window
    values[baseline_n + 10 :] = 40.0  # real, sustained degradation after the baseline

    onset_index = spectral.detect_onset(
        pd.Series(values), baseline_n=baseline_n, n_sigma=3.0, persistence=3
    )

    assert onset_index == baseline_n + 10


def test_detect_onset_ignores_nan_in_baseline() -> None:
    baseline_n = 50
    values = np.concatenate([np.random.default_rng(4).normal(size=baseline_n), np.full(20, 30.0)])
    values[7] = np.nan

    onset_index = spectral.detect_onset(
        pd.Series(values), baseline_n=baseline_n, n_sigma=3.0, persistence=5
    )

    assert onset_index == baseline_n


@pytest.mark.parametrize(("baseline_n", "persistence"), [(0, 5), (-1, 5), (10, 0), (10, -2)])
def test_detect_onset_rejects_non_positive_parameters(baseline_n: int, persistence: int) -> None:
    series = pd.Series(np.arange(100, dtype=float))
    with pytest.raises(ValueError, match="must be positive"):
        spectral.detect_onset(series, baseline_n=baseline_n, n_sigma=3.0, persistence=persistence)


def test_flag_post_shutdown_flags_nan_and_tolerates_nan_in_baseline() -> None:
    max_rms = pd.Series([0.05, np.nan, 0.055, 0.058, 0.06, np.nan, 0.001])

    flags = spectral.flag_post_shutdown(max_rms, baseline_n=4, fraction_of_baseline_median=0.1)

    # A NaN snapshot carries no valid recording, so it can never be the "last valid" one.
    assert list(flags) == [False, True, False, False, False, True, True]
