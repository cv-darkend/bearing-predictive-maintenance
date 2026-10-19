"""Frequency-domain helpers: amplitude spectra, envelope spectra, and exploratory
degradation-onset / post-shutdown detection.

These are analysis primitives for notebook 02 (signal exploration). The onset detector
is explicitly exploratory: a simple mean+n_sigma threshold with a minimum persistence,
used to get a first, honest read on "when does degradation become visible" per bearing.
It is NOT the final labeling rule for modeling (Phase III), which will likely combine
several indicators and a validated methodology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal


def amplitude_spectrum(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided, amplitude-corrected spectrum of a 1D real signal.

    Applies a Hann window (to reduce spectral leakage) and corrects the resulting
    amplitude for the window's coherent gain and for the single-sided folding, so
    that a pure sine of amplitude ``A`` at frequency ``f0`` produces a peak close to
    ``A`` near ``f0`` (subject to windowing/leakage loss when ``f0`` does not fall
    exactly on a frequency bin).

    Returns ``(freqs, amplitudes)`` where ``freqs`` spans ``[0, fs / 2]``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {signal.shape}.")

    n_samples = signal.shape[0]
    window = np.hanning(n_samples)
    coherent_gain = window.mean()

    windowed = signal * window
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    amplitudes = np.abs(spectrum) / (n_samples * coherent_gain)
    # Single-sided correction: double all bins except DC and (for even n) Nyquist,
    # which are not mirrored in the discarded negative-frequency half.
    amplitudes[1:-1] *= 2.0
    if n_samples % 2 != 0:
        amplitudes[-1] *= 2.0

    return freqs, amplitudes


def envelope_spectrum(
    signal: np.ndarray, fs: float, band: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the envelope spectrum of a 1D real signal within a band-pass band.

    Band-pass filters the signal with a Butterworth filter (zero-phase, via
    ``sosfiltfilt``), takes the analytic signal's magnitude (Hilbert envelope), and
    returns the amplitude spectrum of that envelope. This is the standard bearing
    fault-detection technique: resonance/impact energy in ``band`` is amplitude-
    modulated at the fault's characteristic frequency, which shows up as a peak in
    the envelope spectrum even when it is invisible in the raw spectrum.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {signal.shape}.")

    low, high = band
    nyquist = fs / 2.0
    sos = scipy_signal.butter(
        N=4, Wn=[low / nyquist, high / nyquist], btype="bandpass", output="sos"
    )
    filtered = scipy_signal.sosfiltfilt(sos, signal)
    envelope = np.abs(scipy_signal.hilbert(filtered))
    # Remove the envelope's DC offset before spectral analysis so the mean level
    # does not dominate the (already-excluded) DC bin comparison downstream.
    envelope = envelope - envelope.mean()
    return amplitude_spectrum(envelope, fs)


def band_amplitude(freqs: np.ndarray, amps: np.ndarray, target: float, tolerance: float) -> float:
    """Return the largest amplitude within ``target +/- tolerance`` Hz.

    Returns ``nan`` when no frequency bin falls within tolerance of ``target``. Used
    to read the peak near a characteristic fault frequency (or one of its harmonics)
    without depending on exact bin alignment.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    amps = np.asarray(amps, dtype=np.float64)
    mask = np.abs(freqs - target) <= tolerance
    if not np.any(mask):
        return float("nan")
    return float(np.max(amps[mask]))


def detect_onset(
    series: pd.Series, baseline_n: int, n_sigma: float, persistence: int
) -> int | None:
    """Exploratory degradation-onset detector: first index sustained above threshold.

    Computes a baseline mean/std from the first ``baseline_n`` points, then returns
    the first index ``i`` such that ``series[i:i + persistence]`` are all above
    ``baseline_mean + n_sigma * baseline_std``. Returns ``None`` when no such run
    exists.

    This is a deliberately simple, exploratory rule to get a first estimate of "when
    does this indicator start departing from its healthy baseline" for a single
    series (e.g. one bearing's kurtosis over one test). It is NOT the final labeling
    methodology for modeling: a production label would likely combine multiple
    indicators, validate thresholds, and consider domain-specific onset criteria.
    """
    values = series.to_numpy(dtype=np.float64)
    if len(values) < baseline_n:
        return None

    baseline = values[:baseline_n]
    threshold = baseline.mean() + n_sigma * baseline.std(ddof=0)
    above = values > threshold

    run_length = 0
    for index, is_above in enumerate(above):
        run_length = run_length + 1 if is_above else 0
        if run_length >= persistence:
            return index - persistence + 1
    return None


def flag_post_shutdown(
    max_channel_rms: pd.Series, baseline_n: int, fraction_of_baseline_median: float
) -> pd.Series:
    """Flag snapshots whose max-channel RMS looks like a post-shutdown recording.

    A snapshot is flagged when its value is below
    ``fraction_of_baseline_median * median(max_channel_rms[:baseline_n])``. The
    baseline is taken from the first ``baseline_n`` snapshots (early in the test,
    while the rig is known to be running), so a near-zero-amplitude tail snapshot
    (rig already stopped before the recording finished) is flagged without relying
    on a hardcoded absolute threshold.
    """
    values = max_channel_rms.to_numpy(dtype=np.float64)
    baseline_median = float(np.median(values[:baseline_n])) if len(values) else 0.0
    threshold = fraction_of_baseline_median * baseline_median
    flags = values < threshold
    return pd.Series(flags, index=max_channel_rms.index)
