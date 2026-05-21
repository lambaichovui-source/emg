import numpy as np
import pytest

from gold_standard_model import apply_energy_gate

def test_apply_energy_gate_short_signal():
    """Test that short signals (< 8 samples) are returned without modification."""
    fs = 10_000.0
    short_signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    output = apply_energy_gate(short_signal, fs)

    np.testing.assert_array_equal(output, short_signal)


def test_apply_energy_gate_bandpass():
    """Test the happy path: artificial burst should survive the gate, quiet baseline should not."""
    fs = 10_000.0
    n = 4000
    t = np.arange(n) / fs
    sig = 0.01 * np.random.default_rng(0).standard_normal(n)

    # Add an artificial burst in the middle
    burst = 0.8 * np.sin(2 * np.pi * 180.0 * t[1500:1700])
    sig[1500:1700] += burst

    # Process signal
    gated = apply_energy_gate(sig, fs)

    # Baseline noise before the burst should be suppressed
    assert float(np.max(np.abs(gated[:1200]))) < 0.35

    # The burst region should be preserved
    assert float(np.max(np.abs(gated[1500:1700]))) > 0.2


def test_apply_energy_gate_highpass():
    """Test the highpass fallback condition when high_hz > nyquist frequency."""
    fs = 1000.0
    nyq = fs * 0.5 # 500.0

    n = 1000
    t = np.arange(n) / fs

    # Signal with low frequency (to be filtered out) and high frequency (to be preserved)
    low_freq = 0.5 * np.sin(2 * np.pi * 10.0 * t)  # 10 Hz, below low_hz of 30.0
    high_freq = 0.8 * np.sin(2 * np.pi * 200.0 * t) # 200 Hz

    sig = low_freq.copy()

    # Add a burst of high frequency
    sig[400:600] += high_freq[400:600]

    # Process signal with high_hz > nyq to trigger the highpass logic
    gated = apply_energy_gate(sig, fs, high_hz=600.0)

    # Low frequency should be attenuated, burst should be preserved
    # We just ensure it runs and does some gating
    assert float(np.max(np.abs(gated[:300]))) < 0.1  # mostly low freq
    assert float(np.max(np.abs(gated[400:600]))) > 0.2 # high freq burst


def test_apply_energy_gate_pure_noise():
    """Test that a pure noise signal without bursts is mostly thresholded out."""
    fs = 10_000.0
    n = 4000
    sig = 0.05 * np.random.default_rng(42).standard_normal(n)

    gated = apply_energy_gate(sig, fs)

    # With pure noise, MAD will be small, so it depends on the random seed
    # but generally the output should not contain large bursts
    assert float(np.max(np.abs(gated))) < 0.2
