import numpy as np
import pytest

from data_handler import detect_and_blank_artifacts

def test_detect_and_blank_artifacts_empty():
    """Test with an empty signal chunk."""
    empty_signal = np.array([], dtype=np.float64)
    result = detect_and_blank_artifacts(empty_signal)
    assert result.shape == (0,)
    assert isinstance(result, np.ndarray)

def test_detect_and_blank_artifacts_no_artifacts():
    """Test that a clean signal without artifacts is not altered."""
    clean_signal = np.array([100.0, 200.0, 300.0, 200.0, 100.0], dtype=np.float64)
    result = detect_and_blank_artifacts(clean_signal, threshold=2000.0)
    np.testing.assert_array_equal(result, clean_signal)
    assert not np.shares_memory(result, clean_signal) # Ensure it's a copy

def test_detect_and_blank_artifacts_amp_bad():
    """Test amplitude threshold blanking, ensuring neighbors are also zeroed.
    Note: A sharp peak also causes a high ROC on the falling edge,
    which triggers ROC blanking on the subsequent sample."""
    signal = np.array([100.0, 2500.0, 100.0, 50.0], dtype=np.float64)
    result = detect_and_blank_artifacts(signal, threshold=2000.0)
    # i=1: amp_bad=True -> blanks 0, 1, 2
    # i=2: roc_bad=True (abs(100-2500)=2400>1500) -> blanks 1, 2, 3
    expected = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)

def test_detect_and_blank_artifacts_roc_bad():
    """Test rate-of-change (ROC) threshold blanking."""
    # Default roc_threshold is 0.75 * 2000.0 = 1500.0
    signal = np.array([100.0, 100.0, 1700.0, 100.0], dtype=np.float64)
    result = detect_and_blank_artifacts(signal, threshold=2000.0)
    # The jump from 100 to 1700 is 1600 > 1500, so index 2 is roc_bad.
    # Therefore, indices 1, 2, and 3 should be zeroed.
    expected = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)

def test_detect_and_blank_artifacts_edge_cases():
    """Test artifact at the very beginning and very end of the signal."""
    # Artifact at the start
    signal_start = np.array([2500.0, 100.0, 50.0], dtype=np.float64)
    result_start = detect_and_blank_artifacts(signal_start, threshold=2000.0)
    # i=0 is amp_bad -> blanks 0, 1.
    # i=1 is roc_bad (abs(100-2500)>1500) -> blanks 0, 1, 2.
    expected_start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_array_equal(result_start, expected_start)

    # Artifact at the end
    signal_end = np.array([50.0, 100.0, 2500.0], dtype=np.float64)
    result_end = detect_and_blank_artifacts(signal_end, threshold=2000.0)
    # i=1 is not bad
    # i=2 is amp_bad and roc_bad -> blanks 1, 2
    expected_end = np.array([50.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_array_equal(result_end, expected_end)

def test_detect_and_blank_artifacts_low_roc_threshold():
    """Test ROC threshold clamping to 1200.0 when threshold is low."""
    # Threshold 1000 -> 0.75 * 1000 = 750, which is < 1200.
    # So roc_threshold should be clamped to 1200.0.

    # Test triggering clamped ROC threshold without triggering amp_thresh
    signal = np.array([-500.0, 900.0, 0.0], dtype=np.float64)
    result = detect_and_blank_artifacts(signal, threshold=1000.0)
    # Jump from -500 to 900 is 1400.
    # roc_threshold is clamped to 1200.
    # 1400 > 1200, so index 1 is roc_bad.
    # None of the values exceed amp_thresh of 1000.
    # So indices 0, 1, 2 should be zeroed.
    expected = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)

    # Let's test just below the clamped threshold.
    signal_below = np.array([-500.0, 600.0, 0.0], dtype=np.float64)
    result_below = detect_and_blank_artifacts(signal_below, threshold=1000.0)
    # Jump from -500 to 600 is 1100.
    # 1100 < 1200, so NOT roc_bad.
    # Max val 600 < 1000, so NOT amp_bad.
    np.testing.assert_array_equal(result_below, signal_below)
