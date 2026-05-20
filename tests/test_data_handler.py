import pytest
import numpy as np
from data_handler import calc_rms

def test_calc_rms_positive_values():
    # Test with typical positive values
    signal = np.array([3.0, 4.0], dtype=np.float64)
    expected_rms = np.sqrt((3.0**2 + 4.0**2) / 2)
    assert np.isclose(calc_rms(signal), expected_rms)

def test_calc_rms_negative_values():
    # Test with negative values
    signal = np.array([-3.0, -4.0], dtype=np.float64)
    expected_rms = np.sqrt(((-3.0)**2 + (-4.0)**2) / 2)
    assert np.isclose(calc_rms(signal), expected_rms)

def test_calc_rms_mixed_values():
    # Test with mixed positive and negative values
    signal = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float64)
    expected_rms = np.sqrt(((-1.0)**2 + 2.0**2 + (-3.0)**2 + 4.0**2) / 4)
    assert np.isclose(calc_rms(signal), expected_rms)

def test_calc_rms_zero_values():
    # Test with array of zeros
    signal = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    expected_rms = 0.0
    assert np.isclose(calc_rms(signal), expected_rms)

def test_calc_rms_constant_values():
    # Test with constant values
    signal = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    expected_rms = 5.0
    assert np.isclose(calc_rms(signal), expected_rms)

def test_calc_rms_empty_array():
    # Test with empty array. Current implementation raises ZeroDivisionError.
    signal = np.array([], dtype=np.float64)
    with pytest.raises(ZeroDivisionError):
        calc_rms(signal)

def test_calc_rms_large_values():
    # Test with large numbers
    signal = np.array([1e6, 2e6], dtype=np.float64)
    expected_rms = np.sqrt(((1e6)**2 + (2e6)**2) / 2)
    assert np.isclose(calc_rms(signal), expected_rms)
