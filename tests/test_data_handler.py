import pytest
import numpy as np
from data_handler import calc_zcr

def test_calc_zcr_basic():
    # Basic alternating sequence
    # 4 points, 3 crossings
    # signal[0]=1.0, signal[1]=-1.0 (cross), signal[2]=1.0 (cross), signal[3]=-1.0 (cross)
    # crossings = 3, n = 4, n-1 = 3, return 3/3 = 1.0
    signal = np.array([1.0, -1.0, 1.0, -1.0])
    assert calc_zcr(signal) == 1.0

def test_calc_zcr_no_crossings():
    # All positive
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    assert calc_zcr(signal) == 0.0

    # All negative
    signal = np.array([-1.0, -2.0, -3.0, -4.0])
    assert calc_zcr(signal) == 0.0

def test_calc_zcr_zero_boundaries():
    # A transition from 0 to negative
    signal = np.array([0.0, -1.0])
    # The condition is:
    # (signal[i - 1] >= 0.0 and signal[i] < 0.0) -> (0.0 >= 0.0 and -1.0 < 0.0) -> True
    assert calc_zcr(signal) == 1.0

    # A transition from negative to 0
    signal = np.array([-1.0, 0.0])
    # The condition is:
    # (signal[i - 1] < 0.0 and signal[i] >= 0.0) -> (-1.0 < 0.0 and 0.0 >= 0.0) -> True
    assert calc_zcr(signal) == 1.0

    # A transition from 0 to positive
    signal = np.array([0.0, 1.0])
    # Neither condition matches:
    # (0.0 >= 0.0 and 1.0 < 0.0) -> False
    # (0.0 < 0.0 and 1.0 >= 0.0) -> False
    assert calc_zcr(signal) == 0.0

    # A transition from positive to 0
    signal = np.array([1.0, 0.0])
    # Neither condition matches:
    # (1.0 >= 0.0 and 0.0 < 0.0) -> False
    # (1.0 < 0.0 and 0.0 >= 0.0) -> False
    assert calc_zcr(signal) == 0.0

    # Constant 0
    signal = np.array([0.0, 0.0, 0.0])
    assert calc_zcr(signal) == 0.0

def test_calc_zcr_short_signals():
    # Empty signal
    signal = np.array([])
    assert calc_zcr(signal) == 0.0

    # Single element signal
    signal = np.array([1.0])
    assert calc_zcr(signal) == 0.0
