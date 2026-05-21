import pytest
import numpy as np
from ai_model_v2 import frame_signal

def test_frame_signal_basic():
    # 2 channels, 10 samples
    # Ch 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Ch 1: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    channels = np.array([
        np.arange(10),
        np.arange(10, 20)
    ], dtype=np.float32)

    # window_size=4, step_size=2
    # Expect 4 windows:
    # starts: [0, 2, 4, 6]
    # w0: ch0=[0,1,2,3], ch1=[10,11,12,13]
    # w1: ch0=[2,3,4,5], ch1=[12,13,14,15]
    # w2: ch0=[4,5,6,7], ch1=[14,15,16,17]
    # w3: ch0=[6,7,8,9], ch1=[16,17,18,19]

    windows, starts = frame_signal(channels, window_size=4, step_size=2)

    assert starts.shape == (4,)
    np.testing.assert_array_equal(starts, np.array([0, 2, 4, 6], dtype=np.int32))

    assert windows.shape == (4, 2, 4)
    np.testing.assert_array_equal(windows[0, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(windows[0, 1], [10, 11, 12, 13])

    np.testing.assert_array_equal(windows[1, 0], [2, 3, 4, 5])
    np.testing.assert_array_equal(windows[1, 1], [12, 13, 14, 15])

    np.testing.assert_array_equal(windows[3, 0], [6, 7, 8, 9])
    np.testing.assert_array_equal(windows[3, 1], [16, 17, 18, 19])


def test_frame_signal_1d_input():
    channels = np.arange(10, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 2-D channels array"):
        frame_signal(channels, window_size=4, step_size=2)

def test_frame_signal_3d_input():
    channels = np.zeros((2, 10, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 2-D channels array"):
        frame_signal(channels, window_size=4, step_size=2)

def test_frame_signal_insufficient_samples():
    channels = np.zeros((2, 3), dtype=np.float32) # only 3 samples
    windows, starts = frame_signal(channels, window_size=4, step_size=2)

    assert starts.shape == (0,)
    assert starts.dtype == np.int32

    assert windows.shape == (0, 2, 4)
    assert windows.dtype == np.float32

def test_frame_signal_exact_fit():
    # n_samples = 10, window_size = 4, step_size = 3
    # (10 - 4) % 3 == 0 -> exact fit
    channels = np.zeros((1, 10), dtype=np.float32)
    windows, starts = frame_signal(channels, window_size=4, step_size=3)

    assert starts.shape == (3,)
    np.testing.assert_array_equal(starts, np.array([0, 3, 6], dtype=np.int32))
    assert windows.shape == (3, 1, 4)

def test_frame_signal_not_exact_fit():
    # n_samples = 11, window_size = 4, step_size = 3
    # (11 - 4) % 3 == 1 -> not exact fit
    channels = np.zeros((1, 11), dtype=np.float32)
    windows, starts = frame_signal(channels, window_size=4, step_size=3)

    assert starts.shape == (3,)
    np.testing.assert_array_equal(starts, np.array([0, 3, 6], dtype=np.int32))
    assert windows.shape == (3, 1, 4)

def test_frame_signal_equal_to_window_size():
    # n_samples == window_size
    channels = np.zeros((1, 4), dtype=np.float32)
    windows, starts = frame_signal(channels, window_size=4, step_size=2)

    assert starts.shape == (1,)
    np.testing.assert_array_equal(starts, np.array([0], dtype=np.int32))
    assert windows.shape == (1, 1, 4)
