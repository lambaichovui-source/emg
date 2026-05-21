import pytest
import numpy as np
from ai_model_v2 import build_pulse_train

def test_build_pulse_train_valid():
    """Test valid inputs for build_pulse_train."""
    num_samples = 1000
    sample_rate = 1000.0  # 1kHz => 1 sample = 1ms
    num_mus = 3
    pulse_width_ms = 5.0

    annotations = [
        {"label": "MUP", "mu_id": "MU1", "start_time": 0.1, "end_time": 0.1},
        {"label": "MUP", "mu_id": "MU2", "start_time": 0.2, "end_time": 0.2},
        {"label": "AI: MUP", "mu_id": "MU3", "start_time": 0.3, "end_time": 0.3},
    ]

    target = build_pulse_train(num_samples, sample_rate, annotations, num_mus, pulse_width_ms)

    assert target.shape == (num_samples, num_mus)
    assert target.dtype == np.float32

    # Pulse width is 5ms -> 5 samples since fs=1000.
    # pulse_width = 5, half_width = 5 // 2 = 2
    # For start_time=0.1s => center=100 => lo=98, hi=103
    assert np.sum(target[:, 0]) == 5
    assert np.all(target[98:103, 0] == 1.0)
    assert np.all(target[:98, 0] == 0.0)
    assert np.all(target[103:, 0] == 0.0)

    # Check MU2 at 0.2s
    assert np.sum(target[:, 1]) == 5
    assert np.all(target[198:203, 1] == 1.0)

    # Check MU3 at 0.3s (handles AI: MUP correctly)
    assert np.sum(target[:, 2]) == 5
    assert np.all(target[298:303, 2] == 1.0)

def test_build_pulse_train_invalid_inputs():
    """Test invalid input dimensions for build_pulse_train."""
    with pytest.raises(ValueError, match="num_samples must be > 0"):
        build_pulse_train(0, 1000.0, [], 3)

    with pytest.raises(ValueError, match="num_samples must be > 0"):
        build_pulse_train(-10, 1000.0, [], 3)

    with pytest.raises(ValueError, match="num_mus must be > 0"):
        build_pulse_train(1000, 1000.0, [], 0)

    with pytest.raises(ValueError, match="num_mus must be > 0"):
        build_pulse_train(1000, 1000.0, [], -5)

def test_build_pulse_train_filters():
    """Test ignoring wrong labels, bad MU IDs, etc."""
    num_samples = 1000
    sample_rate = 1000.0
    num_mus = 2

    annotations = [
        {"label": "ARTIFACT", "mu_id": "MU1", "start_time": 0.1, "end_time": 0.1},  # Wrong label
        {"label": "MUP", "mu_id": "NOTAMU", "start_time": 0.2, "end_time": 0.2},  # Bad ID prefix
        {"label": "MUP", "mu_id": "MU99", "start_time": 0.3, "end_time": 0.3},    # Out of bounds MU
        {"label": "MUP", "mu_id": "MU-invalid", "start_time": 0.4, "end_time": 0.4}, # Unparseable
    ]

    target = build_pulse_train(num_samples, sample_rate, annotations, num_mus)

    # Should all be zero
    assert np.sum(target) == 0.0

def test_build_pulse_train_edge_clipping():
    """Test when pulse is at the start or end, it gets properly clipped."""
    num_samples = 100
    sample_rate = 1000.0
    num_mus = 1
    pulse_width_ms = 10.0
    # fs=1000 => pulse_width_ms=10 => width=10, half_width=5

    annotations = [
        {"label": "MUP", "mu_id": "MU1", "start_time": 0.001, "end_time": 0.001},  # center=1 => lo=max(0,-4)=0, hi=min(100, 7)=7
        {"label": "MUP", "mu_id": "MU1", "start_time": 0.099, "end_time": 0.099},  # center=99 => lo=max(0,94)=94, hi=min(100, 105)=100
    ]

    target = build_pulse_train(num_samples, sample_rate, annotations, num_mus, pulse_width_ms)

    # Pulse 1 at start
    assert target[0, 0] == 1.0
    assert target[6, 0] == 1.0
    assert target[7, 0] == 0.0

    # Pulse 2 at end
    assert target[93, 0] == 0.0
    assert target[94, 0] == 1.0
    assert target[99, 0] == 1.0
