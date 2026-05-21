import pytest
import numpy as np
from ai_model_v2 import build_pulse_trains_from_annotations

def test_happy_path():
    time_arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    annotations = [
        {"start_time": 0.5, "end_time": 1.5, "mu_id": "MU1"}, # center 1.0 -> idx 1
        {"start_time": 1.5, "end_time": 2.5, "mu_id": "MU2"}, # center 2.0 -> idx 2
        {"start_time": 2.5, "end_time": 3.5, "mu_id": "MU1"}, # center 3.0 -> idx 3
    ]
    result = build_pulse_trains_from_annotations(time_arr, annotations)
    assert list(result.keys()) == ["MU1", "MU2"] or list(result.keys()) == ["MU2", "MU1"]
    np.testing.assert_array_equal(result["MU1"], np.array([1, 3], dtype=np.int32))
    np.testing.assert_array_equal(result["MU2"], np.array([2], dtype=np.int32))

def test_fallback_keys():
    time_arr = np.array([0.0, 1.0, 2.0])
    annotations = [
        {"start_time": 0.8, "end_time": 1.2, "mu": "MU_A"},
        {"start_time": 1.8, "end_time": 2.2, "motor_unit": "MU_B"},
        {"start_time": 0.0, "end_time": 0.2, "unit_id": "MU_C"},
        {"start_time": 0.0, "end_time": 0.2, "unit": "MU_D"},
    ]
    result = build_pulse_trains_from_annotations(time_arr, annotations)
    assert "MU_A" in result
    assert "MU_B" in result
    assert "MU_C" in result
    assert "MU_D" in result

def test_missing_times():
    time_arr = np.array([0.0, 1.0, 2.0, 3.0])
    # missing start_time -> defaults to 0.0, center = (0.0 + end_time)/2
    # missing end_time -> defaults to start_time, center = start_time
    annotations = [
        {"end_time": 2.0, "mu_id": "MU1"}, # center 1.0 -> idx 1
        {"start_time": 2.0, "mu_id": "MU2"}, # center 2.0 -> idx 2
    ]
    result = build_pulse_trains_from_annotations(time_arr, annotations)
    np.testing.assert_array_equal(result["MU1"], np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(result["MU2"], np.array([2], dtype=np.int32))

def test_missing_valid_mu_id():
    time_arr = np.array([0.0, 1.0, 2.0])
    annotations = [
        {"start_time": 1.0, "end_time": 1.0}, # No MU key
        {"start_time": 1.0, "end_time": 1.0, "mu_id": "  "}, # Empty MU key
        {"start_time": 1.0, "end_time": 1.0, "mu_id": "ValidMU"},
    ]
    result = build_pulse_trains_from_annotations(time_arr, annotations)
    assert list(result.keys()) == ["ValidMU"]
    np.testing.assert_array_equal(result["ValidMU"], np.array([1], dtype=np.int32))

def test_deduplication():
    time_arr = np.array([0.0, 1.0, 2.0])
    annotations = [
        {"start_time": 0.5, "end_time": 1.5, "mu_id": "MU1"}, # center 1.0 -> idx 1
        {"start_time": 0.8, "end_time": 1.2, "mu_id": "MU1"}, # center 1.0 -> idx 1
        {"start_time": 0.9, "end_time": 1.1, "mu_id": "MU1"}, # center 1.0 -> idx 1
    ]
    result = build_pulse_trains_from_annotations(time_arr, annotations)
    np.testing.assert_array_equal(result["MU1"], np.array([1], dtype=np.int32))

def test_out_of_bounds():
    time_arr = np.array([0.0, 1.0, 2.0])
    annotations = [
        # searchsorted with side='left' on [-7.5] against [0.0, 1.0, 2.0] gives 0 (valid index according to bounds check).
        # We test the actual logic implemented in the file.
        {"start_time": -10.0, "end_time": -5.0, "mu_id": "MU1"},
        {"start_time": 10.0, "end_time": 20.0, "mu_id": "MU1"}, # Center 15.0 -> searchsorted returns 3 (out of bounds)
        {"start_time": 0.5, "end_time": 1.5, "mu_id": "MU2"}
    ]
    result = build_pulse_trains_from_annotations(time_arr, annotations)
    assert "MU1" in result
    np.testing.assert_array_equal(result["MU1"], np.array([0], dtype=np.int32))
    assert "MU2" in result
    np.testing.assert_array_equal(result["MU2"], np.array([1], dtype=np.int32))

def test_error_time_arr_ndim():
    time_arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    annotations = [{"start_time": 0.0, "end_time": 1.0, "mu_id": "MU1"}]
    with pytest.raises(ValueError, match="time_arr must be 1-D."):
        build_pulse_trains_from_annotations(time_arr, annotations)

def test_error_time_arr_length():
    time_arr = np.array([0.0])
    annotations = [{"start_time": 0.0, "end_time": 1.0, "mu_id": "MU1"}]
    with pytest.raises(ValueError, match="time_arr must contain at least two samples."):
        build_pulse_trains_from_annotations(time_arr, annotations)

def test_error_no_mu_ids():
    time_arr = np.array([0.0, 1.0, 2.0])
    annotations = [{"start_time": 0.0, "end_time": 1.0}]
    with pytest.raises(ValueError, match="No MU IDs found in annotations."):
        build_pulse_trains_from_annotations(time_arr, annotations)
