import pytest
import numpy as np
from ai_model_v2 import mapminmax_normalize

def test_mapminmax_normalize_basic():
    """Test basic normalization to the default [-1, 1] range."""
    # Two channels, three samples each
    x = np.array([
        [0.0, 5.0, 10.0],
        [-10.0, 0.0, 10.0]
    ])

    expected = np.array([
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0]
    ], dtype=np.float32)

    result = mapminmax_normalize(x)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert result.dtype == np.float32

def test_mapminmax_normalize_custom_range():
    """Test normalization to a custom range."""
    x = np.array([
        [0.0, 5.0, 10.0],
        [-10.0, 0.0, 10.0]
    ])

    # Custom range [0, 1]
    expected = np.array([
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0]
    ], dtype=np.float32)

    result = mapminmax_normalize(x, feature_range=(0.0, 1.0))
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert result.dtype == np.float32

def test_mapminmax_normalize_invalid_ndim():
    """Test that ValueError is raised for inputs that are not 2-D."""
    x_1d = np.array([0.0, 5.0, 10.0])
    with pytest.raises(ValueError, match="Expected 2-D array"):
        mapminmax_normalize(x_1d)

    x_3d = np.array([[[0.0, 5.0, 10.0]]])
    with pytest.raises(ValueError, match="Expected 2-D array"):
        mapminmax_normalize(x_3d)

def test_mapminmax_normalize_invalid_range():
    """Test that ValueError is raised for invalid feature_range (hi <= lo)."""
    x = np.array([[0.0, 5.0, 10.0]])
    with pytest.raises(ValueError, match="feature_range must satisfy hi > lo"):
        mapminmax_normalize(x, feature_range=(1.0, -1.0))

    with pytest.raises(ValueError, match="feature_range must satisfy hi > lo"):
        mapminmax_normalize(x, feature_range=(1.0, 1.0))

def test_mapminmax_normalize_constant_array():
    """Test handling of constant arrays (preventing division by zero).
    The algorithm handles a constant array by making (x - ch_min) = 0 and
    denom = 1e-8.
    The formula is (0 / 1e-8) * (hi - lo) + lo = 0 + lo = lo.
    So all values should map to `lo` (the lower bound of the feature_range).
    """
    x = np.array([
        [5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0]
    ])

    expected = np.array([
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0]
    ], dtype=np.float32)

    result = mapminmax_normalize(x)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert result.dtype == np.float32
