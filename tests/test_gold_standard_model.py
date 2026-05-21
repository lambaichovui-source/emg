import math
import pytest
from gold_standard_model import classify_emg_event

@pytest.mark.parametrize(
    "features_dict, expected_label",
    [
        # duration < 1.0, n_spikes <= 1 -> "Spike"
        ({"duration": 0.5, "n_spikes": 1}, "Spike"),
        ({"duration": 0.99, "n_spikes": 0}, "Spike"),

        # duration < 1.0, n_spikes > 1 -> "Burst"
        ({"duration": 0.5, "n_spikes": 2}, "Burst"),
        ({"duration": 0.1, "n_spikes": 10}, "Burst"),

        # duration >= 1.0, mean_fr >= 60.0, cv <= 0.15 -> "A-Train (High Risk)"
        ({"duration": 1.0, "mean_firing_rate": 60.0, "rhythmicity_cv": 0.15}, "A-Train (High Risk)"),
        ({"duration": 2.5, "mean_firing_rate": 80.0, "rhythmicity_cv": 0.10}, "A-Train (High Risk)"),

        # duration >= 1.0, cv > 0.45 -> "C-Train (Polyrhythmic)"
        ({"duration": 1.0, "rhythmicity_cv": 0.46}, "C-Train (Polyrhythmic)"),
        ({"duration": 1.5, "mean_firing_rate": 60.0, "rhythmicity_cv": 0.5}, "C-Train (Polyrhythmic)"),

        # duration >= 1.0, other cases -> "B-Train"
        ({"duration": 1.0, "mean_firing_rate": 50.0, "rhythmicity_cv": 0.15}, "B-Train"),
        ({"duration": 1.5, "mean_firing_rate": 70.0, "rhythmicity_cv": 0.30}, "B-Train"),
        ({"duration": 5.0, "mean_firing_rate": 0.0, "rhythmicity_cv": 0.0}, "B-Train"),

        # Empty dictionary or missing values (defaults: duration 0.0, n_spikes 0) -> "Spike"
        ({}, "Spike"),
        ({"some_other_key": "value"}, "Spike"),

        # duration is NaN (falls through all < and >= checks) -> "Burst"
        ({"duration": float("nan")}, "Burst"),
    ]
)
def test_classify_emg_event(features_dict, expected_label):
    assert classify_emg_event(features_dict) == expected_label

def test_classify_emg_event_type_conversion():
    # Test that values are properly converted to float/int
    features = {
        "duration": "1.5",
        "mean_firing_rate": "65.0",
        "rhythmicity_cv": "0.1",
        "n_spikes": "5"
    }
    # duration >= 1.0, mean_fr >= 60, cv <= 0.15 -> "A-Train (High Risk)"
    assert classify_emg_event(features) == "A-Train (High Risk)"

def test_classify_emg_event_type_conversion_spike():
    # Test that values are properly converted to float/int
    features = {
        "duration": "0.5",
        "n_spikes": "1"
    }
    assert classify_emg_event(features) == "Spike"
