"""
neurotonic_logic.py — Deterministic ISI / frequency rules for neurotonic events.

Stage-2 classifier for pulse trains produced by FastICA peak picking (Stage 1).
All logic is mathematical; no machine learning.
"""

from dataclasses import dataclass

import numpy as np

from data_handler import calc_isi_features


@dataclass
class NeurotonicThresholds:
    """Clinical thresholds for Spike / Burst / Train A–C / Artifact."""

    max_group_gap_s: float = 0.12
    burst_max_duration_s: float = 1.0
    train_min_duration_s: float = 1.0
    min_burst_spikes: int = 2
    min_train_spikes: int = 3

    artifact_center_hz: float = 60.0
    artifact_line_tolerance_hz: float = 1.0
    artifact_var_max_s2: float = 0.00005

    burst_min_frequency_hz: float = 50.0
    train_a_min_frequency_hz: float = 40.0
    train_b_min_frequency_hz: float = 3.0
    train_b_max_frequency_hz: float = 40.0
    train_c_min_isi_variance_s2: float = 0.0025


class NeurotonicClassifier:
    """Classify pulse timestamps into clinical neurotonic event labels."""

    def __init__(self, thresholds: NeurotonicThresholds | None = None):
        self._t = thresholds or NeurotonicThresholds()

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def _max_continuous_group(self, timestamps_s: np.ndarray) -> tuple[int, float]:
        """Return (max_group_spikes, max_group_duration_s) for contiguous pulse groups."""
        n = int(timestamps_s.size)
        if n <= 0:
            return 0, 0.0
        if n == 1:
            return 1, 0.0

        ts = np.sort(timestamps_s.astype(np.float64, copy=False))
        best_count = 1
        best_dur = 0.0
        start = 0
        for i in range(1, n):
            if (ts[i] - ts[i - 1]) > self._t.max_group_gap_s:
                count = i - start
                dur = float(ts[i - 1] - ts[start]) if count > 1 else 0.0
                if (count > best_count) or (count == best_count and dur > best_dur):
                    best_count = count
                    best_dur = dur
                start = i

        count = n - start
        dur = float(ts[n - 1] - ts[start]) if count > 1 else 0.0
        if (count > best_count) or (count == best_count and dur > best_dur):
            best_count = count
            best_dur = dur
        return int(best_count), float(best_dur)

    def largest_group_timestamps(self, timestamps_s: np.ndarray) -> np.ndarray:
        """Timestamps belonging to the longest contiguous pulse group."""
        ts = np.sort(timestamps_s.astype(np.float64, copy=False))
        n = ts.size
        if n <= 1:
            return ts

        best_start = 0
        best_end = 0
        best_count = 1
        start = 0
        for i in range(1, n):
            if (ts[i] - ts[i - 1]) > self._t.max_group_gap_s:
                count = i - start
                if count > best_count:
                    best_count = count
                    best_start = start
                    best_end = i - 1
                start = i
        count = n - start
        if count > best_count:
            best_start = start
            best_end = n - 1
        return ts[best_start : best_end + 1]

    def classify_window(
        self,
        pulse_timestamps_s: np.ndarray,
        *,
        history_timestamps_s: np.ndarray | None = None,
        window_s: float = 1.0,
    ) -> tuple[str, float, dict]:
        """
        Classify activity using a 1-second window plus optional multi-second history.

        Labels: 'Artifact', 'Spike', 'Burst', 'Train A', 'Train B', 'Train C', 'None'
        """
        window_s = float(max(window_s, 1e-6))
        recent = np.sort(pulse_timestamps_s.astype(np.float64, copy=False))
        history = (
            np.sort(history_timestamps_s.astype(np.float64, copy=False))
            if history_timestamps_s is not None and history_timestamps_s.size > 0
            else recent
        )

        pulses_1s = int(recent.size)
        mean_freq_hz, isi_var_s2 = calc_isi_features(recent)
        max_group_spikes, max_group_duration_s = self._max_continuous_group(recent)
        pulse_density_hz = pulses_1s / window_s

        hist_group_spikes, hist_group_duration_s = self._max_continuous_group(history)
        group_ts = self.largest_group_timestamps(history)
        group_freq_hz, group_var_s2 = calc_isi_features(group_ts)

        metrics = {
            "mean_frequency_hz": float(mean_freq_hz),
            "isi_variance_s2": float(isi_var_s2),
            "pulse_count": pulses_1s,
            "pulse_density_hz": float(pulse_density_hz),
            "max_group_spikes": int(max_group_spikes),
            "continuous_duration_s": float(max_group_duration_s),
            "history_group_spikes": int(hist_group_spikes),
            "history_group_duration_s": float(hist_group_duration_s),
            "group_frequency_hz": float(group_freq_hz),
            "group_isi_variance_s2": float(group_var_s2),
        }

        if pulses_1s <= 0:
            return "None", 0.0, metrics

        # Artifact: 60 Hz (+/- 1 Hz) line noise with near-zero ISI variance.
        near_60 = abs(mean_freq_hz - self._t.artifact_center_hz) <= self._t.artifact_line_tolerance_hz
        if near_60 and isi_var_s2 <= self._t.artifact_var_max_s2:
            closeness = abs(mean_freq_hz - self._t.artifact_center_hz)
            conf = 0.6 + 0.4 * self._clip01(
                1.0 - closeness / max(self._t.artifact_line_tolerance_hz, 1e-6)
            )
            return "Artifact", self._clip01(conf), metrics

        # Spike: exactly one pulse in the rolling second.
        if pulses_1s == 1:
            return "Spike", 0.85, metrics

        # Burst: multiple pulses, > 50 Hz, grouped duration strictly < 1 s.
        if (
            pulses_1s >= self._t.min_burst_spikes
            and mean_freq_hz > self._t.burst_min_frequency_hz
            and max_group_duration_s < self._t.burst_max_duration_s
        ):
            conf = (
                0.5
                + 0.25 * self._clip01((mean_freq_hz - self._t.burst_min_frequency_hz) / 50.0)
                + 0.25 * self._clip01(pulses_1s / 10.0)
            )
            return "Burst", self._clip01(conf), metrics

        # Sustained trains require > 1 s continuity in history (not only the 1 s window).
        if (
            hist_group_spikes >= self._t.min_train_spikes
            and hist_group_duration_s > self._t.train_min_duration_s
        ):
            freq = float(group_freq_hz if group_freq_hz > 0.0 else mean_freq_hz)
            var = float(group_var_s2 if group_var_s2 > 0.0 else isi_var_s2)

            if freq > self._t.train_a_min_frequency_hz:
                conf = 0.55 + 0.45 * self._clip01((freq - self._t.train_a_min_frequency_hz) / 40.0)
                return "Train A", self._clip01(conf), metrics

            if self._t.train_b_min_frequency_hz <= freq <= self._t.train_b_max_frequency_hz:
                mid = 0.5 * (self._t.train_b_min_frequency_hz + self._t.train_b_max_frequency_hz)
                span = max(self._t.train_b_max_frequency_hz - self._t.train_b_min_frequency_hz, 1e-6)
                conf = 0.55 + 0.45 * self._clip01(1.0 - abs(freq - mid) / span)
                return "Train B", self._clip01(conf), metrics

            if var >= self._t.train_c_min_isi_variance_s2:
                conf = 0.55 + 0.45 * self._clip01(var / max(self._t.train_c_min_isi_variance_s2 * 4.0, 1e-9))
                return "Train C", self._clip01(conf), metrics

        return "None", 0.15, metrics

    def classify(
        self,
        *,
        history_timestamps_s: np.ndarray,
        mean_frequency_hz: float,
        isi_variance_s2: float,
        rolling_amplitude_uv: float,
        pulse_count_1s: int,
        window_s: float = 1.0,
    ) -> tuple[str, float, dict]:
        """
        Backward-compatible wrapper used by legacy inference threads.

        Reconstructs the 1-second slice from history using the last ``window_s`` seconds.
        """
        hist = history_timestamps_s.astype(np.float64, copy=False)
        if hist.size == 0:
            recent = hist
        else:
            t_end = float(hist[-1])
            recent = hist[hist >= (t_end - float(window_s))]

        label, confidence, metrics = self.classify_window(
            recent,
            history_timestamps_s=hist,
            window_s=window_s,
        )
        metrics["rolling_amplitude_uv"] = float(rolling_amplitude_uv)
        metrics["mean_frequency_hz"] = float(metrics.get("mean_frequency_hz", mean_frequency_hz))
        metrics["isi_variance_s2"] = float(metrics.get("isi_variance_s2", isi_variance_s2))
        metrics["pulse_count"] = int(pulse_count_1s)
        return label, confidence, metrics
