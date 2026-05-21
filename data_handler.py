"""
data_handler.py — CSV parsing, Ring Buffers, medical-grade signal
preprocessing, and Expert Feature extraction for IONM EMG data.

Preprocessing pipeline (SignalPreprocessor):
  1. 60 Hz + 120 Hz IIR notch filters  (scipy.signal, stateful per-chunk)
  2. IIR High-Pass (Baseline Correction) – 1st-order, ~15 Hz cutoff, zero DC drift
  3. SEP Artifact Reduction            – median filter (3-5 pt) + optional trigger blanking
  4. Cautery Blanking                  – amplitude threshold → hold output at 0

Feature functions use Numba @njit for near-C speed.  Two sets exist:
  • Basic features   — RMS, ZCR, Waveform Length, Peak-to-Peak
  • Romstöck features — RMS, MAV, Rectified Peak Envelope,
                        Symmetry Index, Envelope Decrescendo,
                        Spike/Burst Density
"""

import os
import csv
import threading
import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos
from scipy.ndimage import median_filter as _median_filter1d


# ━━ Basic Expert Features (@njit) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@njit(cache=True)
def calc_rms(signal):
    """Root Mean Square of a 1-D signal window → single float."""
    acc = 0.0
    n = signal.shape[0]
    for i in range(n):
        acc += signal[i] * signal[i]
    return np.sqrt(acc / n)


@njit(cache=True)
def calc_zcr(signal):
    """Zero-Crossing Rate normalised to [0, 1] → single float."""
    n = signal.shape[0]
    if n < 2:
        return 0.0
    crossings = 0
    for i in range(1, n):
        if (signal[i - 1] >= 0.0 and signal[i] < 0.0) or \
           (signal[i - 1] < 0.0 and signal[i] >= 0.0):
            crossings += 1
    return crossings / (n - 1)


@njit(cache=True)
def calc_waveform_length(signal):
    """Waveform Length — cumulative absolute first-difference → single float."""
    wl = 0.0
    for i in range(1, signal.shape[0]):
        wl += abs(signal[i] - signal[i - 1])
    return wl


@njit(cache=True)
def calc_peak_to_peak(signal):
    """Peak-to-peak amplitude (max − min) → single float."""
    vmin = signal[0]
    vmax = signal[0]
    for i in range(1, signal.shape[0]):
        if signal[i] < vmin:
            vmin = signal[i]
        if signal[i] > vmax:
            vmax = signal[i]
    return vmax - vmin


@njit(cache=True)
def calc_isi_features(timestamps):
    """Inter-spike interval stats -> (mean_frequency_hz, isi_variance_s2)."""
    n = timestamps.shape[0]
    if n < 2:
        return 0.0, 0.0

    m = n - 1
    sum_isi = 0.0
    for i in range(m):
        dt = timestamps[i + 1] - timestamps[i]
        if dt > 0.0:
            sum_isi += dt
    mean_isi = sum_isi / max(m, 1)
    if mean_isi <= 1e-8:
        return 0.0, 0.0

    var_isi = 0.0
    for i in range(m):
        dt = timestamps[i + 1] - timestamps[i]
        d = dt - mean_isi
        var_isi += d * d
    var_isi /= max(m, 1)
    return 1.0 / mean_isi, var_isi


@njit(cache=True)
def calc_rolling_amplitude(raw_signal, timestamps):
    """Mean local peak-to-peak amplitude around detected pulse indices."""
    n = raw_signal.shape[0]
    k = timestamps.shape[0]
    if n == 0 or k == 0:
        return 0.0

    half_win = 4
    acc = 0.0
    valid = 0
    for i in range(k):
        idx = int(timestamps[i])
        if idx < 0 or idx >= n:
            continue
        lo = idx - half_win
        hi = idx + half_win + 1
        if lo < 0:
            lo = 0
        if hi > n:
            hi = n
        if hi - lo <= 1:
            continue
        vmin = raw_signal[lo]
        vmax = raw_signal[lo]
        for j in range(lo + 1, hi):
            v = raw_signal[j]
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
        acc += (vmax - vmin)
        valid += 1
    if valid == 0:
        return 0.0
    return acc / valid


@njit(cache=True)
def detect_and_blank_artifacts(signal_chunk, threshold=2000.0):
    """Blank impossible signal excursions to 0 baseline."""
    n = signal_chunk.shape[0]
    if n == 0:
        return signal_chunk.copy()
    out = signal_chunk.copy()
    roc_threshold = threshold * 0.75
    if roc_threshold < 1200.0:
        roc_threshold = 1200.0

    # First pass: mark artifact samples by amplitude or impossible slope.
    for i in range(n):
        amp_bad = abs(signal_chunk[i]) > threshold
        roc_bad = False
        if i > 0:
            roc_bad = abs(signal_chunk[i] - signal_chunk[i - 1]) > roc_threshold
        if amp_bad or roc_bad:
            out[i] = 0.0
            if i > 0:
                out[i - 1] = 0.0
            if i + 1 < n:
                out[i + 1] = 0.0
    return out


# ━━ Romstöck Classification Features (@njit) ━━━━━━━━━━━━━━━━━━━━━━━━

@njit(cache=True)
def calc_mav(signal):
    """Mean Absolute Value — rectification + averaging → single float."""
    acc = 0.0
    n = signal.shape[0]
    for i in range(n):
        acc += abs(signal[i])
    return acc / n


@njit(cache=True)
def calc_rectified_moving_avg(signal):
    """Peak of the rectified moving average (smoothed envelope peak).

    A sliding-sum approach gives O(n) complexity.  The sub-window is
    ~1 % of the total window (minimum 4 samples).
    """
    n = signal.shape[0]
    sub_win = max(n // 100, 4)
    if n < sub_win:
        return 0.0

    running = 0.0
    for j in range(sub_win):
        running += abs(signal[j])
    peak = running / sub_win

    for i in range(1, n - sub_win + 1):
        running += abs(signal[i + sub_win - 1]) - abs(signal[i - 1])
        avg = running / sub_win
        if avg > peak:
            peak = avg
    return peak


@njit(cache=True)
def calc_symmetry_index(signal):
    """Symmetry between positive and negative half-cycles.

    Returns 1.0 for a perfectly symmetric (sinusoidal) waveform and
    approaches 0.0 for highly asymmetric shapes.  A-trains score high.
    """
    pos_sum = 0.0
    pos_n = 0
    neg_sum = 0.0
    neg_n = 0
    for i in range(signal.shape[0]):
        if signal[i] >= 0.0:
            pos_sum += signal[i]
            pos_n += 1
        else:
            neg_sum += -signal[i]
            neg_n += 1

    pos_mean = pos_sum / max(pos_n, 1)
    neg_mean = neg_sum / max(neg_n, 1)
    denom = pos_mean + neg_mean
    if denom < 1e-10:
        return 1.0
    return 1.0 - abs(pos_mean - neg_mean) / denom


@njit(cache=True)
def calc_envelope_decrescendo(signal):
    """Normalised slope of the amplitude envelope.

    Negative → classic A-train decrescendo (rapid onset, gradual decay).
    Positive → crescendo.  Near zero → flat / noise.
    """
    n = signal.shape[0]
    env_win = max(n // 20, 4)
    num_pts = n - env_win + 1
    if num_pts < 2:
        return 0.0

    envelope = np.empty(num_pts, dtype=np.float64)
    for i in range(num_pts):
        s = 0.0
        for j in range(env_win):
            s += signal[i + j] * signal[i + j]
        envelope[i] = np.sqrt(s / env_win)

    x_mean = (num_pts - 1) / 2.0
    y_mean = 0.0
    for i in range(num_pts):
        y_mean += envelope[i]
    y_mean /= num_pts

    num = 0.0
    den = 0.0
    for i in range(num_pts):
        dx = i - x_mean
        num += dx * (envelope[i] - y_mean)
        den += dx * dx
    if den < 1e-10 or y_mean < 1e-10:
        return 0.0
    return (num / den) / y_mean


@njit(cache=True)
def calc_spike_burst_density(signal):
    """Peaks exceeding 3 σ above baseline, normalised to peaks-per-sample.

    High values → B-trains (regular/irregular spikes) or C-trains
    (continuous interference).  Low values → quiet or A-train.
    """
    n = signal.shape[0]
    if n < 3:
        return 0.0

    mean_v = 0.0
    for i in range(n):
        mean_v += signal[i]
    mean_v /= n

    var_v = 0.0
    for i in range(n):
        d = signal[i] - mean_v
        var_v += d * d
    var_v /= n

    threshold = mean_v + 3.0 * np.sqrt(var_v)
    count = 0
    for i in range(1, n - 1):
        if signal[i] > threshold \
           and signal[i] > signal[i - 1] \
           and signal[i] > signal[i + 1]:
            count += 1
    return count / (n - 2)


# ━━ Windowed Feature Extraction (@njit) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROMSTOCK_FEATURE_NAMES = [
    "rms", "mav", "rectified_peak_envelope",
    "symmetry_index", "envelope_decrescendo", "spike_burst_density",
]
NUM_ROMSTOCK_FEATURES = len(ROMSTOCK_FEATURE_NAMES)

DEFAULT_FEATURE_WINDOW = 2000    # 200 ms @ 10 kHz
DEFAULT_FEATURE_STRIDE = 1000    # 100 ms  (50 % overlap)


@njit(cache=True)
def extract_all_features(signal, window_size, stride):
    """Compute all 6 Romstöck features over sliding windows.

    Parameters
    ----------
    signal      : 1-D float32/64 array (single channel)
    window_size : samples per window
    stride      : step between successive windows

    Returns
    -------
    features : (6, num_windows) float64 array
    """
    n = signal.shape[0]
    if n < window_size:
        return np.empty((6, 0), dtype=np.float64)

    num_win = (n - window_size) // stride + 1
    out = np.empty((6, num_win), dtype=np.float64)

    for w in range(num_win):
        s = w * stride
        win = signal[s : s + window_size]
        out[0, w] = calc_rms(win)
        out[1, w] = calc_mav(win)
        out[2, w] = calc_rectified_moving_avg(win)
        out[3, w] = calc_symmetry_index(win)
        out[4, w] = calc_envelope_decrescendo(win)
        out[5, w] = calc_spike_burst_density(win)

    return out


def detect_micro_bursts(
    signal: np.ndarray,
    sample_rate: float,
    rms_window_ms: float = 50.0,
    min_duration_ms: float = 20.0,
    threshold_k: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Detect active micro-bursts inside an ROI.

    Returns
    -------
    bursts_idx : (N,2) int array of [start_idx, end_idx)
    envelope   : RMS envelope used for thresholding
    threshold  : dynamic threshold value
    """
    if signal.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32), 0.0

    fs = max(float(sample_rate), 1.0)
    rect = np.abs(signal.astype(np.float32))
    win = max(1, int(rms_window_ms * fs / 1000.0))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    envelope = np.sqrt(np.convolve(rect * rect, kernel, mode="same"))

    baseline_n = max(1, int(0.1 * envelope.size))
    baseline = envelope[:baseline_n]
    threshold = float(np.mean(baseline) + threshold_k * np.std(baseline))

    above = envelope > threshold
    if not np.any(above):
        return np.empty((0, 2), dtype=np.int32), envelope.astype(np.float32), threshold

    d = np.diff(above.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if above[0]:
        starts = np.concatenate(([0], starts))
    if above[-1]:
        ends = np.concatenate((ends, [above.size]))

    min_len = max(1, int(min_duration_ms * fs / 1000.0))
    valid = (ends - starts) >= min_len
    if not np.any(valid):
        spans = np.empty((0, 2), dtype=np.int32)
    else:
        spans = np.column_stack((starts[valid], ends[valid])).astype(np.int32)
    return spans, envelope.astype(np.float32), threshold


# ━━ Ring Buffer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RingBuffer:
    """Fixed-capacity circular buffer backed by a pre-allocated NumPy array.

    Shape: (num_channels, capacity).  Oldest samples are silently
    overwritten when the buffer is full — no list.append() ever needed.
    """

    def __init__(self, capacity: int, num_channels: int = 1,
                 dtype=np.float32):
        self._buf = np.zeros((num_channels, capacity), dtype=dtype)
        self._capacity = capacity
        self._num_ch = num_channels
        self._write_idx = 0
        self._count = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def count(self) -> int:
        return min(self._count, self._capacity)

    @property
    def is_full(self) -> bool:
        return self._count >= self._capacity

    def append(self, data: np.ndarray) -> None:
        """Write *data* shaped (num_channels, n_samples) into the ring."""
        with self._lock:
            n = data.shape[-1]
            if n == 0:
                return
            if n >= self._capacity:
                self._buf[:] = data[:, -self._capacity:]
                self._write_idx = 0
                self._count = self._capacity
                return

            end = self._write_idx + n
            if end <= self._capacity:
                self._buf[:, self._write_idx:end] = data
            else:
                first = self._capacity - self._write_idx
                self._buf[:, self._write_idx:] = data[:, :first]
                self._buf[:, :n - first] = data[:, first:]

            self._write_idx = end % self._capacity
            self._count += n

    def _get_ordered_unlocked(self) -> np.ndarray:
        """Return contents in chronological order without lock acquisition."""
        if self._count == 0:
            return self._buf[:, :0]
        if self._count <= self._capacity:
            return self._buf[:, :self._count].copy()
        return np.concatenate(
            [self._buf[:, self._write_idx:], self._buf[:, :self._write_idx]],
            axis=1,
        )

    def get_ordered(self) -> np.ndarray:
        """Return contents in chronological order → (num_channels, count)."""
        with self._lock:
            return self._get_ordered_unlocked()

    def get_last_seconds(self, sample_rate: float, seconds: float) -> np.ndarray:
        """Thread-safe slice of most recent N seconds."""
        fs = max(float(sample_rate), 1.0)
        want = max(1, int(fs * max(float(seconds), 0.0)))
        with self._lock:
            ordered = self._get_ordered_unlocked()
            n = ordered.shape[1]
            if n <= want:
                return ordered.copy()
            return ordered[:, -want:].copy()

    def reset(self) -> None:
        with self._lock:
            self._buf[:] = 0
            self._write_idx = 0
            self._count = 0


# ━━ Signal Preprocessor ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SignalPreprocessor:
    """Medical-grade preprocessing pipeline for multi-channel EMG.

    Per-chunk operations (maintain IIR state across chunks):
      • 60 Hz notch filter   – removes AC powerline fundamental
      • 120 Hz notch filter  – removes first harmonic

    Full-signal operations (applied once after all chunks concatenated):
      • Cautery blanking     – zeros regions that exceed an extreme
                               voltage threshold ± a safety margin
      • SEP blanking         – detects brief periodic impulses from
                               somatosensory stimulation and replaces
                               them with linear interpolation
    """

    def __init__(
        self,
        sample_rate: int = 10_000,
        num_channels: int = 16,
        notch_freqs: tuple[float, ...] = (60.0, 120.0),
        notch_q: float = 30.0,
        cautery_threshold: float = 5000.0,
        cautery_margin_ms: float = 5.0,
        sep_threshold_std: float = 5.0,
        sep_blank_ms: float = 3.0,
    ):
        self._fs = sample_rate
        self._num_ch = num_channels

        self._cautery_thresh = cautery_threshold
        self._cautery_margin = max(1, int(cautery_margin_ms * sample_rate / 1000))
        self._sep_thresh_std = sep_threshold_std
        self._sep_half_blank = max(1, int(sep_blank_ms * sample_rate / 1000) // 2)
        self._sep_max_dur = int(5.0 * sample_rate / 1000)

        self._sos_list: list[np.ndarray] = []
        self._zi: list[list[np.ndarray]] = []

        for freq in notch_freqs:
            if freq >= sample_rate / 2:
                continue
            b, a = iirnotch(freq, notch_q, fs=sample_rate)
            sos = tf2sos(b, a)
            self._sos_list.append(sos)

            zi_template = sosfilt_zi(sos)
            self._zi.append([zi_template.copy() for _ in range(num_channels)])

    # ── per-chunk: notch filtering (stateful) ─────────────────────────

    def apply_notch(self, channels: np.ndarray) -> np.ndarray:
        """Apply all notch filters to a (num_ch, chunk_len) array in-place-ish."""
        out = channels.copy()
        for f_idx, sos in enumerate(self._sos_list):
            for ch in range(min(out.shape[0], self._num_ch)):
                zi = self._zi[f_idx][ch]
                filtered, self._zi[f_idx][ch] = sosfilt(
                    sos, out[ch], zi=zi,
                )
                out[ch] = filtered.astype(np.float32)
        return out

    # ── full-signal: artifact blanking ────────────────────────────────

    def blank_artifacts(self, channels: np.ndarray) -> np.ndarray:
        """Cautery + SEP blanking on the full concatenated signal (default parameters)."""
        out = channels.copy()
        default_hold = max(1, int(150.0 * self._fs / 1000))   # 150 ms hold
        for ch in range(out.shape[0]):
            out[ch] = self._blank_cautery_hold(out[ch], threshold=self._cautery_thresh, hold_samples=default_hold)
            out[ch] = self._reduce_sep(out[ch], median_window=5)
        return out

    def apply_user_filters(
        self,
        channels: np.ndarray,
        *,
        show_raw: bool,
        enable_notch: bool,
        notch_freq: float,
        enable_cautery: bool,
        enable_sep: bool,
        enable_baseline: bool,
        baseline_cutoff_hz: float,
        cautery_threshold: float,
        cautery_hold_ms: float,
        sep_blank_ms: float,
        sep_trigger: bool,
        low_cut_hz: float,
        high_cut_hz: float,
    ) -> np.ndarray:
        """Apply all user-selected artifact reduction stages.

        Pipeline order (when enabled):
          1. Notch + band-pass (existing)
          2. Baseline correction  – IIR high-pass removes DC / slow drift
          3. SEP reduction        – median filter, optional trigger blanking
          4. Cautery blanking     – amplitude-hold zeros
        """
        out = channels.copy().astype(np.float32)
        if show_raw:
            return out

        nyq = max(self._fs * 0.5, 1.0)

        # ── 1. Notch + bandpass (unchanged) ──────────────────────────
        if enable_notch and 0.5 < notch_freq < nyq:
            b, a = iirnotch(notch_freq, 30.0, fs=self._fs)
            sos = tf2sos(b, a)
            for ch in range(out.shape[0]):
                out[ch] = sosfiltfilt(sos, out[ch]).astype(np.float32)

        if 0.1 < low_cut_hz < nyq:
            sos = butter(4, low_cut_hz, btype="highpass", fs=self._fs, output="sos")
            for ch in range(out.shape[0]):
                out[ch] = sosfiltfilt(sos, out[ch]).astype(np.float32)

        if 0.1 < high_cut_hz < nyq:
            sos = butter(4, high_cut_hz, btype="lowpass", fs=self._fs, output="sos")
            for ch in range(out.shape[0]):
                out[ch] = sosfiltfilt(sos, out[ch]).astype(np.float32)

        # ── 2. Baseline correction (IIR high-pass) ────────────────────
        if enable_baseline:
            for ch in range(out.shape[0]):
                out[ch] = self._apply_iir_highpass(
                    out[ch], baseline_cutoff_hz
                )

        # ── 3. SEP artifact reduction (median + trigger blank) ────────
        if enable_sep:
            blanking_samples = max(1, round(sep_blank_ms * self._fs / 1000.0))
            for ch in range(out.shape[0]):
                out[ch] = self._reduce_sep(
                    out[ch],
                    median_window=5,
                    sep_trigger=sep_trigger,
                    blanking_samples=blanking_samples,
                )

        # ── 4. Cautery blanking (amplitude-hold) ─────────────────────
        if enable_cautery:
            hold_samples = max(1, round(cautery_hold_ms * self._fs / 1000.0))
            for ch in range(out.shape[0]):
                out[ch] = self._blank_cautery_hold(
                    out[ch],
                    threshold=cautery_threshold,
                    hold_samples=hold_samples,
                )

        return out

    # ── Algorithm 1: IIR High-Pass Baseline Correction ────────────────

    def _apply_iir_highpass(self, signal: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """First-order IIR high-pass filter for baseline drift removal.

        y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        alpha = 1 / (1 + 2π·cutoff / fs)

        Cheap, causal, stateless.  Default cutoff ~15 Hz removes DC and
        slow baseline wander so ZCR calculations are anchored to zero.
        """
        cutoff_hz = max(0.1, min(cutoff_hz, self._fs * 0.45))
        alpha = 1.0 / (1.0 + 2.0 * np.pi * cutoff_hz / self._fs)
        n = len(signal)
        out = np.empty(n, dtype=np.float32)
        if n == 0:
            return out
        out[0] = signal[0]
        for i in range(1, n):
            out[i] = alpha * (out[i - 1] + signal[i] - signal[i - 1])
        return out

    # ── Algorithm 2: SEP Artifact Reduction ───────────────────────────

    @staticmethod
    def _reduce_sep(
        signal: np.ndarray,
        median_window: int = 5,
        sep_trigger: bool = False,
        blanking_samples: int = 0,
    ) -> np.ndarray:
        """SEP artifact reduction via small-window median filter.

        Step A: Apply a tight median filter (3–5 samples) to flatten sharp
                ultra-fast spikes from somatosensory stimulation.
        Step B: If sep_trigger is True, force output[0:blanking_samples] = 0
                (useful when a hardware trigger marks each stimulus onset).
        """
        win = max(3, median_window | 1)  # ensure odd
        out = _median_filter1d(signal, size=win).astype(np.float32)
        if sep_trigger and blanking_samples > 0:
            out[:blanking_samples] = 0.0
        return out

    # ── Algorithm 3: Cautery Blanking (Amplitude-Hold) ────────────────

    @staticmethod
    def _blank_cautery_hold(
        signal: np.ndarray,
        threshold: float,
        hold_samples: int,
    ) -> np.ndarray:
        """Amplitude-based cautery blanking with hold.

        Whenever |signal[n]| > threshold the output is forced to 0 and
        a hold counter is (re)started.  The output stays at 0 for
        hold_samples after the last exceedance — mimicking a triggered
        relay used in clinical cautery blankers.
        """
        out = signal.copy()
        hold = 0
        for i in range(len(out)):
            if abs(out[i]) > threshold:
                hold = hold_samples
            if hold > 0:
                out[i] = 0.0
                hold -= 1
        return out


# ━━ CSV Data Handler ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DataHandler:
    """Manages CSV ingestion via pandas chunked reading for memory
    efficiency on multi-million-row files."""

    DEFAULT_CHUNKSIZE = 50_000

    def __init__(self):
        self.time: np.ndarray | None = None
        self.channels: np.ndarray | None = None   # (num_ch, N)
        self.raw_data_buffer: np.ndarray | None = None
        self.filtered_data_buffer: np.ndarray | None = None
        self.sample_rate: float = 0.0
        self.num_channels: int = 0
        self.metadata: dict = {}

    @staticmethod
    def read_csv_metadata(filepath: str) -> dict:
        """Parse custom metadata rows (0..7) and extract filter defaults."""
        meta = {
            "custom_format": False,
            "channel_names": [],
            "start_time": "",
            "period_s": None,
            "sample_rate": None,
            "low_cut_hz": 20.0,
            "high_cut_hz": 500.0,
            "notch_hz": 60.0,
            "data_start_row": 8,
        }
        def _norm(x) -> str:
            return str(x).replace("\ufeff", "").strip().lower()

        with open(filepath, "r", newline="", encoding="utf-8", errors="ignore") as f:
            rows = []
            reader = csv.reader(f)
            for _ in range(20):
                try:
                    rows.append(next(reader))
                except StopIteration:
                    break
        if not rows:
            return meta

        name_row_idx = -1
        period_row_idx = -1
        units_row_idx = -1
        for idx, row in enumerate(rows):
            if not row:
                continue
            key = _norm(row[0])
            if key == "name" and name_row_idx < 0:
                name_row_idx = idx
            if "period" in key and period_row_idx < 0:
                period_row_idx = idx
            if key in ("units", "unit") and units_row_idx < 0:
                units_row_idx = idx

        if name_row_idx >= 0:
            r0 = rows[name_row_idx]
            meta["custom_format"] = True
            meta["channel_names"] = [c.strip() for c in r0[1:] if str(c).strip()]
            if units_row_idx >= 0:
                meta["data_start_row"] = units_row_idx + 1
            else:
                # Fallback expected by this CSV profile: first 8 metadata rows.
                meta["data_start_row"] = 8

        r1 = rows[name_row_idx + 1] if (name_row_idx >= 0 and name_row_idx + 1 < len(rows)) else []
        if r1 and _norm(r1[0]) == "time" and len(r1) > 1:
            meta["start_time"] = str(r1[1]).strip()

        r6 = rows[period_row_idx] if period_row_idx >= 0 else []
        if r6 and len(r6) > 1:
            try:
                period = float(r6[1])
                meta["period_s"] = period
            except Exception:
                pass

        for row in rows:
            if not row:
                continue
            key = _norm(row[0])
            val = None
            for c in row[1:]:
                try:
                    val = float(c)
                    break
                except Exception:
                    continue
            if val is None:
                continue
            if "high" in key and "cut" in key:
                meta["high_cut_hz"] = float(val)
            elif "low" in key and "cut" in key:
                meta["low_cut_hz"] = float(val)
            elif "notch" in key:
                meta["notch_hz"] = float(val)

        # This project standardizes to 1280 Hz for this device profile.
        meta["sample_rate"] = 1280.0
        return meta

    @staticmethod
    def estimate_rows(filepath: str) -> int:
        """Fast byte-level newline count for progress estimation."""
        count = 0
        with open(filepath, "rb") as f:
            while block := f.read(1 << 20):
                count += block.count(b"\n")
        return max(count - 1, 0)

    @staticmethod
    def detect_columns(filepath: str) -> tuple[str, list[str]]:
        """Read the header and return (time_col, [ch_col, ...])."""
        meta = DataHandler.read_csv_metadata(filepath)
        if meta.get("custom_format", False):
            return "__generated_time__", list(meta.get("channel_names", []))
        header = pd.read_csv(filepath, nrows=0)
        cols = list(header.columns)

        time_col = None
        for candidate in ("Time", "time", "TIME", "t", "timestamp"):
            if candidate in cols:
                time_col = candidate
                break
        if time_col is None:
            time_col = cols[0]

        ch_cols = [c for c in cols if c != time_col]
        return time_col, ch_cols

    @staticmethod
    def iter_chunks(filepath: str, chunksize: int = DEFAULT_CHUNKSIZE):
        """Yield DataFrames of *chunksize* rows from a CSV."""
        yield from pd.read_csv(
            filepath,
            chunksize=chunksize,
            dtype=np.float32,
            engine="c",
        )

    @staticmethod
    def iter_signal_chunks(filepath: str, chunksize: int = DEFAULT_CHUNKSIZE):
        """Yield tuples: (time_arr, channel_arr_T, metadata)."""
        meta = DataHandler.read_csv_metadata(filepath)
        if meta.get("custom_format", False):
            ch_names = list(meta.get("channel_names", []))
            n_ch = len(ch_names)
            fs = float(meta.get("sample_rate", 1280.0))
            offset = 0
            usecols = list(range(1, n_ch + 1)) if n_ch > 0 else None
            for df in pd.read_csv(
                filepath,
                skiprows=int(meta.get("data_start_row", 8)),
                header=None,
                chunksize=chunksize,
                usecols=usecols,
                dtype=np.float32 if usecols is not None else None,
                engine="c",
                on_bad_lines="skip",
            ):
                if df.empty:
                    continue

                # First column is intentionally excluded (time/index not needed).
                sig = df.to_numpy(dtype=np.float32) if usecols is not None else (
                    df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                )
                if sig.size == 0 or sig.shape[1] == 0:
                    continue

                t = np.arange(offset, offset + sig.shape[0], dtype=np.float32) / fs
                offset += sig.shape[0]
                yield t, sig.T, meta
            return

        time_col, ch_cols = DataHandler.detect_columns(filepath)
        for chunk_df in DataHandler.iter_chunks(filepath, chunksize=chunksize):
            t_arr = chunk_df[time_col].to_numpy(dtype=np.float32)
            ch_arr = chunk_df[ch_cols].to_numpy(dtype=np.float32).T
            yield t_arr, ch_arr, meta

    def finalize(self, time: np.ndarray, channels: np.ndarray) -> None:
        """Store the fully-loaded arrays and derive sample rate."""
        self.time = time
        self.channels = channels
        self.filtered_data_buffer = channels
        self.raw_data_buffer = channels.copy()
        self.num_channels = channels.shape[0]
        if len(time) > 1:
            dt = float(time[1] - time[0])
            self.sample_rate = round(1.0 / dt) if dt > 0 else 0.0
