"""
gold_standard_model.py — Deterministic Stage-1 pipeline from methodology.md.

Per-channel pipeline (16 muscles = 16 independent MUP streams):
  1. Energy-linked noise gate (bandpass + TKEO + MAD mask + dilation)
  2. Adaptive TKEO peak detection with morphological guardrails
  3. ISI train clustering and kinematic feature extraction
  4. Romstöck clinical event classification
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from scipy import ndimage, stats
from scipy.signal import butter, find_peaks, iirnotch, sosfiltfilt, tf2sos

# Clinical ISI gap between separate neurotonic events (seconds).
DEFAULT_CLUSTER_GAP_S = 0.5
# Guardrails so noisy recordings cannot create millions of UI regions / peaks.
MAX_PEAK_CANDIDATES = 12_000
DEFAULT_MAX_EVENTS_PER_CHANNEL = 30
DEFAULT_MAX_UI_ANNOTATIONS = 320


def _zero_crossing_count(signal: np.ndarray) -> int:
    """Count sign changes in a micro-window (morphological guardrail)."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return 0
    count = 0
    for i in range(1, x.size):
        if (x[i - 1] >= 0.0 and x[i] < 0.0) or (x[i - 1] < 0.0 and x[i] >= 0.0):
            count += 1
    return count


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (robust noise scale)."""
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def apply_energy_gate(
    signal: np.ndarray,
    fs: float,
    *,
    low_hz: float = 30.0,
    high_hz: float = 500.0,
    notch_hz: float = 60.0,
    mad_k: float = 3.0,
    dilation_ms: float = 2.0,
) -> np.ndarray:
    """
    Filter EMG and apply an energy-linked noise gate.

    Returns gated signal with baseline fuzz zeroed while preserving MUP crossings.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < 8:
        return x.astype(np.float64, copy=False)

    fs = float(max(fs, 1.0))
    nyq = fs * 0.5
    out = x.copy()

    # Single bandpass covers the high-pass + band-limit requirement (faster than two filtfilts).
    if low_hz < high_hz < nyq:
        sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
        out = sosfiltfilt(sos, out)
    elif 0.1 < low_hz < nyq:
        sos = butter(4, low_hz, btype="highpass", fs=fs, output="sos")
        out = sosfiltfilt(sos, out)
    if 0.5 < notch_hz < nyq:
        b, a = iirnotch(notch_hz, 30.0, fs=fs)
        sos = tf2sos(b, a)
        out = sosfiltfilt(sos, out)

    # Teager-Kaiser Energy Operator
    tkeo = np.empty_like(out)
    tkeo[0] = out[0] * out[0]
    tkeo[-1] = out[-1] * out[-1]
    if out.size > 2:
        tkeo[1:-1] = out[1:-1] ** 2 - out[:-2] * out[2:]

    noise = _mad(tkeo)
    threshold = mad_k * noise
    mask = tkeo > threshold

    dil_samples = max(1, int(round(fs * dilation_ms * 1e-3)))
    mask = ndimage.binary_dilation(mask, structure=np.ones(dil_samples * 2 + 1))

    return (out * mask.astype(np.float64)).astype(np.float64, copy=False)


def _adaptive_tkeo_threshold(tkeo: np.ndarray, win: int, sigma_k: float) -> np.ndarray:
    """Vectorized rolling mean + std threshold (O(n), not O(n·win))."""
    win = max(8, int(win))
    tkeo = np.asarray(tkeo, dtype=np.float64).reshape(-1)
    mu = ndimage.uniform_filter1d(tkeo, size=win, mode="nearest")
    mean_sq = ndimage.uniform_filter1d(tkeo * tkeo, size=win, mode="nearest")
    sigma = np.sqrt(np.maximum(mean_sq - mu * mu, 0.0))
    return mu + float(sigma_k) * sigma


def detect_mup_timestamps(
    gated_signal: np.ndarray,
    fs: float,
    *,
    window_s: float = 1.0,
    sigma_k: float = 5.0,
    micro_window_ms: float = 10.0,
    min_zcr: int = 1,
    max_zcr: int = 4,
    min_duration_ms: float = 2.0,
    max_duration_ms: float = 15.0,
    refractory_ms: float = 2.0,
) -> np.ndarray:
    """
    Detect MUP trigger times (seconds) using adaptive TKEO thresholding and guardrails.
    """
    x = np.asarray(gated_signal, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 16:
        return np.empty((0,), dtype=np.float64)

    fs = float(max(fs, 1.0))
    tkeo = np.empty(n, dtype=np.float64)
    tkeo[0] = x[0] * x[0]
    tkeo[-1] = x[-1] * x[-1]
    if n > 2:
        tkeo[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]

    win = max(8, int(round(window_s * fs)))
    threshold = _adaptive_tkeo_threshold(tkeo, win, sigma_k)

    min_distance = max(1, int(round(fs * refractory_ms * 1e-3)))
    peak_idx, _props = find_peaks(tkeo, height=threshold, distance=min_distance)
    if peak_idx.size == 0:
        return np.empty((0,), dtype=np.float64)
    if peak_idx.size > MAX_PEAK_CANDIDATES:
        strengths = tkeo[peak_idx]
        keep = np.argpartition(strengths, -MAX_PEAK_CANDIDATES)[-MAX_PEAK_CANDIDATES:]
        peak_idx = np.sort(peak_idx[keep])

    half_micro = max(1, int(round(fs * micro_window_ms * 0.5 * 1e-3)))
    min_dur = int(round(fs * min_duration_ms * 1e-3))
    max_dur = int(round(fs * max_duration_ms * 1e-3))

    kept: list[int] = []
    for idx in peak_idx:
        lo = max(0, int(idx) - half_micro)
        hi = min(n, int(idx) + half_micro + 1)
        micro = x[lo:hi]
        if micro.size < 4:
            continue

        zcr_count = _zero_crossing_count(micro)
        if zcr_count < min_zcr or zcr_count > max_zcr:
            continue

        nz = np.flatnonzero(np.abs(micro) > 1e-9)
        if nz.size == 0:
            continue
        wave_dur = int(nz[-1] - nz[0] + 1)
        if wave_dur < min_dur or wave_dur > max_dur:
            continue
        kept.append(int(idx))

    if not kept:
        return np.empty((0,), dtype=np.float64)

    times = np.asarray(kept, dtype=np.float64) / fs
    return np.sort(times)


def extract_train_features(
    timestamps: np.ndarray,
    raw_signal: np.ndarray,
    fs: float,
    *,
    cluster_gap_s: float = DEFAULT_CLUSTER_GAP_S,
    vpp_window_ms: float = 5.0,
) -> list[dict]:
    """
    Cluster spike trains and extract kinematic features per clinical event.
    """
    ts = np.sort(np.asarray(timestamps, dtype=np.float64).reshape(-1))
    raw = np.asarray(raw_signal, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        return []

    fs = float(max(fs, 1.0))
    half_vpp = max(1, int(round(fs * vpp_window_ms * 0.5 * 1e-3)))

    clusters: list[list[float]] = []
    current: list[float] = [float(ts[0])]
    for t in ts[1:]:
        if (float(t) - current[-1]) > cluster_gap_s:
            clusters.append(current)
            current = [float(t)]
        else:
            current.append(float(t))
    clusters.append(current)

    events: list[dict] = []
    for cluster in clusters:
        if not cluster:
            continue
        t_first = float(cluster[0])
        t_last = float(cluster[-1])
        duration = t_last - t_first

        if len(cluster) >= 2:
            isi = np.diff(np.asarray(cluster, dtype=np.float64))
            isi = isi[isi > 1e-9]
            if isi.size > 0:
                inst_freq = 1.0 / isi
                mean_freq = float(np.mean(inst_freq))
                median_freq = float(np.median(inst_freq))
                mean_isi = float(np.mean(isi))
                cv = float(np.std(isi) / mean_isi) if mean_isi > 1e-9 else 0.0
                rel_t = np.cumsum(isi)
                if inst_freq.size >= 2:
                    slope_res = stats.linregress(rel_t, inst_freq)
                    tendency_slope = float(slope_res.slope)
                else:
                    tendency_slope = 0.0
            else:
                mean_freq = median_freq = cv = tendency_slope = 0.0
        else:
            mean_freq = median_freq = cv = tendency_slope = 0.0

        vpp_vals: list[float] = []
        for t in cluster:
            idx = int(round(t * fs))
            lo = max(0, idx - half_vpp)
            hi = min(raw.size, idx + half_vpp + 1)
            if hi - lo >= 2:
                seg = raw[lo:hi]
                vpp_vals.append(float(np.max(seg) - np.min(seg)))

        events.append(
            {
                "t_first": t_first,
                "t_last": t_last,
                "start_time": t_first,
                "end_time": max(t_last, t_first + 1.0 / fs),
                "duration": float(duration),
                "n_spikes": int(len(cluster)),
                "mean_firing_rate": mean_freq,
                "median_firing_rate": median_freq,
                "rhythmicity_cv": cv,
                "tendency_slope_hz_s": tendency_slope,
                "median_vpp": float(np.median(vpp_vals)) if vpp_vals else 0.0,
            }
        )
    return events


def classify_emg_event(features_dict: dict) -> str:
    """
    Romstöck decision tree for IONM clinical event labels.
    """
    duration = float(features_dict.get("duration", 0.0))
    mean_fr = float(features_dict.get("mean_firing_rate", 0.0))
    cv = float(features_dict.get("rhythmicity_cv", 0.0))
    n_spikes = int(features_dict.get("n_spikes", 0))

    if duration < 1.0:
        if n_spikes <= 1:
            return "Spike"
        return "Burst"

    if duration >= 1.0 and mean_fr >= 60.0 and cv <= 0.15:
        return "A-Train (High Risk)"

    if duration >= 1.0 and cv > 0.45:
        return "C-Train (Polyrhythmic)"

    if duration >= 1.0:
        return "B-Train"

    return "Burst"


def analyze_channel_events(
    raw_signal: np.ndarray,
    fs: float,
    *,
    channel_idx: int = 0,
    cluster_gap_s: float = DEFAULT_CLUSTER_GAP_S,
) -> tuple[np.ndarray, list[dict]]:
    """
    Full per-channel gold-standard analysis.

    Returns
    -------
    pulse_timestamps_s : sorted MUP times (seconds)
    events : list of feature dicts with clinical labels and exact event bounds
    """
    raw = np.asarray(raw_signal, dtype=np.float64).reshape(-1)
    if raw.size < 16:
        return np.empty((0,), dtype=np.float64), []

    gated = apply_energy_gate(raw, fs)
    pulses = detect_mup_timestamps(gated, fs)
    events = extract_train_features(
        pulses,
        raw,
        fs,
        cluster_gap_s=cluster_gap_s,
    )
    for ev in events:
        ev["label"] = classify_emg_event(ev)
        ev["channel_idx"] = int(channel_idx)
    return pulses, events


def _analyze_channel_packed(
    ch_idx: int,
    signal: np.ndarray,
    fs: float,
    pad_s: float,
    cluster_gap_s: float,
) -> tuple[int, np.ndarray, list[dict], list[dict]]:
    """Picklable worker entry for multi-core processing (one muscle channel)."""
    pulses, events = analyze_channel_events(
        signal,
        fs,
        channel_idx=ch_idx,
        cluster_gap_s=cluster_gap_s,
    )
    events = filter_events_for_display(events)
    anns = events_to_annotations(events, pad_s=pad_s)
    return ch_idx, pulses, events, anns


def recommended_worker_count(n_channels: int) -> int:
    """Use most CPU cores, leave one for the GUI thread."""
    cores = int(os.cpu_count() or 4)
    return max(1, min(int(n_channels), max(1, cores - 1)))


def run_gold_standard_parallel(
    channels: np.ndarray,
    sample_rate: float,
    *,
    max_channels: int = 16,
    annotation_pad_s: float = 0.02,
    cluster_gap_s: float = DEFAULT_CLUSTER_GAP_S,
    n_workers: int | None = None,
    on_channel_done=None,
    prefer_processes: bool = True,
) -> tuple[list[np.ndarray], list[list[dict]], list[dict]]:
    """
    Analyze channels in parallel using available CPU cores.

    Parameters
    ----------
    on_channel_done : optional callback(ch_idx, n_channels, pulses, n_events, n_anns)
    """
    if channels.ndim != 2:
        raise ValueError("channels must be (n_channels, n_samples)")

    n_ch = min(int(channels.shape[0]), int(max_channels))
    fs = float(max(sample_rate, 1.0))
    n_workers = int(n_workers) if n_workers else recommended_worker_count(n_ch)
    n_workers = max(1, min(n_workers, n_ch))

    pulses_per_ch: list[np.ndarray] = [np.empty((0,), dtype=np.float64) for _ in range(n_ch)]
    events_per_ch: list[list[dict]] = [[] for _ in range(n_ch)]
    all_annotations: list[dict] = []

    tasks = [
        (
            ch,
            np.ascontiguousarray(channels[ch], dtype=np.float32),
            fs,
            annotation_pad_s,
            cluster_gap_s,
        )
        for ch in range(n_ch)
    ]

    def _run_pool(executor_cls, **kwargs):
        with executor_cls(max_workers=n_workers, **kwargs) as pool:
            futures = [pool.submit(_analyze_channel_packed, *task) for task in tasks]
            for fut in as_completed(futures):
                ch_idx, pulses, events, anns = fut.result()
                pulses_per_ch[ch_idx] = pulses
                events_per_ch[ch_idx] = events
                all_annotations.extend(anns)
                if on_channel_done is not None:
                    on_channel_done(ch_idx, n_ch, pulses, len(events), len(anns))

    n_samples = int(channels.shape[1])
    # Process pool helps long recordings; threads are faster for typical 1–2 min clips.
    use_processes = bool(prefer_processes) and n_workers > 1 and n_samples >= 250_000

    prev_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        if use_processes:
            try:
                _run_pool(ProcessPoolExecutor)
            except Exception:
                _run_pool(ThreadPoolExecutor)
        elif n_workers > 1:
            _run_pool(ThreadPoolExecutor)
        else:
            for task in tasks:
                ch_idx, pulses, events, anns = _analyze_channel_packed(*task)
                pulses_per_ch[ch_idx] = pulses
                events_per_ch[ch_idx] = events
                all_annotations.extend(anns)
                if on_channel_done is not None:
                    on_channel_done(ch_idx, n_ch, pulses, len(events), len(anns))
    finally:
        if prev_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev_omp

    return pulses_per_ch, events_per_ch, cap_annotation_list(all_annotations)


def filter_events_for_display(
    events: list[dict],
    *,
    max_per_channel: int = DEFAULT_MAX_EVENTS_PER_CHANNEL,
    min_duration_s: float = 0.08,
) -> list[dict]:
    """
    Keep the most clinically relevant events so the GUI is not flooded (prevents crashes).
    """
    if not events or max_per_channel <= 0:
        return []

    scored = [
        (dur * (n_sp if n_sp > 1 else 1) * (3.0 if is_train else 1.0), ev)
        for ev in events
        for dur in (float(ev.get("duration", 0.0)),)
        for n_sp in (int(ev.get("n_spikes", 0)),)
        for is_train in ("Train" in str(ev.get("label", "")) or dur >= 1.0,)
        if not (dur < min_duration_s and n_sp < 2 and not is_train)
    ]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [ev for _, ev in scored[:max_per_channel]]


def cap_annotation_list(
    annotations: list[dict],
    *,
    max_total: int = DEFAULT_MAX_UI_ANNOTATIONS,
) -> list[dict]:
    """Global cap on regions sent to the UI."""
    if len(annotations) <= max_total:
        return annotations
    ranked = sorted(
        annotations,
        key=lambda a: float(a.get("duration", 0.0)) * max(int(a.get("n_spikes", 1)), 1),
        reverse=True,
    )
    return ranked[:max_total]


def events_to_annotations(
    events: list[dict],
    *,
    pad_s: float = 0.02,
    source: str = "gold_standard",
) -> list[dict]:
    """Convert clustered events to UI/training annotation dicts with exact duration."""
    out: list[dict] = []
    for ev in events:
        label = str(ev.get("label", "Burst"))
        if label in ("None", "Artifact"):
            continue
        t0 = float(ev.get("t_first", ev.get("start_time", 0.0))) - pad_s
        t1 = float(ev.get("t_last", ev.get("end_time", t0))) + pad_s
        if t1 <= t0:
            t1 = t0 + 1e-4
        out.append(
            {
                "channel_idx": int(ev.get("channel_idx", 0)),
                "start_time": max(0.0, t0),
                "end_time": t1,
                "label": label,
                "duration": float(t1 - t0),
                "amplitude": float(ev.get("median_vpp", 0.0)),
                "frequency": float(ev.get("mean_firing_rate", 0.0)),
                "n_spikes": int(ev.get("n_spikes", 0)),
                "rhythmicity_cv": float(ev.get("rhythmicity_cv", 0.0)),
                "tendency_slope_hz_s": float(ev.get("tendency_slope_hz_s", 0.0)),
                "source": source,
            }
        )
    return out


def run_gold_standard_multichannel(
    channels: np.ndarray,
    time_s: np.ndarray,
    sample_rate: float,
    *,
    max_channels: int = 16,
    cluster_gap_s: float = DEFAULT_CLUSTER_GAP_S,
    annotation_pad_s: float = 0.02,
) -> tuple[list[np.ndarray], list[dict], list[dict]]:
    """
    Analyze up to ``max_channels`` EMG rows independently.

    Returns
    -------
    pulses_per_ch : list of pulse timestamp arrays
    events_per_ch : list of event feature lists per channel
    annotations : flat list of annotation dicts for all channels
    """
    if channels.ndim != 2:
        raise ValueError("channels must be shaped (n_channels, n_samples)")

    n_ch = min(int(channels.shape[0]), int(max_channels))
    n_s = channels.shape[1]
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if t.shape[0] != n_s:
        n = min(t.shape[0], n_s)
        t = t[:n]
        channels = channels[:, :n]

    fs = float(max(sample_rate, 1.0))
    if fs <= 0 and t.size > 1:
        fs = 1.0 / max(float(t[1] - t[0]), 1e-9)

    pulses_per_ch: list[np.ndarray] = []
    events_per_ch: list[dict] = []
    all_annotations: list[dict] = []

    for ch in range(n_ch):
        pulses, events = analyze_channel_events(
            channels[ch],
            fs,
            channel_idx=ch,
            cluster_gap_s=cluster_gap_s,
        )
        pulses_per_ch.append(pulses)
        events_per_ch.append(events)
        all_annotations.extend(
            events_to_annotations(events, pad_s=annotation_pad_s)
        )

    return pulses_per_ch, events_per_ch, all_annotations


# ── Backward-compatible aliases (legacy imports) ─────────────────────────

def isolate_mup_pulse_timestamps(
    channels: np.ndarray,
    time_s: np.ndarray,
    *,
    sample_rate: float,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Legacy API: pulse times for channel 0 only."""
    if channels.ndim != 2 or channels.shape[1] < 8:
        return np.empty((0,), dtype=np.float64), {"components": []}
    pulses, events = analyze_channel_events(
        channels[0],
        float(max(sample_rate, 1.0)),
        channel_idx=0,
    )
    return pulses, {
        "n_peaks": int(pulses.size),
        "n_events": len(events),
        "method": "tkeo_energy_gate",
    }


def isolate_mup_pulse_timestamps_per_channel(
    channels: np.ndarray,
    time_s: np.ndarray,
    *,
    sample_rate: float,
    **kwargs,
) -> tuple[list[np.ndarray], list[dict]]:
    """Legacy API: per-channel pulse timestamps."""
    pulses_per_ch, _events_per_ch, _anns = run_gold_standard_multichannel(
        channels,
        time_s,
        sample_rate,
        max_channels=channels.shape[0],
    )
    diags = [
        {"channel_idx": i, "n_peaks": int(p.size), "method": "tkeo_energy_gate"}
        for i, p in enumerate(pulses_per_ch)
    ]
    return pulses_per_ch, diags


if __name__ == "__main__":
    # PROMPT 1 smoke test — HF burst should survive gate; quiet baseline should not.
    fs = 10_000.0
    n = 4000
    t = np.arange(n) / fs
    sig = 0.01 * np.random.default_rng(0).standard_normal(n)
    burst = 0.8 * np.sin(2 * np.pi * 180.0 * t[1500:1700])
    sig[1500:1700] += burst
    gated = apply_energy_gate(sig, fs)
    assert float(np.max(np.abs(gated[:1200]))) < 0.35
    assert float(np.max(np.abs(gated[1500:1700]))) > 0.2

    # PROMPT 4 unit tests
    assert classify_emg_event({"duration": 0.2, "n_spikes": 1}) == "Spike"
    assert classify_emg_event({"duration": 0.4, "n_spikes": 4, "mean_firing_rate": 80}) == "Burst"
    assert classify_emg_event(
        {"duration": 1.5, "mean_firing_rate": 65, "rhythmicity_cv": 0.1, "n_spikes": 80}
    ) == "A-Train (High Risk)"
    assert classify_emg_event(
        {"duration": 1.2, "mean_firing_rate": 20, "rhythmicity_cv": 0.6, "n_spikes": 30}
    ) == "C-Train (Polyrhythmic)"
    assert classify_emg_event(
        {"duration": 1.2, "mean_firing_rate": 20, "rhythmicity_cv": 0.3, "n_spikes": 30}
    ) == "B-Train"
    print("gold_standard_model: all self-tests passed")
