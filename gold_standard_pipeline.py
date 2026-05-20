"""
gold_standard_pipeline.py — Step-by-step gold standard (methodology.md).

Run stages 1→5 individually; each stage caches results so later steps do not
repeat heavy work. Stages map to methodology prompts 1–4 plus prep and UI export.

  1. Artifact prep     — cautery / artifact blanking (per channel)
  2. Energy gate       — bandpass + TKEO + MAD mask (methodology prompt 1)
  3. MUP detection     — adaptive TKEO peaks + guardrails (prompt 2)
  4. Train clustering  — ISI clusters + kinematic features (prompt 3)
  5. Classify + plot   — Romstöck labels + annotation regions (prompt 4 + UI)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from gold_standard_model import (
    DEFAULT_CLUSTER_GAP_S,
    apply_energy_gate,
    cap_annotation_list,
    classify_emg_event,
    detect_mup_timestamps,
    events_to_annotations,
    extract_train_features,
    filter_events_for_display,
    recommended_worker_count,
)

STAGE_LABELS: dict[int, str] = {
    1: "1 · Artifact prep",
    2: "2 · Energy gate",
    3: "3 · MUP detection",
    4: "4 · Train clustering",
    5: "5 · Classify & annotate",
}

STAGE_DESCRIPTIONS: dict[int, str] = {
    1: "Blank cautery/artifact excursions (fast).",
    2: "Bandpass + TKEO energy gate — heaviest filter step.",
    3: "Find MUP pulse timestamps with morphological guardrails.",
    4: "Cluster spikes into events; compute duration, rate, CV, Vpp.",
    5: "Romstöck A/B/C labels and draw regions (lightweight).",
}


@dataclass
class GoldStandardCache:
    """Cached per-channel pipeline state (independent muscles)."""

    sample_rate: float = 1280.0
    n_channels: int = 0
    n_samples: int = 0
    time_offset_s: float = 0.0
    stages_done: set[int] = field(default_factory=set)

    clean: np.ndarray | None = None
    gated: np.ndarray | None = None
    pulses_per_ch: list[np.ndarray] | None = None
    events_per_ch: list[list[dict]] | None = None
    annotations: list[dict] | None = None

    def reset(self) -> None:
        self.stages_done.clear()
        self.clean = None
        self.gated = None
        self.pulses_per_ch = None
        self.events_per_ch = None
        self.annotations = None

    def invalidate_from(self, stage: int) -> None:
        """Drop cached data for this stage and all later stages."""
        if stage <= 1:
            self.clean = None
            self.stages_done.discard(1)
        if stage <= 2:
            self.gated = None
            self.stages_done.discard(2)
        if stage <= 3:
            self.pulses_per_ch = None
            self.stages_done.discard(3)
        if stage <= 4:
            self.events_per_ch = None
            self.stages_done.discard(4)
        if stage <= 5:
            self.annotations = None
            self.stages_done.discard(5)

    def status_text(self) -> str:
        done = ", ".join(str(s) for s in sorted(self.stages_done)) or "none"
        return f"Gold cache: steps [{done}] | {self.n_channels} ch × {self.n_samples:,} samples"


def _parallel_channels(
    n_ch: int,
    worker: Callable[[int], None],
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    for ch in range(n_ch):
        worker(ch)
        if on_progress is not None:
            on_progress(ch + 1, n_ch)


def run_stage(
    stage: int,
    cache: GoldStandardCache,
    channels: np.ndarray,
    *,
    max_channels: int = 16,
    cluster_gap_s: float = DEFAULT_CLUSTER_GAP_S,
    annotation_pad_s: float = 0.02,
    artifact_threshold: float = 2000.0,
    on_progress: Callable[[int, str], None] | None = None,
) -> GoldStandardCache:
    """
    Run a single pipeline stage (1–5). Earlier outputs must exist unless
    stage is 1 (uses raw ``channels``).
    """
    if stage < 1 or stage > 5:
        raise ValueError("stage must be 1..5")

    if channels.ndim != 2:
        raise ValueError("channels must be (n_channels, n_samples)")

    n_ch = min(int(channels.shape[0]), int(max_channels))
    n_s = int(channels.shape[1])
    fs = float(max(cache.sample_rate, 1.0))
    cache.n_channels = n_ch
    cache.n_samples = n_s
    cache.invalidate_from(stage)

    def _emit(pct: int, msg: str) -> None:
        if on_progress is not None:
            on_progress(pct, msg)

    if stage == 1:
        from data_handler import detect_and_blank_artifacts

        _emit(0, STAGE_DESCRIPTIONS[1])
        clean = np.empty((n_ch, n_s), dtype=np.float32)
        for ch in range(n_ch):
            clean[ch] = detect_and_blank_artifacts(
                channels[ch].astype(np.float32, copy=False),
                threshold=artifact_threshold,
            )
            _emit(int((ch + 1) / max(n_ch, 1) * 100), f"Step 1 Ch{ch + 1}/{n_ch}")
        cache.clean = clean
        cache.stages_done.add(1)
        return cache

    if stage == 2:
        if cache.clean is None:
            raise RuntimeError("Run Step 1 (Artifact prep) first.")
        _emit(0, STAGE_DESCRIPTIONS[2])
        gated = np.empty((n_ch, n_s), dtype=np.float32)
        for ch in range(n_ch):
            gated[ch] = apply_energy_gate(cache.clean[ch], fs).astype(np.float32, copy=False)
            _emit(int((ch + 1) / max(n_ch, 1) * 100), f"Step 2 Ch{ch + 1}/{n_ch}")
        cache.gated = gated
        cache.stages_done.add(2)
        return cache

    if stage == 3:
        if cache.gated is None:
            raise RuntimeError("Run Step 2 (Energy gate) first.")
        _emit(0, STAGE_DESCRIPTIONS[3])
        pulses_per_ch: list[np.ndarray] = []
        for ch in range(n_ch):
            pulses = detect_mup_timestamps(cache.gated[ch], fs)
            pulses_per_ch.append(pulses)
            _emit(int((ch + 1) / max(n_ch, 1) * 100), f"Step 3 Ch{ch + 1}/{n_ch}: {pulses.size} pulses")
        cache.pulses_per_ch = pulses_per_ch
        cache.stages_done.add(3)
        return cache

    if stage == 4:
        if cache.clean is None or cache.pulses_per_ch is None:
            raise RuntimeError("Run Steps 1 and 3 first.")
        _emit(0, STAGE_DESCRIPTIONS[4])
        events_per_ch: list[list[dict]] = []
        for ch in range(n_ch):
            events = extract_train_features(
                cache.pulses_per_ch[ch],
                cache.clean[ch],
                fs,
                cluster_gap_s=cluster_gap_s,
            )
            for ev in events:
                ev["channel_idx"] = ch
            events_per_ch.append(events)
            _emit(int((ch + 1) / max(n_ch, 1) * 100), f"Step 4 Ch{ch + 1}/{n_ch}: {len(events)} clusters")
        cache.events_per_ch = events_per_ch
        cache.stages_done.add(4)
        return cache

    # Stage 5
    if cache.events_per_ch is None:
        raise RuntimeError("Run Step 4 (Train clustering) first.")
    _emit(0, STAGE_DESCRIPTIONS[5])
    all_anns: list[dict] = []
    for ch in range(n_ch):
        events = []
        for ev in cache.events_per_ch[ch]:
            labeled = dict(ev)
            labeled["label"] = classify_emg_event(labeled)
            labeled["channel_idx"] = ch
            events.append(labeled)
        events = filter_events_for_display(events)
        all_anns.extend(events_to_annotations(events, pad_s=annotation_pad_s))
        _emit(int((ch + 1) / max(n_ch, 1) * 100), f"Step 5 Ch{ch + 1}/{n_ch}")

    t_off = float(cache.time_offset_s)
    if t_off != 0.0:
        for ann in all_anns:
            ann["start_time"] = float(ann["start_time"]) + t_off
            ann["end_time"] = float(ann["end_time"]) + t_off

    cache.annotations = cap_annotation_list(all_anns)
    cache.events_per_ch = [
        [
            {**ev, "label": classify_emg_event(ev), "channel_idx": ch}
            for ev in cache.events_per_ch[ch]
        ]
        for ch in range(n_ch)
    ]
    cache.stages_done.add(5)
    _emit(100, f"Step 5 complete: {len(cache.annotations)} region(s)")
    return cache


def run_stages(
    start_stage: int,
    end_stage: int,
    cache: GoldStandardCache,
    channels: np.ndarray,
    **kwargs,
) -> GoldStandardCache:
    """Run inclusive range of stages (e.g. 1–5 for full pipeline)."""
    for s in range(start_stage, end_stage + 1):
        run_stage(s, cache, channels, **kwargs)
    return cache
