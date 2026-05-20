"""
annotation_store.py — Training dataset CSV with waveform + event annotations.

Format (IONM training v1):
  Metadata rows (schema, source, sample rate, channel names, …)
  Waveform block: Time + up to 16 channel columns
  Annotation block: one row per clinical event with exact start/end times
"""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "ionm_training_v1"
ANNOTATION_MARKER = "__ANNOTATIONS__"
WAVEFORM_MARKER = "__WAVEFORM__"


def _annotation_fieldnames() -> list[str]:
    return [
        "channel_idx",
        "channel",
        "start_time",
        "end_time",
        "label",
        "mu_id",
        "duration",
        "amplitude",
        "frequency",
        "n_spikes",
        "rhythmicity_cv",
        "tendency_slope_hz_s",
        "median_firing_rate",
        "source",
    ]


def export_training_csv(
    filepath: str,
    time_s: np.ndarray,
    channels: np.ndarray,
    annotations: list[dict],
    *,
    sample_rate_hz: float,
    channel_names: list[str] | None = None,
    linked_source_csv: str = "",
) -> None:
    """
    Write waveform + annotation events to a single training CSV.

    Annotations use spike-cluster boundaries (start_time/end_time) so event
    duration matches clinical train extent, not arbitrary UI padding.
    """
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    ch = np.asarray(channels, dtype=np.float32)
    if ch.ndim != 2:
        raise ValueError("channels must be (n_channels, n_samples)")
    if t.shape[0] != ch.shape[1]:
        n = min(t.shape[0], ch.shape[1])
        t = t[:n]
        ch = ch[:, :n]

    n_ch = min(int(ch.shape[0]), 16)
    ch = ch[:n_ch]
    names = list(channel_names or [])
    while len(names) < n_ch:
        names.append(f"Ch{len(names) + 1}")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["schema", SCHEMA_VERSION])
        writer.writerow(["linked_source_csv", os.path.basename(linked_source_csv)])
        writer.writerow(["linked_source_path", os.path.abspath(linked_source_csv)])
        writer.writerow(["sample_rate_hz", f"{float(sample_rate_hz):.6f}"])
        writer.writerow(["n_samples", str(t.shape[0])])
        writer.writerow(["n_channels", str(n_ch)])
        writer.writerow(["channel_names", *names])
        writer.writerow([WAVEFORM_MARKER])
        header = ["Time", *names[:n_ch]]
        writer.writerow(header)

        for i in range(t.shape[0]):
            row = [f"{t[i]:.9f}", *[f"{ch[c, i]:.9g}" for c in range(n_ch)]]
            writer.writerow(row)

        writer.writerow([ANNOTATION_MARKER])
        writer.writerow(_annotation_fieldnames())
        for ann in annotations:
            ch_idx = int(ann.get("channel_idx", 0))
            ch_idx = max(0, min(n_ch - 1, ch_idx))
            st = float(ann.get("start_time", 0.0))
            et = float(ann.get("end_time", st))
            dur = float(ann.get("duration", max(0.0, et - st)))
            writer.writerow(
                [
                    ch_idx,
                    str(ann.get("channel", f"Ch{ch_idx + 1}")),
                    f"{st:.9f}",
                    f"{et:.9f}",
                    str(ann.get("label", "MUP")),
                    str(ann.get("mu_id", "")),
                    f"{dur:.9f}",
                    f"{float(ann.get('amplitude', 0.0)):.9g}",
                    f"{float(ann.get('frequency', 0.0)):.9g}",
                    int(ann.get("n_spikes", 0)),
                    f"{float(ann.get('rhythmicity_cv', 0.0)):.9g}",
                    f"{float(ann.get('tendency_slope_hz_s', 0.0)):.9g}",
                    f"{float(ann.get('median_firing_rate', ann.get('frequency', 0.0))):.9g}",
                    str(ann.get("source", "human")),
                ]
            )


def load_training_csv(filepath: str) -> dict[str, Any]:
    """
    Load a training CSV produced by ``export_training_csv``.

    Returns dict with keys: time, channels, annotations, metadata.
    """
    meta: dict[str, Any] = {
        "schema_version": "",
        "linked_source_csv": "",
        "linked_source_path": "",
        "sample_rate_hz": 1280.0,
        "channel_names": [],
        "custom_format": True,
    }
    annotations: list[dict] = []
    waveform_rows: list[list[str]] = []
    waveform_header: list[str] = []
    section = "meta"
    ann_fieldnames = _annotation_fieldnames()

    with open(filepath, "r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key = str(row[0]).strip()

            if key == WAVEFORM_MARKER:
                section = "waveform_header"
                continue
            if key == ANNOTATION_MARKER:
                section = "annotation_header"
                continue

            if section == "meta":
                k = key.lower()
                if k == "schema":
                    meta["schema_version"] = str(row[1]).strip() if len(row) > 1 else ""
                elif k == "linked_source_csv":
                    meta["linked_source_csv"] = str(row[1]).strip() if len(row) > 1 else ""
                elif k == "linked_source_path":
                    meta["linked_source_path"] = str(row[1]).strip() if len(row) > 1 else ""
                elif k == "sample_rate_hz" and len(row) > 1:
                    try:
                        meta["sample_rate_hz"] = float(row[1])
                    except ValueError:
                        pass
                elif k == "channel_names":
                    meta["channel_names"] = [c.strip() for c in row[1:] if str(c).strip()]
                continue

            if section == "waveform_header":
                waveform_header = [c.strip() for c in row]
                section = "waveform"
                continue

            if section == "waveform":
                waveform_rows.append(row)
                continue

            if section == "annotation_header":
                ann_fieldnames = [c.strip() for c in row]
                section = "annotations"
                continue

            if section == "annotations":
                rec = {ann_fieldnames[i]: row[i] if i < len(row) else "" for i in range(len(ann_fieldnames))}
                try:
                    annotations.append(
                        {
                            "channel_idx": int(float(rec.get("channel_idx", 0))),
                            "channel": str(rec.get("channel", "")),
                            "start_time": float(rec.get("start_time", 0.0)),
                            "end_time": float(rec.get("end_time", 0.0)),
                            "label": str(rec.get("label", "MUP")),
                            "mu_id": str(rec.get("mu_id", "")),
                            "duration": float(rec.get("duration", 0.0)),
                            "amplitude": float(rec.get("amplitude", 0.0)),
                            "frequency": float(rec.get("frequency", 0.0)),
                            "n_spikes": int(float(rec.get("n_spikes", 0))),
                            "rhythmicity_cv": float(rec.get("rhythmicity_cv", 0.0)),
                            "tendency_slope_hz_s": float(rec.get("tendency_slope_hz_s", 0.0)),
                            "source": str(rec.get("source", "file")),
                        }
                    )
                except (TypeError, ValueError):
                    continue

    if not waveform_rows or not waveform_header:
        raise ValueError(f"No waveform block found in {filepath}")

    df = pd.DataFrame(waveform_rows, columns=waveform_header[: len(waveform_rows[0])])
    time_col = waveform_header[0]
    for c in ("Time", "time", "TIME"):
        if c in df.columns:
            time_col = c
            break
    ch_cols = [c for c in df.columns if c != time_col]
    if not ch_cols:
        raise ValueError("Training CSV has no channel columns.")

    time_s = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ch_mat = df[ch_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32).T

    if meta.get("sample_rate_hz", 0) <= 0 and time_s.size > 1:
        dt = float(time_s[1] - time_s[0])
        if dt > 0:
            meta["sample_rate_hz"] = 1.0 / dt

    return {
        "time": time_s,
        "channels": ch_mat,
        "annotations": annotations,
        "metadata": meta,
    }


def is_training_csv(filepath: str) -> bool:
    """Return True if file begins with IONM training schema marker."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(6):
                line = f.readline()
                if not line:
                    break
                if SCHEMA_VERSION in line or "ionm_training" in line.lower():
                    return True
    except OSError:
        return False
    return False
