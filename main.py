"""
main.py — Application entry point for the IONM EMG AI Assistant.

Houses the AppController that wires UI signals to the DataLoadingThread
and manages gain-scaled plot updates.  Pass --gen-test-csv to create a
synthetic CSV for development testing.
"""

import sys
import os
import json
import shutil
import numpy as np
from time import perf_counter
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

_TORCH_DLL_HANDLES: list[object] = []
_EARLY_TORCH_PRELOAD_ERROR: str | None = None


def _prepare_windows_torch_dll_paths() -> None:
    """Ensure Windows can resolve torch native DLL dependencies."""
    if os.name != "nt":
        return
    py_root = os.path.dirname(sys.executable)
    candidates = (
        os.path.join(py_root, "DLLs"),
        os.path.join(py_root, "Library", "bin"),
        os.path.join(py_root, "Lib", "site-packages", "torch", "lib"),
    )
    for folder in candidates:
        if not os.path.isdir(folder):
            continue
        try:
            handle = os.add_dll_directory(folder)
            _TORCH_DLL_HANDLES.append(handle)
        except Exception:
            # Keep startup resilient if directory registration fails.
            continue


def _preload_torch_runtime_early() -> None:
    """Preload torch before NumPy/SciPy/Numba-heavy modules are used."""
    global _EARLY_TORCH_PRELOAD_ERROR
    if os.name != "nt":
        return
    # Avoid OpenMP duplicate-runtime aborts on some Windows stacks.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        _prepare_windows_torch_dll_paths()
        import torch  # noqa: F401
        _EARLY_TORCH_PRELOAD_ERROR = None
    except Exception as exc:
        _EARLY_TORCH_PRELOAD_ERROR = str(exc)


_preload_torch_runtime_early()

from ui_mainwindow import MainWindow
from worker_threads import (
    DataLoadingThread,
    RefilterThread,
    AITrainingThreadV2,
    NeurotonicInferenceThread,
    GoldStandardStageThread,
    TrainingDatasetLoadThread,
)
from gold_standard_pipeline import GoldStandardCache, STAGE_DESCRIPTIONS, STAGE_LABELS
from annotation_store import export_training_csv, is_training_csv
from data_handler import (
    calc_envelope_decrescendo,
    calc_peak_to_peak,
    calc_rms,
    calc_symmetry_index,
    calc_zcr,
    detect_micro_bursts,
)

PREVIEW_EVERY_N_CHUNKS = 5


class AppController:
    """Bridges the MainWindow UI with background data-loading threads."""

    def __init__(self, window: MainWindow):
        self.window = window

        self._time: np.ndarray | None = None
        self._raw_channels: np.ndarray | None = None
        self._channels: np.ndarray | None = None
        self._num_channels: int = 0
        self._sample_rate: float = 0.0
        self._csv_meta: dict = {}
        self._current_csv_path: str | None = None
        self._features: dict | None = None
        self._loader: DataLoadingThread | None = None
        self._refilter_thread: RefilterThread | None = None
        self._trainer: AITrainingThreadV2 | None = None
        self._neuro_thread: NeurotonicInferenceThread | None = None
        self._gold_thread: GoldStandardStageThread | None = None
        self._gold_cache = GoldStandardCache()
        self._training_load_thread: TrainingDatasetLoadThread | None = None
        self._infer_started_at: float | None = None
        self._torch_runtime_checked = False
        self._torch_runtime_ok = False
        self._torch_runtime_error: str | None = None
        self._dll_dir_handles: list[object] = []

        self._time_chunks: list[np.ndarray] = []
        self._ch_chunks: list[np.ndarray] = []
        self._chunks_received: int = 0

        window.btn_load_csv.clicked.connect(self._on_load_csv)
        window.btn_save.clicked.connect(self._on_save_annotations)
        window.btn_load_annotations.clicked.connect(self._on_load_annotations)
        window.btn_export_training_csv.clicked.connect(self._on_export_training_csv)
        window.btn_load_training_csv.clicked.connect(self._on_load_training_csv)
        window.btn_train.clicked.connect(self._on_start_training)
        window.btn_batch_train_folder.clicked.connect(self._on_start_batch_training)
        window.btn_auto_annotate_test.clicked.connect(self._on_start_auto_annotate_test)
        window.btn_gold_standard.clicked.connect(self._on_start_gold_standard_all)
        window.btn_gold_step1.clicked.connect(lambda: self._on_start_gold_stage(1))
        window.btn_gold_step2.clicked.connect(lambda: self._on_start_gold_stage(2))
        window.btn_gold_step3.clicked.connect(lambda: self._on_start_gold_stage(3))
        window.btn_gold_step4.clicked.connect(lambda: self._on_start_gold_stage(4))
        window.btn_gold_step5.clicked.connect(lambda: self._on_start_gold_stage(5))
        window.btn_inference.clicked.connect(self._on_start_auto_annotate_test)
        window.btn_create_backup.clicked.connect(self._create_model_backup)
        window.btn_restore_backup.clicked.connect(self._restore_model_backup)
        window.gain_slider.valueChanged.connect(self._on_gain_changed)
        window.annotation_selected.connect(self._on_annotation_selected)
        window.filter_settings_changed.connect(self._on_filter_settings_changed)
        window.trace_overlap_changed.connect(self._refresh_plots)
        window.auto_fit_requested.connect(self._on_auto_fit_y_range)
        self.window.set_gain_scale(self.window.gain_slider.value())
        self._refresh_backup_status_on_startup()
        # Windows safeguard: preload torch on GUI/main thread once.
        self._ensure_torch_runtime(show_dialog=False)

    def _prepare_torch_dll_path(self) -> None:
        """Add torch/lib to DLL search path (Windows only)."""
        _prepare_windows_torch_dll_paths()
        self._dll_dir_handles.extend(_TORCH_DLL_HANDLES)

    @staticmethod
    def _format_torch_runtime_error(exc: Exception) -> str:
        base = (
            "Failed to initialize PyTorch runtime.\n\n"
            f"Python: {sys.executable}\n"
            f"Error: {exc}"
        )
        text = str(exc)
        if ("WinError 1114" in text) or ("c10.dll" in text):
            base += (
                "\n\nWindows fix steps:\n"
                "1) Close the app.\n"
                "2) Run: python -m pip install --upgrade pip setuptools wheel msvc-runtime\n"
                "3) Reinstall CPU torch build:\n"
                "   python -m pip uninstall -y torch torchvision torchaudio\n"
                "   python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio\n"
                "4) Re-open the app and retry training/auto-annotate."
            )
        return base

    def _ensure_torch_runtime(self, show_dialog: bool = True) -> bool:
        """Initialize torch runtime once and cache status."""
        if self._torch_runtime_checked:
            if (not self._torch_runtime_ok) and show_dialog and self._torch_runtime_error:
                QMessageBox.critical(
                    self.window,
                    "PyTorch Runtime Error",
                    self._torch_runtime_error,
                )
            return self._torch_runtime_ok

        self._torch_runtime_checked = True
        try:
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            self._prepare_torch_dll_path()
            import torch  # noqa: F401
            self._torch_runtime_ok = True
            self._torch_runtime_error = None
            return True
        except Exception as exc:
            self._torch_runtime_ok = False
            prior_hint = f"\n\nStartup preload error: {_EARLY_TORCH_PRELOAD_ERROR}" if _EARLY_TORCH_PRELOAD_ERROR else ""
            self._torch_runtime_error = self._format_torch_runtime_error(exc) + prior_hint
            if show_dialog:
                QMessageBox.critical(
                    self.window,
                    "PyTorch Runtime Error",
                    self._torch_runtime_error,
                )
            return False

    # ── Load CSV flow ─────────────────────────────────────────────────

    def _on_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Open EMG CSV", "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        if is_training_csv(path):
            self._on_load_training_csv(path)
            return

        self._current_csv_path = path
        self._reset_accumulator()
        self._gold_cache.reset()
        self.window.set_gold_cache_status(self._gold_cache.status_text())
        self.window.clear_annotations()
        self.window.clear_all_curves()
        self.window.btn_load_csv.setEnabled(False)
        self.window.show_progress(0)
        self.window.set_status(f"Opening {os.path.basename(path)}…")

        self._loader = DataLoadingThread(path)
        self._loader.progress_updated.connect(self._on_progress)
        self._loader.chunk_ready.connect(self._on_chunk)
        self._loader.metadata_loaded.connect(self._on_metadata_loaded)
        self._loader.loading_finished.connect(self._on_finished)
        self._loader.features_extracted.connect(self._on_features)
        self._loader.loading_error.connect(self._on_error)
        self._loader.start()

    def _reset_accumulator(self):
        self._time_chunks.clear()
        self._ch_chunks.clear()
        self._chunks_received = 0
        self._time = None
        self._raw_channels = None
        self._channels = None
        self._sample_rate = 0.0
        self._csv_meta = {}
        self._features = None

    # ── Thread signal handlers ────────────────────────────────────────

    def _on_progress(self, pct: int, msg: str):
        self.window.show_progress(pct)
        self.window.set_status(msg)

    def _on_chunk(self, t_arr: np.ndarray, ch_arr: np.ndarray):
        self._time_chunks.append(t_arr)
        self._ch_chunks.append(ch_arr)
        self._chunks_received += 1

        if self._chunks_received % PREVIEW_EVERY_N_CHUNKS == 0:
            self._preview_plots()

    def _on_metadata_loaded(self, meta: dict):
        self._csv_meta = meta or {}
        self.window.apply_filter_metadata(self._csv_meta)
        ch_names = list(self._csv_meta.get("channel_names", []))
        if ch_names:
            self.window.set_channel_names(ch_names)

    def _on_finished(
        self,
        full_time: np.ndarray,
        raw_channels: np.ndarray,
        filtered_channels: np.ndarray,
        meta: dict,
    ):
        self._time = full_time
        self._raw_channels = raw_channels
        self._channels = filtered_channels
        self._num_channels = filtered_channels.shape[0]
        self._csv_meta = meta or {}

        self._time_chunks.clear()
        self._ch_chunks.clear()

        self.window.set_active_channels(self._num_channels)
        if len(full_time) > 0:
            self.window.set_time_extent(0.0, float(full_time[-1]))
        self._refresh_plots()
        self.window.show_progress(100)
        self.window.btn_load_csv.setEnabled(True)

        sr = 0.0
        if len(full_time) > 1:
            sr = 1.0 / float(full_time[1] - full_time[0])
        if self._csv_meta.get("sample_rate"):
            sr = float(self._csv_meta["sample_rate"])
        self._sample_rate = sr
        self.window.set_mup_sampling_text(sr if sr > 0 else 1280.0)
        self.window.set_status(
            f"Loaded {filtered_channels.shape[1]:,} samples × "
            f"{self._num_channels} channels @ {sr:,.0f} Hz"
        )

    def _on_features(self, feat_dict: dict):
        self._features = feat_dict
        n_ch = len(feat_dict["per_channel"])
        n_win = feat_dict["per_channel"][0].shape[1] if n_ch else 0
        n_feat = len(feat_dict["names"])
        self.window.set_status(
            self.window.status_label.text()
            + f"  |  {n_feat} features × {n_win} windows extracted"
        )

    def _on_error(self, msg: str):
        self.window.show_progress(100)
        self.window.btn_load_csv.setEnabled(True)
        self.window.set_status(f"Error: {msg}")
        QMessageBox.critical(self.window, "Load Error", msg)

    # ── Plot helpers ──────────────────────────────────────────────────

    def _gain_multiplier(self) -> float:
        return self.window.gain_slider.value() / 10.0

    def _preview_plots(self):
        """Concatenate accumulated chunks and push to curves (interim view)."""
        t = np.concatenate(self._time_chunks)
        ch = np.concatenate(self._ch_chunks, axis=1)
        gain = self._gain_multiplier()
        for i in range(ch.shape[0]):
            self.window.update_curve(i, t, ch[i] * gain)
        self.window.update_overlap_traces(t, ch * gain)

    def _refresh_plots(self):
        """Redraw all curves from the fully-loaded data."""
        if self._time is None or self._channels is None:
            return
        gain = self._gain_multiplier()
        for i in range(self._num_channels):
            self.window.update_curve(i, self._time, self._channels[i] * gain)
        self.window.update_overlap_traces(self._time, self._channels * gain)

    def _on_gain_changed(self):
        self.window.set_gain_scale(self.window.gain_slider.value())
        self._refresh_plots()

    def _on_auto_fit_y_range(self):
        if self._channels is None or self._num_channels <= 0:
            self.window.set_status("Load a CSV before using Auto Fit Y.")
            return
        gain = self._gain_multiplier()
        visible = self._channels[:self._num_channels] * gain
        if visible.size == 0:
            self.window.set_status("No channel data available for Auto Fit Y.")
            return
        y_min = float(np.min(visible))
        y_max = float(np.max(visible))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            self.window.set_status("Auto Fit Y failed: invalid signal range.")
            return
        if y_min == y_max:
            pad = max(abs(y_min) * 0.01, 0.1)
        else:
            # Use a tiny safety margin so the waveform touches the frame
            # closely without edge clipping.
            pad = max((y_max - y_min) * 0.0025, 0.1)
        self.window.set_fixed_y_range(y_min - pad, y_max + pad)
        self.window.set_status(
            f"Auto Fit Y applied: [{y_min - pad:.2f}, {y_max + pad:.2f}]"
        )

    def _on_filter_settings_changed(self):
        if self._raw_channels is None:
            return
        if self._refilter_thread is not None and self._refilter_thread.isRunning():
            return

        self.window.set_status("Applying filter updates...")
        self._refilter_thread = RefilterThread(
            raw_channels=self._raw_channels,
            sample_rate=self._sample_rate if self._sample_rate > 0 else 1280.0,
            filter_settings=self.window.get_filter_settings(),
        )
        self._refilter_thread.refilter_finished.connect(self._on_refilter_finished)
        self._refilter_thread.refilter_error.connect(self._on_refilter_error)
        self._refilter_thread.start()

    def _on_refilter_finished(self, filtered_channels: np.ndarray):
        self._channels = filtered_channels
        self._refresh_plots()
        self.window.set_status("Filters updated.")

    def _on_refilter_error(self, msg: str):
        self.window.set_status(f"Filter update failed: {msg}")

    def _on_annotation_selected(self, region):
        if self._time is None or self._channels is None:
            return

        x0, x1 = region.getRegion()
        start = min(float(x0), float(x1))
        end = max(float(x0), float(x1))
        ch_idx = int(region.channel_idx)
        if ch_idx < 0 or ch_idx >= self._channels.shape[0]:
            return

        s_idx = int(np.searchsorted(self._time, start, side="left"))
        e_idx = int(np.searchsorted(self._time, end, side="right"))
        if e_idx - s_idx < 4:
            return

        segment = self._channels[ch_idx, s_idx:e_idx].astype(np.float32)
        duration = max(0.0, end - start)
        amplitude = float(calc_peak_to_peak(segment))
        rms = float(calc_rms(segment))
        zcr = float(calc_zcr(segment))
        symmetry = float(calc_symmetry_index(segment))
        decrescendo = float(calc_envelope_decrescendo(segment))

        # Dominant frequency estimate from zero-crossings (fast + stable).
        frequency = float((zcr * 0.5 * len(segment)) / max(duration, 1e-6))
        burst_idx, _env, burst_thr = detect_micro_bursts(
            segment,
            sample_rate=self._sample_rate if self._sample_rate > 0 else 1280.0,
            rms_window_ms=50.0,
            min_duration_ms=20.0,
            threshold_k=3.0,
        )
        burst_spans: list[tuple[float, float]] = []
        for s_rel, e_rel in burst_idx.tolist():
            s_abs = s_idx + int(s_rel)
            e_abs = min(e_idx - 1, s_idx + int(e_rel) - 1)
            burst_spans.append((float(self._time[s_abs]), float(self._time[e_abs])))

        self.window.set_analysis_values(rms, zcr, symmetry, decrescendo)
        self.window.set_micro_burst_values(len(burst_spans), float(burst_thr))
        self.window.show_micro_burst_overlay(ch_idx, burst_spans)
        self.window.update_annotation_metrics(region, duration, amplitude, frequency)

    # ── Annotation export ─────────────────────────────────────────────

    def _on_save_annotations(self):
        if not self._current_csv_path:
            QMessageBox.warning(
                self.window,
                "No CSV Linked",
                "Load a CSV first so the annotation file can be tagged to its source recording.",
            )
            return
        data = self.window.get_annotations_data()
        if not data:
            self.window.set_status("No annotations to save.")
            return

        normalized = self._standardize_v2_annotations(data)

        path, _ = QFileDialog.getSaveFileName(
            self.window, "Save Annotations", "annotations_v2.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

        linked_csv_abs = os.path.abspath(self._current_csv_path) if self._current_csv_path else ""
        linked_csv_name = os.path.basename(linked_csv_abs) if linked_csv_abs else ""
        payload = {
            "schema_version": "ionm_v2_annotations_1",
            "pipeline": "v2_muap_only",
            "linked_csv": {
                "path": linked_csv_abs,
                "basename": linked_csv_name,
            },
            "sample_rate_hz": float(self._sample_rate) if self._sample_rate > 0 else 0.0,
            "annotations": normalized,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self.window.set_status(
            f"Saved {len(normalized)} annotation(s) to {os.path.basename(path)}"
        )

    @staticmethod
    def _standardize_v2_annotations(annotations: list[dict]) -> list[dict]:
        """
        Normalize labels to MUP/Artifact and auto-assign mu_id for MUP regions.
        """
        normalized: list[dict] = []
        for ann in annotations:
            label_raw = str(ann.get("label", "")).replace("AI:", "").strip()
            label_up = label_raw.upper()
            clinical_map = {
                "SPIKE": "Spike",
                "BURST": "Burst",
                "A-TRAIN (HIGH RISK)": "A-Train (High Risk)",
                "A-TRAIN": "A-Train (High Risk)",
                "B-TRAIN": "B-Train",
                "C-TRAIN (POLYRHYTHMIC)": "C-Train (Polyrhythmic)",
                "C-TRAIN": "C-Train (Polyrhythmic)",
                "ARTIFACT": "Artifact",
                "MUP": "MUP",
            }
            if label_up in clinical_map:
                label = clinical_map[label_up]
            elif label_up.startswith("MU"):
                label = "MUP"
            elif label_raw:
                label = label_raw
            else:
                label = "MUP"
            normalized.append(
                {
                    "channel_idx": int(ann.get("channel_idx", 0)),
                    "channel": str(ann.get("channel", "")),
                    "start_time": float(ann.get("start_time", 0.0)),
                    "end_time": float(ann.get("end_time", 0.0)),
                    "label": label,
                    "mu_id": str(ann.get("mu_id", "")).strip(),
                    "duration": float(ann.get("duration", 0.0)),
                    "amplitude": float(ann.get("amplitude", 0.0)),
                    "frequency": float(ann.get("frequency", 0.0)),
                }
            )

        # Online morphology grouping per channel -> deterministic MU IDs.
        clusters_per_ch: dict[int, list[dict]] = {}
        next_mu_id = 1
        ordered_idx = sorted(
            range(len(normalized)),
            key=lambda i: (int(normalized[i]["channel_idx"]), float(normalized[i]["start_time"])),
        )
        for idx in ordered_idx:
            ann = normalized[idx]
            if ann["label"] != "MUP":
                ann["mu_id"] = ""
                continue
            # Honor explicit MU IDs when provided.
            if str(ann["mu_id"]).upper().startswith("MU"):
                continue

            ch = int(ann["channel_idx"])
            amp = float(ann["amplitude"])
            dur = float(ann["duration"])
            freq = float(ann["frequency"])
            feat = np.array([amp, dur, freq], dtype=np.float64)
            clusters = clusters_per_ch.setdefault(ch, [])

            best_k = -1
            best_score = 1e9
            for k, c in enumerate(clusters):
                cfeat = c["centroid"]
                d_amp = abs(feat[0] - cfeat[0]) / max(abs(cfeat[0]), 25.0)
                d_dur = abs(feat[1] - cfeat[1]) / max(abs(cfeat[1]), 0.02)
                d_freq = abs(feat[2] - cfeat[2]) / max(abs(cfeat[2]), 20.0)
                score = 0.5 * d_amp + 0.2 * d_dur + 0.3 * d_freq
                if score < best_score:
                    best_score = score
                    best_k = k

            # Threshold tuned for within-channel MU morphology similarity.
            if best_k >= 0 and best_score <= 0.9:
                c = clusters[best_k]
                c["count"] += 1
                alpha = 1.0 / float(c["count"])
                c["centroid"] = (1.0 - alpha) * c["centroid"] + alpha * feat
                ann["mu_id"] = c["mu_id"]
            else:
                mu_id = f"MU{next_mu_id}"
                next_mu_id += 1
                clusters.append(
                    {"mu_id": mu_id, "centroid": feat, "count": 1}
                )
                ann["mu_id"] = mu_id
        return normalized

    def _on_load_annotations(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Load Annotations",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self.window, "Load Error", f"Failed to read annotation file:\n{exc}")
            return

        if isinstance(payload, list):
            # Backward compatibility with legacy save format.
            anns = payload
            linked_basename = ""
        else:
            anns = list(payload.get("annotations", []))
            linked = payload.get("linked_csv", {})
            linked_basename = str(linked.get("basename", "")).strip()

        if not anns:
            QMessageBox.warning(self.window, "Load Annotations", "No annotations found in file.")
            return

        if linked_basename and self._current_csv_path:
            cur_base = os.path.basename(self._current_csv_path)
            if cur_base != linked_basename:
                answer = QMessageBox.question(
                    self.window,
                    "CSV Mismatch",
                    f"Annotation file is tagged for '{linked_basename}', current CSV is '{cur_base}'.\nLoad anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return

        self.window.set_annotations_data(anns)
        self.window.set_status(f"Loaded {len(anns)} annotation(s) from {os.path.basename(path)}")

    def _on_export_training_csv(self):
        if self._time is None or self._channels is None:
            QMessageBox.warning(self.window, "No Data", "Load a CSV file first.")
            return
        data = self.window.get_annotations_data()
        if not data:
            QMessageBox.warning(
                self.window,
                "No Annotations",
                "Draw or run Gold Standard to create event regions before exporting.",
            )
            return

        normalized = self._standardize_v2_annotations(data)
        default_name = "training_dataset.csv"
        if self._current_csv_path:
            base = os.path.splitext(os.path.basename(self._current_csv_path))[0]
            default_name = f"{base}_training.csv"

        path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Export Training CSV",
            default_name,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        ch_names = list(self._csv_meta.get("channel_names", []))
        try:
            export_training_csv(
                path,
                self._time,
                self._channels,
                normalized,
                sample_rate_hz=float(self._sample_rate) if self._sample_rate > 0 else 1280.0,
                channel_names=ch_names,
                linked_source_csv=self._current_csv_path or "",
            )
        except Exception as exc:
            QMessageBox.critical(self.window, "Export Failed", str(exc))
            return

        self.window.set_status(
            f"Exported training CSV: {len(normalized)} event(s), {self._channels.shape[1]:,} samples"
        )

    def _on_load_training_csv(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self.window,
                "Load Training CSV",
                "",
                "CSV Files (*.csv);;All Files (*)",
            )
        if not path:
            return

        self.window.btn_load_csv.setEnabled(False)
        self.window.btn_load_training_csv.setEnabled(False)
        self.window.show_progress(0)
        self.window.set_status(f"Loading training dataset {os.path.basename(path)}…")

        self._training_load_thread = TrainingDatasetLoadThread(path)
        self._training_load_thread.loading_finished.connect(self._on_training_csv_loaded)
        self._training_load_thread.loading_error.connect(self._on_training_csv_error)
        self._training_load_thread.start()

    def _on_training_csv_loaded(
        self,
        full_time: np.ndarray,
        channels: np.ndarray,
        annotations: list,
        meta: dict,
    ):
        self._current_csv_path = str(meta.get("linked_source_path") or meta.get("linked_source_csv") or "")
        self._time = full_time
        self._raw_channels = channels.copy()
        self._channels = channels.copy()
        self._num_channels = int(channels.shape[0])
        self._csv_meta = meta or {}
        self._sample_rate = float(meta.get("sample_rate_hz", 1280.0))

        self.window.set_active_channels(self._num_channels)
        ch_names = list(meta.get("channel_names", []))
        if ch_names:
            self.window.set_channel_names(ch_names)
        if len(full_time) > 0:
            self.window.set_time_extent(0.0, float(full_time[-1]))
        self._refresh_plots()
        self.window.clear_annotations()
        self.window.set_annotations_data(annotations)
        self.window.set_mup_sampling_text(self._sample_rate)
        self.window.show_progress(100)
        self.window.btn_load_csv.setEnabled(True)
        self.window.btn_load_training_csv.setEnabled(True)
        self.window.set_status(
            f"Training CSV: {channels.shape[1]:,} samples × {self._num_channels} ch, "
            f"{len(annotations)} annotation(s)"
        )

    def _on_training_csv_error(self, msg: str):
        self.window.show_progress(100)
        self.window.btn_load_csv.setEnabled(True)
        self.window.btn_load_training_csv.setEnabled(True)
        self.window.set_status(f"Training CSV error: {msg}")
        QMessageBox.critical(self.window, "Load Training CSV", msg)

    # ── AI Training ───────────────────────────────────────────────────

    def _on_start_training(self):
        if self._time is None or self._channels is None:
            QMessageBox.warning(
                self.window, "No Data", "Load a CSV file first."
            )
            return

        annotations = self.window.get_annotations_data()
        if len(annotations) < 2:
            QMessageBox.warning(
                self.window, "Not Enough Annotations",
                "Draw and label at least 2 annotation regions before training.",
            )
            return
        normalized = self._standardize_v2_annotations(annotations)
        mu_ready = sum(1 for ann in normalized if str(ann.get("mu_id", "")).strip())
        if mu_ready < 2:
            QMessageBox.warning(
                self.window,
                "Not Enough MUP",
                "Mark at least 2 MUP regions (non-Artifact) for v2 training.",
            )
            return

        if not self._ensure_torch_runtime(show_dialog=True):
            return

        self.window.btn_train.setEnabled(False)
        self.window.btn_batch_train_folder.setEnabled(False)
        self.window.btn_load_csv.setEnabled(False)
        self._set_model_snapshot_controls_enabled(False)
        self.window.show_progress(0)
        self.window.set_status("Preparing v2 training…")
        self._trainer = AITrainingThreadV2(
            time_arr=self._time,
            channels_arr=self._channels,
            annotations=normalized,
            sample_rate=int(round(self._sample_rate)) if self._sample_rate > 0 else 1280,
            save_path="model_weights_v2.pth",
        )
        self._trainer.batch_metrics.connect(self._on_batch_metrics)
        self._trainer.epoch_metrics.connect(self._on_epoch_metrics)
        self._trainer.training_finished.connect(self._on_training_done)
        self._trainer.training_error.connect(self._on_training_error)
        self._trainer.start()

    def _on_batch_metrics(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        train_loss: float,
        val_loss: float,
        f1_score: float,
        ram_gb: float,
        vram_gb: float,
    ):
        epoch_ratio = (epoch - 1) + (batch / max(1, total_batches))
        total_epochs = max(1, self._trainer._epochs if self._trainer else 1)
        pct = int((epoch_ratio / total_epochs) * 100)
        self.window.show_progress(pct)
        self.window.set_loss_text(f"Current Model Loss: {train_loss:.5f}")
        self.window.set_val_loss_text(f"Validation Loss: {val_loss:.5f}")
        self.window.set_f1_text(f"F1-Score: {f1_score:.4f}")
        self.window.set_mup_live_metrics(
            mode="Training",
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            f1_score=float(f1_score),
            decision_status="Updating model criteria...",
        )
        self.window.set_ram_text(
            f"System RAM Usage: {ram_gb:.2f} GB"
            + (f" | VRAM: {vram_gb:.2f} GB" if vram_gb > 0 else "")
        )
        self.window.set_training_progress_text(
            f"Training Progress: Epoch {epoch}, Batch {batch}/{total_batches} ({pct}%)"
        )
        self.window.set_status(
            f"Training E{epoch} B{batch}/{total_batches} | "
            f"train {train_loss:.4f} val {val_loss:.4f} f1 {f1_score:.4f}"
        )

    def _on_epoch_metrics(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        f1_score: float,
        ram_gb: float,
        vram_gb: float,
    ):
        pct = int((epoch / max(1, total_epochs)) * 100)
        self.window.show_progress(pct)
        self.window.set_loss_text(f"Current Model Loss: {train_loss:.5f}")
        self.window.set_val_loss_text(f"Validation Loss: {val_loss:.5f}")
        self.window.set_f1_text(f"F1-Score: {f1_score:.4f}")
        self.window.set_mup_live_metrics(
            mode="Training",
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            f1_score=float(f1_score),
            decision_status="Criteria stabilized for this epoch.",
        )
        self.window.set_ram_text(
            f"System RAM Usage: {ram_gb:.2f} GB"
            + (f" | VRAM: {vram_gb:.2f} GB" if vram_gb > 0 else "")
        )
        self.window.set_training_progress_text(
            f"Training Progress: Epoch {epoch}/{total_epochs} ({pct}%)"
        )
        self.window.set_status(
            f"Epoch {epoch}/{total_epochs} done | "
            f"train {train_loss:.4f} val {val_loss:.4f} f1 {f1_score:.4f}"
        )

    def _on_training_done(self, path: str):
        self.window.show_progress(100)
        self.window.btn_train.setEnabled(True)
        self.window.btn_batch_train_folder.setEnabled(True)
        self.window.btn_load_csv.setEnabled(True)
        self._set_model_snapshot_controls_enabled(True)
        self.window.set_training_progress_text("Training Progress: Completed (100%)")
        self.window.set_mup_live_metrics(
            mode="Training",
            decision_status="Training completed. Ready for MUP inference.",
        )
        self.window.set_status(f"Training complete! Model saved -> {path}")

    def _on_training_error(self, msg: str):
        self.window.show_progress(100)
        self.window.btn_train.setEnabled(True)
        self.window.btn_batch_train_folder.setEnabled(True)
        self.window.btn_load_csv.setEnabled(True)
        self._set_model_snapshot_controls_enabled(True)
        self.window.set_training_progress_text("Training Progress: Failed")
        self.window.set_mup_live_metrics(
            mode="Training",
            decision_status="Training failed. Review data/labels.",
        )
        self.window.set_status(f"Training error: {msg}")
        QMessageBox.critical(self.window, "Training Error", msg)

    def _gold_working_channels(self) -> tuple[np.ndarray, float]:
        """Slice to visible window when enabled to reduce computation."""
        if self._channels is None:
            raise ValueError("No channel data loaded.")
        if (
            self._time is not None
            and self._time.size > 16
            and self.window.chk_gold_visible_window.isChecked()
        ):
            x0, x1 = self.window.get_visible_time_range()
            s = int(np.searchsorted(self._time, x0, side="left"))
            e = int(np.searchsorted(self._time, x1, side="right"))
            e = min(max(e, s + 16), self._time.size)
            return self._channels[:, s:e].copy(), float(self._time[s])
        return self._channels, 0.0

    def _gold_cache_payload(self) -> dict:
        c = self._gold_cache
        return {
            "stages_done": sorted(c.stages_done),
            "clean": c.clean,
            "gated": c.gated,
            "pulses_per_ch": c.pulses_per_ch,
            "events_per_ch": c.events_per_ch,
            "annotations": c.annotations,
            "time_offset_s": c.time_offset_s,
        }

    def _merge_gold_cache_payload(self, payload: dict) -> None:
        c = self._gold_cache
        c.sample_rate = float(self._sample_rate) if self._sample_rate > 0 else 1280.0
        c.stages_done = set(payload.get("stages_done", []))
        c.clean = payload.get("clean")
        c.gated = payload.get("gated")
        c.pulses_per_ch = payload.get("pulses_per_ch")
        c.events_per_ch = payload.get("events_per_ch")
        c.annotations = payload.get("annotations")
        c.n_channels = int(payload.get("n_channels", 0))
        c.n_samples = int(payload.get("n_samples", 0))
        c.time_offset_s = float(payload.get("time_offset_s", 0.0))
        self.window.set_gold_cache_status(c.status_text())

    def _on_start_gold_stage(self, stage: int, *, end_stage: int | None = None) -> None:
        if self._time is None or self._channels is None:
            QMessageBox.warning(self.window, "No Data", "Load a CSV file first.")
            return
        if self._gold_thread is not None and self._gold_thread.isRunning():
            self.window.set_status("Gold standard step already running.")
            return

        try:
            ch_work, t_off = self._gold_working_channels()
        except ValueError as exc:
            QMessageBox.warning(self.window, "No Data", str(exc))
            return

        slice_key = (int(ch_work.shape[0]), int(ch_work.shape[1]), round(t_off, 4))
        if slice_key != getattr(self, "_gold_slice_key", None):
            self._gold_cache.reset()
            self._gold_slice_key = slice_key

        self._gold_cache.time_offset_s = t_off
        self._gold_cache.sample_rate = float(self._sample_rate) if self._sample_rate > 0 else 1280.0

        end = int(end_stage) if end_stage is not None else int(stage)
        self.window.set_gold_step_buttons_enabled(False)
        self.window.btn_inference.setEnabled(False)
        self.window.btn_auto_annotate_test.setEnabled(False)
        self._set_model_snapshot_controls_enabled(False)
        self.window.show_progress(0)
        title = STAGE_LABELS.get(stage, f"Step {stage}")
        self.window.set_status(f"Running {title}: {STAGE_DESCRIPTIONS.get(stage, '')}")
        self.window.set_mup_live_metrics(
            mode="Gold Standard",
            decision_status=f"Step {stage}" + (f"–{end}" if end != stage else ""),
        )

        self._gold_thread = GoldStandardStageThread(
            channels_arr=ch_work,
            sample_rate=self._gold_cache.sample_rate,
            stage=int(stage),
            end_stage=end,
            cache_payload=self._gold_cache_payload(),
            time_offset_s=t_off,
        )
        self._gold_thread.progress_updated.connect(self._on_infer_progress)
        self._gold_thread.stage_finished.connect(self._on_gold_stage_finished)
        self._gold_thread.neurotonic_updated.connect(self._on_neurotonic_update)
        self._gold_thread.annotations_ready.connect(self._on_gold_annotations)
        self._gold_thread.inference_finished.connect(self._on_gold_standard_done)
        self._gold_thread.inference_error.connect(self._on_gold_standard_error)
        self._gold_thread.start()

    def _on_start_gold_standard_all(self) -> None:
        """Run all five steps in one background job (same as Steps 1→5)."""
        self._on_start_gold_stage(1, end_stage=5)

    def _on_gold_stage_finished(self, stage: int, payload: object) -> None:
        if isinstance(payload, dict):
            self._merge_gold_cache_payload(payload)
        self.window.set_status(f"Completed {STAGE_LABELS.get(stage, f'step {stage}')}")

    def _on_gold_annotations(self, marks: list[dict]):
        if not marks:
            return
        try:
            self.window.add_ai_annotations(marks)
        except Exception as exc:
            QMessageBox.warning(
                self.window,
                "Annotation Display",
                f"Computed {len(marks)} events but the UI could not draw all regions.\n"
                f"Use Export Training CSV to keep the results.\n\n{exc}",
            )

    def _on_gold_standard_done(self):
        self.window.btn_inference.setEnabled(True)
        self.window.btn_auto_annotate_test.setEnabled(True)
        self.window.set_gold_step_buttons_enabled(True)
        self._set_model_snapshot_controls_enabled(True)
        self.window.show_progress(100)
        self.window.set_mup_live_metrics(
            mode="Gold Standard",
            decision_status="Gold standard pipeline finished.",
        )
        self.window.set_status("Gold standard annotation pass completed.")

    def _on_gold_standard_error(self, msg: str):
        self.window.btn_inference.setEnabled(True)
        self.window.btn_auto_annotate_test.setEnabled(True)
        self.window.set_gold_step_buttons_enabled(True)
        self._set_model_snapshot_controls_enabled(True)
        self.window.show_progress(100)
        self.window.set_mup_live_metrics(
            mode="Gold Standard",
            decision_status="Gold standard pipeline failed.",
        )
        self.window.set_status(f"Gold standard error: {msg}")
        QMessageBox.critical(self.window, "Gold Standard Error", msg)

    def _on_start_auto_annotate_test(self):
        if self._time is None or self._channels is None:
            QMessageBox.warning(self.window, "No Data", "Load a CSV file first.")
            return

        if not self._ensure_torch_runtime(show_dialog=True):
            return

        model_path = self._latest_model_path(model_key="v2")
        if model_path is None:
            QMessageBox.warning(
                self.window, "No Model Weights",
                "No model_weights_v2.pth found. Train v2 model first.",
            )
            return

        self.window.btn_inference.setEnabled(False)
        self.window.btn_auto_annotate_test.setEnabled(False)
        self.window.btn_gold_standard.setEnabled(False)
        self._set_model_snapshot_controls_enabled(False)
        self.window.show_progress(0)
        self.window.set_status("Running Neurotonic Identification Pipeline (v2)...")
        confidence = float(self.window.slider_confidence.value()) / 100.0
        self.window.set_mup_threshold_text(confidence)
        self.window.set_mup_live_metrics(
            mode="Inference",
            decision_status="Stage1 MUAP detect + Stage2 neurotonic classify...",
        )
        self._neuro_thread = NeurotonicInferenceThread(
            time_arr=self._time,
            channels_arr=self._channels,
            model_path=model_path,
            sample_rate=float(self._sample_rate) if self._sample_rate > 0 else 1280.0,
            pulse_threshold=confidence,
        )
        self._neuro_thread.progress_updated.connect(self._on_infer_progress)
        self._neuro_thread.neurotonic_updated.connect(self._on_neurotonic_update)
        self._neuro_thread.inference_finished.connect(self._on_neurotonic_done)
        self._neuro_thread.inference_error.connect(self._on_infer_error)
        self._neuro_thread.start()

    def _on_infer_progress(self, pct: int, msg: str):
        self.window.show_progress(pct)
        self.window.set_mup_live_metrics(
            mode="Inference",
            decision_status=f"Inference running ({pct}%)",
        )
        self.window.set_status(msg)

    def _on_infer_ready(self, ai_marks: list[dict]):
        self.window.btn_inference.setEnabled(True)
        self.window.btn_auto_annotate_test.setEnabled(True)
        self.window.btn_gold_standard.setEnabled(True)
        self._set_model_snapshot_controls_enabled(True)
        self.window.show_progress(100)
        self.window.add_ai_annotations(ai_marks)
        mean_conf = (
            float(np.mean([float(x.get("confidence", 0.0)) for x in ai_marks]))
            if ai_marks
            else 0.0
        )
        elapsed_ms = 0.0
        if self._infer_started_at is not None:
            elapsed_ms = (perf_counter() - self._infer_started_at) * 1000.0
        self._infer_started_at = None
        per_window_ms = elapsed_ms / max(len(ai_marks), 1)
        self.window.set_mup_live_metrics(
            mode="Inference",
            mean_confidence=mean_conf,
            latency_ms=per_window_ms,
            decision_status=f"{len(ai_marks)} MUP candidate window(s) detected",
        )
        self.window.set_status(f"Auto-annotate complete. {len(ai_marks)} AI marks added.")

    def _on_neurotonic_update(self, label: str, confidence: float, metrics: dict):
        self.window.set_neurotonic_status(label, confidence, metrics)
        ch = int(metrics.get("channel_idx", -1))
        n_ch = int(metrics.get("n_channels", 0))
        ch_prefix = f"Ch{ch + 1}: " if ch >= 0 else ""
        active = metrics.get("active_channels")
        active_txt = f" | {active}/{n_ch} ch active" if active is not None and n_ch > 0 else ""
        self.window.set_status(
            f"Neurotonic: {ch_prefix}{label} ({confidence:.2f}) | "
            f"f={float(metrics.get('mean_frequency_hz', 0.0)):.1f}Hz "
            f"amp={float(metrics.get('rolling_amplitude_uv', 0.0)):.1f}uV"
            f"{active_txt}"
        )

    def _on_neurotonic_done(self):
        self.window.btn_inference.setEnabled(True)
        self.window.btn_auto_annotate_test.setEnabled(True)
        self.window.btn_gold_standard.setEnabled(True)
        self._set_model_snapshot_controls_enabled(True)
        self.window.show_progress(100)
        self.window.set_mup_live_metrics(
            mode="Inference",
            decision_status="Neurotonic pipeline finished.",
        )
        self.window.set_status("Neurotonic identification completed.")

    def _on_infer_error(self, msg: str):
        self.window.btn_inference.setEnabled(True)
        self.window.btn_auto_annotate_test.setEnabled(True)
        self.window.btn_gold_standard.setEnabled(True)
        self._set_model_snapshot_controls_enabled(True)
        self.window.show_progress(100)
        self.window.set_mup_live_metrics(
            mode="Inference",
            decision_status="Inference failed. Criteria not updated.",
        )
        self._infer_started_at = None
        self.window.set_status(f"Auto-annotate error: {msg}")
        QMessageBox.critical(self.window, "Inference Error", msg)

    @staticmethod
    def _latest_model_path(model_key: str = "v2") -> str | None:
        candidates: list[str] = []
        if model_key != "v2":
            return None
        search_list = ("model_weights_v2.pth",)
        for p in search_list:
            if os.path.exists(p):
                candidates.append(p)
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def _on_start_batch_training(self):
        QMessageBox.information(
            self.window,
            "Batch Training",
            "v1 pipeline has been removed. v2 batch trainer is not implemented yet.\n"
            "Use single-file v2 training for now.",
        )

    def _on_batch_file_progress(
        self, file_idx: int, total_files: int, case_name: str, total_samples: int, ram_gb: float
    ):
        pct = int((file_idx / max(1, total_files)) * 100)
        self.window.show_progress(pct)
        self.window.set_ram_text(f"System RAM Usage: {ram_gb:.2f} GB")
        self.window.set_training_progress_text(
            f"Processing File {file_idx} of {total_files}..."
        )
        self.window.set_status(
            f"Batch mode: {case_name} processed | Total Samples Extracted: {total_samples:,}"
        )

    # ── Model snapshot controls ───────────────────────────────────────

    def _create_model_backup(self):
        model_path = "model_weights_v2.pth"
        backup_path = "model_weights_v2.bak"
        if not os.path.exists(model_path):
            QMessageBox.information(
                self.window,
                "No Model Found",
                "No model_weights_v2.pth found. Train a model first.",
            )
            return
        try:
            shutil.copy2(model_path, backup_path)
        except Exception as exc:
            QMessageBox.critical(
                self.window, "Backup Failed", f"Failed to create backup:\n{exc}"
            )
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.window.set_backup_status_text(f"Last Backup: {now}")
        self.window.set_status("Model Snapshot Created Successfully.")

    def _restore_model_backup(self):
        model_path = "model_weights_v2.pth"
        backup_path = "model_weights_v2.bak"
        if not os.path.exists(backup_path):
            QMessageBox.information(
                self.window,
                "No Backup Found",
                "No backup found. Please create a backup first.",
            )
            return

        answer = QMessageBox.question(
            self.window,
            "Confirm Restore",
            "This will overwrite your current model with the saved backup. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        try:
            shutil.copy2(backup_path, model_path)
            self._reload_model_dependent_components()
        except Exception as exc:
            QMessageBox.critical(
                self.window, "Restore Failed", f"Failed to restore backup:\n{exc}"
            )
            return

        self.window.set_status("Model successfully restored to last snapshot.")
        self._refresh_backup_status_on_startup()

    def _reload_model_dependent_components(self):
        # Inference thread loads weights per run from disk; forcing state reset
        # ensures restored model is used in the next inference invocation.
        self._neuro_thread = None

    def _refresh_backup_status_on_startup(self):
        backup_path = "model_weights_v2.bak"
        if not os.path.exists(backup_path):
            self.window.set_backup_status_text("Last Backup: No Backup Found")
            return
        ts = os.path.getmtime(backup_path)
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        self.window.set_backup_status_text(f"Last Backup: {dt}")

    def _set_model_snapshot_controls_enabled(self, enabled: bool):
        self.window.btn_create_backup.setEnabled(enabled)
        self.window.btn_restore_backup.setEnabled(enabled)


# ── Test CSV generator ────────────────────────────────────────────────

def generate_test_csv(
    filepath: str = "data/test_emg.csv",
    duration: float = 5.0,
    sample_rate: int = 10_000,
    num_channels: int = 16,
):
    """Write a synthetic multi-channel EMG CSV for development testing."""
    import pandas as pd

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, dtype=np.float32)
    rng = np.random.default_rng(42)

    data: dict[str, np.ndarray] = {"Time": t}
    for ch in range(1, num_channels + 1):
        freq = 30 + ch * 12
        amp = 0.5 + 0.1 * ch
        sig = amp * np.sin(2 * np.pi * freq * t)

        burst_centre = rng.uniform(1.0, duration - 1.0)
        envelope = np.exp(-0.5 * ((t - burst_centre) / 0.15) ** 2)
        sig += envelope * np.sin(2 * np.pi * rng.uniform(150, 400) * t)
        sig += 0.05 * rng.standard_normal(n)

        data[f"Ch{ch}"] = sig.astype(np.float32)

    pd.DataFrame(data).to_csv(filepath, index=False)
    print(f"Generated: {filepath}  ({n:,} rows, {num_channels} channels)")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    # Required on Windows when gold-standard uses process-based parallelism.
    import multiprocessing as mp
    mp.freeze_support()

    if "--gen-test-csv" in sys.argv:
        idx = sys.argv.index("--gen-test-csv")
        path = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "data/test_emg.csv"
        generate_test_csv(path)
        return

    app = QApplication(sys.argv)
    window = MainWindow()
    controller = AppController(window)  # noqa: F841  prevent GC
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
