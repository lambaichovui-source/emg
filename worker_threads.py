"""
worker_threads.py — QThread-based background workers for data loading,
feature extraction, AI training, and inference.

All heavy I/O and computation runs here so the main thread stays
exclusively responsible for GUI updates.
"""

import os
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from data_handler import (
    DataHandler,
    RingBuffer,
    SignalPreprocessor,
    extract_all_features,
    ROMSTOCK_FEATURE_NAMES,
    DEFAULT_FEATURE_WINDOW,
    DEFAULT_FEATURE_STRIDE,
    calc_isi_features,
    calc_rolling_amplitude,
    detect_and_blank_artifacts,
)


def _human_only_annotations(annotations: list[dict]) -> list[dict]:
    """Drop AI-generated regions and keep only human-authored annotations."""
    clean: list[dict] = []
    for ann in annotations:
        label = str(ann.get("label", "")).strip()
        if label.startswith("AI:"):
            continue
        clean.append(ann)
    return clean


class TrainingDatasetLoadThread(QThread):
    """Load IONM training CSV (waveform + annotations) off the GUI thread."""

    loading_finished = pyqtSignal(object, object, object, object)
    loading_error = pyqtSignal(str)

    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self._filepath = filepath

    def run(self):
        try:
            from annotation_store import load_training_csv

            payload = load_training_csv(self._filepath)
            self.loading_finished.emit(
                payload["time"],
                payload["channels"],
                payload["annotations"],
                payload["metadata"],
            )
        except Exception as exc:
            self.loading_error.emit(str(exc))


class DataLoadingThread(QThread):
    """Load one CSV file with metadata-aware parsing + preprocessing."""

    progress_updated = pyqtSignal(int, str)
    chunk_ready = pyqtSignal(object, object)
    metadata_loaded = pyqtSignal(object)
    loading_finished = pyqtSignal(object, object, object, object)
    features_extracted = pyqtSignal(object)
    loading_error = pyqtSignal(str)

    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self._filepath = filepath
        self._handler = DataHandler()

    def run(self):
        try:
            total_rows = self._handler.estimate_rows(self._filepath)
            self.progress_updated.emit(0, f"Scanning… {total_rows:,} rows detected")

            meta = self._handler.read_csv_metadata(self._filepath)
            self.metadata_loaded.emit(meta)

            preprocessor: SignalPreprocessor | None = None
            time_chunks: list[np.ndarray] = []
            raw_chunks: list[np.ndarray] = []
            rows_loaded = 0
            num_ch = 0

            for t_arr, ch_arr, meta in self._handler.iter_signal_chunks(self._filepath):
                if num_ch == 0:
                    num_ch = ch_arr.shape[0]

                if preprocessor is None:
                    fs = float(meta.get("sample_rate", 0.0))
                    if fs <= 0 and len(t_arr) > 1:
                        dt = float(t_arr[1] - t_arr[0])
                        fs = round(1.0 / dt) if dt > 0 else 1280.0
                    if fs <= 0:
                        fs = 1280.0
                    preprocessor = SignalPreprocessor(sample_rate=fs, num_channels=num_ch)

                notch_chunk = preprocessor.apply_notch(ch_arr) if preprocessor else ch_arr
                time_chunks.append(t_arr)
                raw_chunks.append(ch_arr)
                rows_loaded += len(t_arr)

                pct = min(int(rows_loaded / max(total_rows, 1) * 100), 95)
                self.progress_updated.emit(pct, f"Loading… {rows_loaded:,} / {total_rows:,} rows")
                self.chunk_ready.emit(t_arr, notch_chunk)

            full_time = np.concatenate(time_chunks)
            raw_channels = np.concatenate(raw_chunks, axis=1)

            self.progress_updated.emit(96, "Blanking cautery / SEP artifacts…")
            filtered_channels = (
                preprocessor.blank_artifacts(raw_channels) if preprocessor else raw_channels.copy()
            )

            self._handler.finalize(full_time, filtered_channels)
            self._handler.raw_data_buffer = raw_channels
            self._handler.filtered_data_buffer = filtered_channels
            self._handler.metadata = meta

            sr = self._handler.sample_rate
            self.progress_updated.emit(98, "Extracting Romstöck features…")
            win = DEFAULT_FEATURE_WINDOW
            stride = DEFAULT_FEATURE_STRIDE
            feat_per_ch: list[np.ndarray] = []
            for ch in range(num_ch):
                feat_per_ch.append(
                    extract_all_features(filtered_channels[ch], win, stride).astype(np.float32)
                )
            self.features_extracted.emit(
                {"per_channel": feat_per_ch, "names": list(ROMSTOCK_FEATURE_NAMES), "window": win, "stride": stride}
            )

            self.progress_updated.emit(
                100,
                f"Loaded {rows_loaded:,} samples × {num_ch} ch @ {sr:,.0f} Hz (preprocessed)",
            )
            self.loading_finished.emit(full_time, raw_channels, filtered_channels, meta)

        except Exception as exc:
            self.loading_error.emit(str(exc))


class RefilterThread(QThread):
    """Re-apply user filter settings on in-memory raw channels."""

    refilter_finished = pyqtSignal(object)
    refilter_error = pyqtSignal(str)

    def __init__(self, raw_channels: np.ndarray, sample_rate: float, filter_settings: dict, parent=None):
        super().__init__(parent)
        self._raw = raw_channels
        self._fs = float(sample_rate)
        self._settings = filter_settings

    def run(self):
        try:
            pre = SignalPreprocessor(
                sample_rate=max(self._fs, 1.0),
                num_channels=self._raw.shape[0],
            )
            filtered = pre.apply_user_filters(
                self._raw,
                show_raw=bool(self._settings.get("show_raw", False)),
                enable_notch=bool(self._settings.get("enable_notch", True)),
                notch_freq=float(self._settings.get("notch_hz", 60.0)),
                enable_cautery=bool(self._settings.get("enable_cautery", True)),
                enable_sep=bool(self._settings.get("enable_sep", True)),
                enable_baseline=bool(self._settings.get("enable_baseline", False)),
                baseline_cutoff_hz=float(self._settings.get("baseline_cutoff_hz", 15.0)),
                cautery_threshold=float(self._settings.get("cautery_threshold", 2000.0)),
                cautery_hold_ms=float(self._settings.get("cautery_hold_ms", 150.0)),
                sep_blank_ms=float(self._settings.get("sep_blank_ms", 3.0)),
                sep_trigger=bool(self._settings.get("sep_trigger", False)),
                low_cut_hz=float(self._settings.get("low_cut_hz", 20.0)),
                high_cut_hz=float(self._settings.get("high_cut_hz", 500.0)),
            )
            self.refilter_finished.emit(filtered)
        except Exception as exc:
            self.refilter_error.emit(str(exc))


class AITrainingThreadV2(QThread):
    """Background trainer for the v2 MO-GRU-EA sequence model."""

    batch_metrics = pyqtSignal(int, int, int, float, float, float, float, float)
    epoch_metrics = pyqtSignal(int, int, float, float, float, float, float)
    training_finished = pyqtSignal(str)
    training_error = pyqtSignal(str)

    def __init__(
        self,
        time_arr: np.ndarray,
        channels_arr: np.ndarray,
        annotations: list[dict],
        sample_rate: int = 2048,
        n_mu: int | None = None,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
        validation_split: float = 0.2,
        early_stop_patience: int = 6,
        early_stop_target_loss: float = 0.02,
        save_path: str = "model_weights_v2.pth",
        parent=None,
    ):
        super().__init__(parent)
        self._time = time_arr
        self._channels = channels_arr
        self._annotations = annotations
        self._sample_rate = int(sample_rate)
        self._n_mu = n_mu
        self._epochs = int(epochs)
        self._lr = float(lr)
        self._batch_size = int(batch_size)
        self._validation_split = float(validation_split)
        self._early_stop_patience = int(early_stop_patience)
        self._early_stop_target = float(early_stop_target_loss)
        self._save_path = save_path

    def run(self):
        try:
            import psutil
            import torch
            from ai_model_v2 import (
                MOGRUEANet,
                HDSemgWindowDataset,
                V2DataInterfaceSpec,
                build_pulse_trains_from_annotations,
                multilabel_f1_score,
            )
            from torch.utils.data import DataLoader, random_split

            anns = _human_only_annotations(self._annotations)
            normalized_anns: list[dict] = []
            for ann in anns:
                clean = dict(ann)
                label = str(clean.get("label", "")).replace("AI:", "").strip()
                mu_id = str(clean.get("mu_id", "")).strip()
                if not mu_id:
                    if label.upper().startswith("MU"):
                        mu_id = label
                    elif label:
                        mu_id = f"MU_{label}"
                if mu_id:
                    clean["mu_id"] = mu_id
                normalized_anns.append(clean)

            pulse_trains = build_pulse_trains_from_annotations(
                self._time,
                normalized_anns,
            )
            interface = V2DataInterfaceSpec(
                sample_rate_hz=self._sample_rate,
                window_size=120,
                step_size=20,
                center_span=20,
            )
            dataset = HDSemgWindowDataset(
                channels=self._channels,
                pulse_trains=pulse_trains,
                n_mu=self._n_mu,
                normalize=True,
                interface=interface,
            )
            if len(dataset) < 8:
                self.training_error.emit(
                    f"Need at least 8 window samples for v2 training, got {len(dataset)}."
                )
                return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MOGRUEANet(
                in_channels=self._channels.shape[0],
                n_mu=dataset.n_mu,
            ).to(device)

            val_len = max(1, int(len(dataset) * self._validation_split))
            train_len = len(dataset) - val_len
            if train_len < 1:
                train_len = len(dataset) - 1
                val_len = 1
            train_set, val_set = random_split(dataset, [train_len, val_len])

            train_loader = DataLoader(
                train_set,
                batch_size=min(self._batch_size, len(train_set)),
                shuffle=True,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=min(self._batch_size, len(val_set)),
                shuffle=False,
                drop_last=False,
            )

            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=1e-5)
            process = psutil.Process(os.getpid())

            best_val = float("inf")
            best_state = None
            bad_epochs = 0

            for epoch in range(1, self._epochs + 1):
                model.train()
                running_loss = 0.0
                n_batches = 0
                total_batches = max(1, len(train_loader))

                for batch_idx, (x_batch, y_batch) in enumerate(train_loader, start=1):
                    x_batch = x_batch.to(device=device, dtype=torch.float32)
                    y_batch = y_batch.to(device=device, dtype=torch.float32)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

                    running_loss += float(loss.item())
                    n_batches += 1

                    train_avg = running_loss / max(1, n_batches)
                    val_loss, val_f1 = self._validate_v2(
                        model=model,
                        val_loader=val_loader,
                        criterion=criterion,
                        device=device,
                        threshold=0.5,
                        f1_fn=multilabel_f1_score,
                    )
                    ram = process.memory_info().rss / (1024 ** 3)
                    vram = (
                        torch.cuda.memory_allocated(device) / (1024 ** 3)
                        if device.type == "cuda"
                        else 0.0
                    )
                    self.batch_metrics.emit(
                        epoch,
                        batch_idx,
                        total_batches,
                        float(train_avg),
                        float(val_loss),
                        float(val_f1),
                        float(ram),
                        float(vram),
                    )

                avg_loss = running_loss / max(1, n_batches)
                val_loss, val_f1 = self._validate_v2(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    device=device,
                    threshold=0.5,
                    f1_fn=multilabel_f1_score,
                )
                ram = process.memory_info().rss / (1024 ** 3)
                vram = (
                    torch.cuda.memory_allocated(device) / (1024 ** 3)
                    if device.type == "cuda"
                    else 0.0
                )
                self.epoch_metrics.emit(
                    epoch,
                    self._epochs,
                    float(avg_loss),
                    float(val_loss),
                    float(val_f1),
                    float(ram),
                    float(vram),
                )

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                    os.makedirs(os.path.dirname(self._save_path) or ".", exist_ok=True)
                    torch.save(best_state, self._save_path)
                else:
                    bad_epochs += 1

                if val_loss <= self._early_stop_target or bad_epochs >= self._early_stop_patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            self.training_finished.emit(os.path.abspath(self._save_path))
        except Exception as exc:
            self.training_error.emit(str(exc))

    @staticmethod
    def _validate_v2(model, val_loader, criterion, device, threshold: float, f1_fn):
        import torch

        model.eval()
        running = 0.0
        n = 0
        y_true: list[np.ndarray] = []
        y_prob: list[np.ndarray] = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device=device, dtype=torch.float32)
                y_batch = y_batch.to(device=device, dtype=torch.float32)
                logits = model(x_batch)
                running += float(criterion(logits, y_batch).item())
                n += 1
                y_true.append(y_batch.cpu().numpy())
                y_prob.append(torch.sigmoid(logits).cpu().numpy())

        if y_true:
            yt = np.concatenate(y_true, axis=0)
            yp = np.concatenate(y_prob, axis=0)
            f1 = float(f1_fn(yt, yp, threshold=threshold))
        else:
            f1 = 0.0
        return running / max(1, n), f1


class GoldStandardStageThread(QThread):
    """Run one gold-standard step (1–5) or steps 1–5 in sequence."""

    progress_updated = pyqtSignal(int, str)
    stage_finished = pyqtSignal(int, object)
    annotations_ready = pyqtSignal(object)
    neurotonic_updated = pyqtSignal(str, float, object)
    inference_finished = pyqtSignal()
    inference_error = pyqtSignal(str)

    def __init__(
        self,
        channels_arr: np.ndarray,
        sample_rate: float,
        stage: int,
        *,
        end_stage: int | None = None,
        cache_payload: dict | None = None,
        max_channels: int = 16,
        annotation_pad_s: float = 0.02,
        time_offset_s: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self._channels = channels_arr.astype(np.float32, copy=False)
        self._sample_rate = float(max(sample_rate, 1.0))
        self._stage = int(stage)
        self._end_stage = int(end_stage) if end_stage is not None else int(stage)
        self._max_channels = int(max(1, max_channels))
        self._pad_s = float(max(annotation_pad_s, 0.0))
        self._time_offset_s = float(time_offset_s)
        self._cache_payload = cache_payload or {}

    def run(self):
        try:
            from gold_standard_pipeline import GoldStandardCache, run_stage, run_stages

            if self._channels.ndim != 2 or self._channels.shape[1] < 16:
                self.inference_error.emit("Not enough loaded signal for gold-standard analysis.")
                return

            cache = GoldStandardCache(
                sample_rate=self._sample_rate,
                time_offset_s=self._time_offset_s,
            )
            if self._cache_payload:
                for key, val in self._cache_payload.items():
                    if hasattr(cache, key):
                        setattr(cache, key, val)
                cache.stages_done = set(self._cache_payload.get("stages_done", []))

            def _progress(pct: int, msg: str) -> None:
                self.progress_updated.emit(int(pct), msg)

            if self._stage == self._end_stage:
                run_stage(
                    self._stage,
                    cache,
                    self._channels,
                    max_channels=self._max_channels,
                    annotation_pad_s=self._pad_s,
                    on_progress=_progress,
                )
            else:
                run_stages(
                    self._stage,
                    self._end_stage,
                    cache,
                    self._channels,
                    max_channels=self._max_channels,
                    annotation_pad_s=self._pad_s,
                    on_progress=_progress,
                )

            payload = {
                "stages_done": sorted(cache.stages_done),
                "clean": cache.clean,
                "gated": cache.gated,
                "pulses_per_ch": cache.pulses_per_ch,
                "events_per_ch": cache.events_per_ch,
                "annotations": cache.annotations,
                "n_channels": cache.n_channels,
                "n_samples": cache.n_samples,
                "time_offset_s": cache.time_offset_s,
            }
            self.stage_finished.emit(self._end_stage, payload)

            if cache.annotations:
                marks = []
                for ann in cache.annotations:
                    row = dict(ann)
                    row["label"] = f"AI: {row.get('label', 'Burst')}"
                    row["confidence"] = 0.85
                    marks.append(row)
                self.annotations_ready.emit(marks)

            self._emit_summary(cache)
            self.progress_updated.emit(100, cache.status_text())
            self.inference_finished.emit()
        except Exception as exc:
            self.inference_error.emit(str(exc))

    def _emit_summary(self, cache) -> None:
        label = "None"
        conf = 0.0
        ch_best = -1
        if cache.events_per_ch:
            for ch, events in enumerate(cache.events_per_ch):
                if not events:
                    continue
                best = max(events, key=lambda e: float(e.get("duration", 0.0)))
                lab = str(best.get("label", "None"))
                if lab not in ("None", "Artifact"):
                    label = lab
                    conf = 0.85
                    ch_best = ch
        n_pulse = 0
        if cache.pulses_per_ch:
            n_pulse = sum(int(p.size) for p in cache.pulses_per_ch)
        self.neurotonic_updated.emit(
            label,
            conf,
            {
                "channel_idx": ch_best,
                "n_channels": cache.n_channels,
                "pulse_count": n_pulse,
                "n_annotations": len(cache.annotations or []),
                "stages_done": sorted(cache.stages_done),
            },
        )


# Backward-compatible alias
GoldStandardInferenceThread = GoldStandardStageThread


class NeurotonicInferenceThread(QThread):
    """Two-stage neurotonic pipeline (MUAP detect -> train classification)."""

    progress_updated = pyqtSignal(int, str)
    neurotonic_updated = pyqtSignal(str, float, object)
    inference_finished = pyqtSignal()
    inference_error = pyqtSignal(str)

    def __init__(
        self,
        time_arr: np.ndarray,
        channels_arr: np.ndarray,
        model_path: str = "model_weights_v2.pth",
        sample_rate: float = 1280.0,
        window_size: int = 120,
        step_size: int = 20,
        chunk_seconds: float = 0.25,
        pulse_threshold: float = 0.5,
        parent=None,
    ):
        super().__init__(parent)
        self._time = time_arr.astype(np.float64, copy=False)
        self._channels = channels_arr.astype(np.float32, copy=False)
        self._model_path = model_path
        self._sample_rate = float(max(sample_rate, 1.0))
        self._window_size = int(window_size)
        self._step_size = int(step_size)
        self._chunk_s = float(max(chunk_seconds, 0.05))
        self._pulse_threshold = float(pulse_threshold)
        self._history_s = 3.0
        self._rolling_timestamps_s = np.empty((0,), dtype=np.float64)

    def run(self):
        try:
            import torch
            from ai_model_v2 import MOGRUEANet, mapminmax_normalize, frame_signal
            from neurotonic_logic import NeurotonicClassifier

            if not os.path.exists(self._model_path):
                self.inference_error.emit(f"Model file not found: {self._model_path}")
                return
            if self._channels.ndim != 2 or self._channels.shape[1] < self._window_size:
                self.inference_error.emit("Not enough loaded signal for neurotonic inference.")
                return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state = torch.load(self._model_path, map_location=device, weights_only=True)
            # Recover architecture from either legacy or rebuilt state_dict keys.
            if "input_proj.weight" in state:
                in_channels = int(state["input_proj.weight"].shape[1])
            elif "sequence_model.backbone.proj.0.weight" in state:
                in_channels = int(state["sequence_model.backbone.proj.0.weight"].shape[1])
            else:
                raise KeyError("Unable to infer input channel count from checkpoint.")

            if "fc_out.weight" in state:
                n_mu = int(state["fc_out.weight"].shape[0])
            elif "sequence_model.head.3.weight" in state:
                n_mu = int(state["sequence_model.head.3.weight"].shape[0])
            else:
                raise KeyError("Unable to infer output MU count from checkpoint.")
            model = MOGRUEANet(in_channels=in_channels, n_mu=n_mu).to(device)
            model.load_state_dict(state)
            model.eval()
            classifier = NeurotonicClassifier()

            n_samples = self._channels.shape[1]
            chunk_samples = max(1, int(self._chunk_s * self._sample_rate))
            total_chunks = max(1, int(np.ceil(n_samples / chunk_samples)))

            # Thread-safe rolling buffers: signal + timeline.
            signal_rb = RingBuffer(capacity=max(chunk_samples * 8, int(self._sample_rate * 2.0)), num_channels=self._channels.shape[0])
            time_rb = RingBuffer(capacity=max(chunk_samples * 8, int(self._sample_rate * 2.0)), num_channels=1, dtype=np.float64)

            processed_chunks = 0
            with torch.no_grad():
                for start in range(0, n_samples, chunk_samples):
                    end = min(start + chunk_samples, n_samples)
                    raw_chunk = self._channels[:, start:end]
                    t_chunk = self._time[start:end]

                    # Pre-inference artifact blanking for each channel.
                    clean_chunk = np.empty_like(raw_chunk, dtype=np.float32)
                    for ch in range(raw_chunk.shape[0]):
                        clean_chunk[ch] = detect_and_blank_artifacts(
                            raw_chunk[ch].astype(np.float32, copy=False),
                            threshold=2000.0,
                        ).astype(np.float32, copy=False)

                    signal_rb.append(clean_chunk)
                    time_rb.append(t_chunk.reshape(1, -1))

                    # Fixed-size read from thread-safe ring: latest 1 second.
                    sig_1s = signal_rb.get_last_seconds(self._sample_rate, 1.0)
                    t_1s = time_rb.get_last_seconds(self._sample_rate, 1.0)
                    if sig_1s.shape[1] < self._window_size or t_1s.shape[1] < self._window_size:
                        continue

                    sig_norm = mapminmax_normalize(sig_1s)
                    windows, starts = frame_signal(
                        sig_norm,
                        window_size=self._window_size,
                        step_size=self._step_size,
                    )
                    if windows.shape[0] == 0:
                        continue

                    xb = torch.from_numpy(windows.astype(np.float32, copy=False)).to(device)
                    logits = model(xb)
                    probs = torch.sigmoid(logits).cpu().numpy()  # (n_win, n_mu)
                    pulse_mask = np.any(probs >= self._pulse_threshold, axis=1)
                    if np.any(pulse_mask):
                        center_offset = self._window_size // 2
                        center_idx = starts[pulse_mask] + center_offset
                        center_idx = center_idx[(center_idx >= 0) & (center_idx < t_1s.shape[1])]
                        new_ts = t_1s[0, center_idx].astype(np.float64, copy=False)
                        if new_ts.size > 0:
                            self._rolling_timestamps_s = np.concatenate([self._rolling_timestamps_s, new_ts], axis=0)

                    # Keep rolling multi-second history so Train (>1s) is detectable.
                    now_t = float(t_1s[0, -1])
                    keep = self._rolling_timestamps_s >= (now_t - self._history_s)
                    self._rolling_timestamps_s = self._rolling_timestamps_s[keep]

                    # Stage-2 features from the most recent 1-second window.
                    recent_1s = self._rolling_timestamps_s[self._rolling_timestamps_s >= (now_t - 1.0)]
                    mean_freq_hz, isi_var_s2 = calc_isi_features(
                        recent_1s.astype(np.float64, copy=False)
                    )
                    if recent_1s.size > 0:
                        idx = np.searchsorted(t_1s[0], recent_1s, side="left").astype(np.int64)
                        idx = idx[(idx >= 0) & (idx < sig_1s.shape[1])]
                    else:
                        idx = np.empty((0,), dtype=np.int64)
                    mean_amp_uv = calc_rolling_amplitude(
                        sig_1s[0].astype(np.float32, copy=False),
                        idx.astype(np.int64, copy=False),
                    )

                    label, confidence, metrics = classifier.classify(
                        history_timestamps_s=self._rolling_timestamps_s.astype(np.float64, copy=False),
                        mean_frequency_hz=float(mean_freq_hz),
                        isi_variance_s2=float(isi_var_s2),
                        rolling_amplitude_uv=float(mean_amp_uv),
                        pulse_count_1s=int(recent_1s.size),
                        window_s=1.0,
                    )
                    metrics["muap_windows"] = int(np.sum(pulse_mask))
                    metrics["time_s"] = now_t
                    self.neurotonic_updated.emit(label, float(confidence), metrics)

                    processed_chunks += 1
                    pct = int(processed_chunks / max(total_chunks, 1) * 100)
                    self.progress_updated.emit(
                        pct,
                        f"Neurotonic pipeline: {processed_chunks}/{total_chunks} chunks",
                    )

            self.inference_finished.emit()
        except Exception as exc:
            self.inference_error.emit(str(exc))
