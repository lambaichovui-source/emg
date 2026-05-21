"""
ai_model_v2.py

Research-oriented MO-GRU-EA implementation for online HD-sEMG decomposition.

This module is intentionally self-contained and provides:
1) Data interface helpers (normalization, framing, pulse-label conversion)
2) External Attention + Bi-GRU multi-output network (MO-GRU-EA)
3) Dataset for multi-label MU pulse detection
4) Training utilities with early stopping (BCE objective)
5) Single-window latency benchmark utility

Target specification highlights:
- Sample rate: 2048 Hz (configurable, default fixed to 2048)
- Window size: 120 samples
- Step size: 20 samples
- Multi-label output: one sigmoid probability per MU
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------- Defaults ---------------------------------

DEFAULT_SAMPLE_RATE = 2048
DEFAULT_WINDOW_SIZE = 120
DEFAULT_STEP_SIZE = 20
DEFAULT_CENTER_LABEL_SPAN = 20


# ----------------------- Data Interface Contracts ------------------------

@dataclass
class V2DataInterfaceSpec:
    """Contract for v2 data shape and label conventions."""

    sample_rate_hz: int = DEFAULT_SAMPLE_RATE
    window_size: int = DEFAULT_WINDOW_SIZE
    step_size: int = DEFAULT_STEP_SIZE
    center_span: int = DEFAULT_CENTER_LABEL_SPAN
    mapminmax_range: tuple[float, float] = (-1.0, 1.0)


@dataclass
class V2TrainingConfig:
    """Training hyperparameters for MO-GRU-EA."""

    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    val_split: float = 0.2
    positive_class_weight: float = 1.0
    early_stop_patience: int = 6
    early_stop_target_loss: float = 0.02
    threshold: float = 0.5
    weight_decay: float = 1e-5
    save_path: str = "model_weights_v2.pth"


def mapminmax_normalize(
    x: np.ndarray,
    feature_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """MapMinMax normalization per channel.

    Parameters
    ----------
    x : np.ndarray
        Shape: (channels, samples)
    feature_range : (float, float)
        Output range, default [-1, 1].
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D array (channels, samples), got shape={x.shape}")

    lo, hi = feature_range
    if not hi > lo:
        raise ValueError("feature_range must satisfy hi > lo")

    x = x.astype(np.float32, copy=True)
    ch_min = np.min(x, axis=1, keepdims=True)
    ch_max = np.max(x, axis=1, keepdims=True)
    denom = np.maximum(ch_max - ch_min, 1e-8)
    x01 = (x - ch_min) / denom
    return (x01 * (hi - lo) + lo).astype(np.float32)


def build_pulse_trains_from_annotations(
    time_arr: np.ndarray,
    annotations: list[dict],
    *,
    mu_key_candidates: tuple[str, ...] = ("mu_id", "mu", "motor_unit", "unit_id", "unit"),
) -> dict[str, np.ndarray]:
    """Convert interval annotations to MU pulse indices (one pulse per interval center).

    Notes
    -----
    - This adapter is strict by design. If MU IDs are unavailable, it raises
      a clear error because multi-output decomposition requires MU-specific targets.
    - Input annotations must include:
      - start/end time fields: `start_time`, `end_time`
      - MU identifier in one of `mu_key_candidates`
    """
    if time_arr.ndim != 1:
        raise ValueError("time_arr must be 1-D.")
    if len(time_arr) < 2:
        raise ValueError("time_arr must contain at least two samples.")

    mu_to_idx: dict[str, list[int]] = {}
    for ann in annotations:
        mu_id = None
        for key in mu_key_candidates:
            if key in ann and str(ann[key]).strip():
                mu_id = str(ann[key]).strip()
                break
        if mu_id is None:
            continue

        st = float(ann.get("start_time", 0.0))
        et = float(ann.get("end_time", st))
        center_t = (st + et) * 0.5
        pulse_idx = int(np.searchsorted(time_arr, center_t, side="left"))
        if pulse_idx < 0 or pulse_idx >= len(time_arr):
            continue
        mu_to_idx.setdefault(mu_id, []).append(pulse_idx)

    if not mu_to_idx:
        raise ValueError(
            "No MU IDs found in annotations. "
            "For MO-GRU-EA multi-output training, provide MU-specific annotation keys: "
            f"{mu_key_candidates}."
        )

    out: dict[str, np.ndarray] = {}
    for k, idxs in mu_to_idx.items():
        uniq = np.unique(np.asarray(idxs, dtype=np.int32))
        out[k] = uniq
    return out


def frame_signal(
    channels: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from continuous signal.

    Parameters
    ----------
    channels : np.ndarray
        Shape: (channels, samples)

    Returns
    -------
    windows : np.ndarray
        Shape: (num_windows, channels, window_size)
    starts : np.ndarray
        Window start indices in sample domain
    """
    if channels.ndim != 2:
        raise ValueError(f"Expected 2-D channels array, got shape={channels.shape}")
    n_ch, n_samples = channels.shape
    if n_samples < window_size:
        return (
            np.empty((0, n_ch, window_size), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    # Shape from sliding_window_view: (n_ch, n_samples-window+1, window)
    sw = np.lib.stride_tricks.sliding_window_view(channels, window_shape=window_size, axis=1)
    sw = sw[:, ::step_size, :]  # downsample windows by step size
    windows = np.transpose(sw, (1, 0, 2)).astype(np.float32, copy=True)
    starts = np.arange(0, (n_samples - window_size + 1), step_size, dtype=np.int32)
    return windows, starts


def build_pulse_train(
    num_samples: int,
    sample_rate: float,
    annotations: list[dict],
    num_mus: int,
    pulse_width_ms: float = 5.0,
) -> np.ndarray:
    """Build dense binary target array shaped ``(num_samples, num_mus)``.

    The resulting target matrix is sequence-aligned so it can be used with
    sequence heads that emit per-time logits. Each `MUP` annotation is
    converted into a short block of ones centered at its annotated time.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0.")
    if num_mus <= 0:
        raise ValueError("num_mus must be > 0.")

    fs = max(float(sample_rate), 1.0)
    pulse_width = max(1, int(round((pulse_width_ms / 1000.0) * fs)))
    half_width = pulse_width // 2
    target = np.zeros((int(num_samples), int(num_mus)), dtype=np.float32)

    for ann in annotations:
        label = str(ann.get("label", "")).replace("AI:", "").strip().upper()
        if label != "MUP":
            continue

        mu_id = str(ann.get("mu_id", "")).strip().upper()
        if not mu_id.startswith("MU"):
            continue
        try:
            mu_idx = int(mu_id[2:]) - 1
        except ValueError:
            continue
        if mu_idx < 0 or mu_idx >= num_mus:
            continue

        start_t = float(ann.get("start_time", 0.0))
        end_t = float(ann.get("end_time", start_t))
        center_t = 0.5 * (start_t + end_t)
        center_idx = int(round(center_t * fs))
        lo = max(0, center_idx - half_width)
        hi = min(num_samples, center_idx + half_width + 1)
        if hi > lo:
            target[lo:hi, mu_idx] = 1.0
    return target


class EMGWindowDataset(Dataset):
    """Sliding-window dataset for sequence-to-sequence MUAP detection.

    Parameters
    ----------
    signals:
        Continuous EMG array shaped ``(channels, time)``.
    labels:
        Dense binary target array shaped ``(time, num_mus)``.
    window_size:
        Number of time samples per training item.
    step_size:
        Hop length between windows.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        step_size: int,
    ):
        super().__init__()
        x = np.asarray(signals, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("signals must have shape (channels, time)")
        if y.ndim != 2:
            raise ValueError("labels must have shape (time, num_mus)")
        if x.shape[1] != y.shape[0]:
            raise ValueError("signals and labels must share the same time length")

        self.signals = x
        self.labels = y
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        last_start = x.shape[1] - self.window_size
        if last_start < 0:
            self.starts = np.empty((0,), dtype=np.int32)
        else:
            self.starts = np.arange(0, last_start + 1, self.step_size, dtype=np.int32)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])
        stop = start + self.window_size

        # Slice continuous channels -> (C, W), then transpose for GRU input:
        # final tensor shape = (W, C)
        x_win = self.signals[:, start:stop].T.copy()

        # Slice dense pulse labels -> (W, num_mus)
        y_win = self.labels[start:stop].copy()
        return torch.from_numpy(x_win), torch.from_numpy(y_win)


def _pulse_dict_to_matrix(
    pulse_trains: dict[str, np.ndarray],
    n_samples: int,
    n_mu: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert pulse index dictionary into binary pulse matrix (n_mu, n_samples)."""
    keys = sorted(pulse_trains.keys())
    if len(keys) == 0:
        raise ValueError("pulse_trains is empty.")

    if n_mu is None:
        n_mu = len(keys)
    if n_mu <= 0:
        raise ValueError("n_mu must be > 0.")

    selected = keys[:n_mu]
    mat = np.zeros((n_mu, n_samples), dtype=np.uint8)
    for i, k in enumerate(selected):
        idx = np.asarray(pulse_trains[k], dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_samples)]
        mat[i, idx] = 1
    return mat, selected


def label_windows_center_span(
    pulse_matrix: np.ndarray,
    starts: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    center_span: int = DEFAULT_CENTER_LABEL_SPAN,
) -> np.ndarray:
    """Build window-wise multi-label targets from center span logic.

    Window target for a given MU is 1 if any pulse appears in the central span.
    """
    if pulse_matrix.ndim != 2:
        raise ValueError("pulse_matrix must be (n_mu, n_samples)")
    if starts.ndim != 1:
        raise ValueError("starts must be 1-D")
    if center_span <= 0 or center_span > window_size:
        raise ValueError("center_span must be within (0, window_size]")

    n_mu, n_samples = pulse_matrix.shape
    n_win = starts.shape[0]
    out = np.zeros((n_win, n_mu), dtype=np.float32)

    center0 = (window_size - center_span) // 2
    center1 = center0 + center_span

    for w, s in enumerate(starts):
        lo = int(s + center0)
        hi = int(min(s + center1, n_samples))
        if hi <= lo:
            continue
        center_slice = pulse_matrix[:, lo:hi]
        out[w] = (np.sum(center_slice, axis=1) > 0).astype(np.float32)
    return out


class HDSemgWindowDataset(Dataset):
    """Windowed multi-label HD-sEMG dataset for MU pulse detection."""

    def __init__(
        self,
        channels: np.ndarray,
        pulse_trains: dict[str, np.ndarray],
        *,
        n_mu: int | None = None,
        normalize: bool = True,
        interface: V2DataInterfaceSpec | None = None,
    ):
        super().__init__()
        self.interface = interface or V2DataInterfaceSpec()
        x = channels.astype(np.float32, copy=False)
        if x.ndim != 2:
            raise ValueError("channels must have shape (n_channels, n_samples)")
        self.n_channels = int(x.shape[0])
        self.n_samples = int(x.shape[1])

        self.x_cont = mapminmax_normalize(x) if normalize else x.copy()
        self.windows, self.starts = frame_signal(
            self.x_cont,
            window_size=self.interface.window_size,
            step_size=self.interface.step_size,
        )
        pulse_matrix, mu_keys = _pulse_dict_to_matrix(pulse_trains, self.n_samples, n_mu=n_mu)
        self.y = label_windows_center_span(
            pulse_matrix,
            self.starts,
            window_size=self.interface.window_size,
            center_span=self.interface.center_span,
        )
        self.mu_keys = mu_keys
        self.n_mu = len(mu_keys)

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx])        # (C, 120)
        y = torch.from_numpy(self.y[idx])              # (n_mu,)
        return x, y


# ------------------------------ Model -----------------------------------

class ExternalAttention(nn.Module):
    """External attention with fixed memory complexity.

    Input tensor shape:
    - ``x``: (batch, time, d_model)

    Internal steps:
    1. ``mk`` projects from ``d_model`` -> ``memory_size``
    2. softmax across time dimension so each memory slot attends over time
    3. L1 normalization across memory dimension for numerical stability
    4. ``mv`` projects back from ``memory_size`` -> ``d_model``
    """

    def __init__(self, d_model: int, memory_size: int = 64):
        super().__init__()
        self.mk = nn.Linear(d_model, memory_size, bias=False)
        self.mv = nn.Linear(memory_size, d_model, bias=False)
        nn.init.orthogonal_(self.mk.weight)
        nn.init.orthogonal_(self.mv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        attn = self.mk(x)  # (B, T, S)
        attn = torch.softmax(attn, dim=1)
        attn = attn / torch.clamp(attn.sum(dim=2, keepdim=True), min=1e-6)
        out = self.mv(attn)  # (B, T, D)
        return out


class BiGRUEA(nn.Module):
    """Backbone that extracts sequence features from multi-channel EMG.

    Expected input:
    - ``x``: (batch, time, channels)

    Output:
    - feature map: (batch, time, hidden_size * 2)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        proj_size: int = 64,
        attn_memory: int = 64,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.out_dim = self.hidden_size * 2

        self.proj = nn.Sequential(
            nn.Linear(self.input_channels, proj_size),
            nn.LayerNorm(proj_size),
            nn.ReLU(),
        )
        self.bigru = nn.GRU(
            input_size=proj_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = ExternalAttention(d_model=self.out_dim, memory_size=attn_memory)

        # Compatibility aliases for existing app-side state inspection.
        self.input_proj = self.proj[0]
        self.bigru_out = self.bigru
        self.ea = self.attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x_proj = self.proj(x)         # (B, T, P)
        gru_out, _ = self.bigru(x_proj)  # (B, T, 2H)
        attn_out = self.attn(gru_out)    # (B, T, 2H)
        return gru_out + attn_out        # residual feature map


class SOGRUEA(nn.Module):
    """Single-output head for generalized MUP detection.

    Output tensor shape:
    - logits: (batch, time, 1)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        attn_memory: int = 64,
    ):
        super().__init__()
        self.backbone = BiGRUEA(
            input_channels=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_memory=attn_memory,
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, self.backbone.out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.backbone.out_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        feat = self.backbone(x)  # (B, T, 2H)
        return self.head(feat)   # (B, T, 1)


class MOGRUEA(nn.Module):
    """Multi-output head for clustered MU tracking.

    Output tensor shape:
    - logits: (batch, time, num_mus)
    """

    def __init__(
        self,
        input_channels: int,
        num_mus: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        attn_memory: int = 64,
    ):
        super().__init__()
        self.num_mus = int(num_mus)
        self.backbone = BiGRUEA(
            input_channels=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_memory=attn_memory,
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, self.backbone.out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.backbone.out_dim // 2, self.num_mus),
        )

        # Compatibility aliases used by the current app.
        self.input_proj = self.backbone.input_proj
        self.bigru = self.backbone.bigru
        self.fc_out = self.head[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        feat = self.backbone(x)  # (B, T, 2H)
        return self.head(feat)   # (B, T, M)


class MOGRUEANet(nn.Module):
    """Compatibility wrapper used by the current app controller/workers.

    The new sequence model emits per-time logits, but the current app still
    expects window-level logits shaped ``(batch, num_mus)`` from input
    ``(batch, channels, time)``. This wrapper adapts the new backbone/head
    architecture to the existing app training and inference paths.
    """

    def __init__(
        self,
        input_channels: int | None = None,
        num_mus: int | None = None,
        *,
        in_channels: int | None = None,
        n_mu: int | None = None,
        hidden_size: int = 128,
        gru_layers: int = 2,
        num_layers: int | None = None,
        attn_memory: int = 64,
    ):
        super().__init__()
        resolved_in_channels = input_channels if input_channels is not None else in_channels
        resolved_num_mus = num_mus if num_mus is not None else n_mu
        if resolved_in_channels is None:
            raise ValueError("input_channels/in_channels must be provided")
        if resolved_num_mus is None:
            raise ValueError("num_mus/n_mu must be provided")

        self.in_channels = int(resolved_in_channels)
        self.n_mu = int(resolved_num_mus)
        layers = int(num_layers if num_layers is not None else gru_layers)

        if self.n_mu <= 1:
            self.sequence_model: nn.Module = SOGRUEA(
                input_channels=self.in_channels,
                hidden_size=hidden_size,
                num_layers=layers,
                attn_memory=attn_memory,
            )
        else:
            self.sequence_model = MOGRUEA(
                input_channels=self.in_channels,
                num_mus=self.n_mu,
                hidden_size=hidden_size,
                num_layers=layers,
                attn_memory=attn_memory,
            )

        # Compatibility aliases expected elsewhere in the app.
        if isinstance(self.sequence_model, MOGRUEA):
            self.input_proj = self.sequence_model.input_proj
            self.bigru = self.sequence_model.bigru
            self.fc_out = self.sequence_model.fc_out
        else:
            self.input_proj = self.sequence_model.backbone.input_proj
            self.bigru = self.sequence_model.backbone.bigru
            self.fc_out = self.sequence_model.head[-1]

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-time logits.

        Parameters
        ----------
        x:
            Input shaped either ``(B, C, T)`` or ``(B, T, C)``.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3-D tensor, got shape={tuple(x.shape)}")

        # Current app sends (B, C, T). Sequence heads expect (B, T, C).
        if x.shape[1] == self.in_channels:
            x = x.transpose(1, 2)
        elif x.shape[2] != self.in_channels:
            raise ValueError(
                f"Expected channel dimension {self.in_channels}, got shape={tuple(x.shape)}"
            )
        return self.sequence_model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_logits = self.forward_sequence(x)  # (B, T, M) or (B, T, 1)
        # Pool over time to maintain app compatibility with window-level BCE.
        pooled = torch.amax(seq_logits, dim=1)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(-1)
        return pooled

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


# --------------------------- Training Utilities --------------------------

def multilabel_f1_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Macro-F1 for multi-label predictions."""
    y_pred = (y_prob >= threshold).astype(np.uint8)
    n_labels = y_true.shape[1]
    f1s: list[float] = []
    for j in range(n_labels):
        t = y_true[:, j]
        p = y_pred[:, j]
        tp = float(np.sum((t == 1) & (p == 1)))
        fp = float(np.sum((t == 0) & (p == 1)))
        fn = float(np.sum((t == 1) & (p == 0)))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(np.asarray(f1s, dtype=np.float64)))


def train_model_v2(
    model: MOGRUEANet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: V2TrainingConfig,
    device: torch.device | str = "cpu",
) -> dict:
    """Train MO-GRU-EA with BCE and early stopping."""
    device = torch.device(device)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.full((model.n_mu,), float(cfg.positive_class_weight), device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            tr_loss += float(loss.item())
            tr_n += 1
        tr_loss = tr_loss / max(tr_n, 1)

        model.eval()
        va_loss = 0.0
        va_n = 0
        all_prob: list[np.ndarray] = []
        all_true: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.float32)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += float(loss.item())
                va_n += 1
                prob = torch.sigmoid(logits).cpu().numpy()
                all_prob.append(prob)
                all_true.append(yb.cpu().numpy())
        va_loss = va_loss / max(va_n, 1)

        if all_true:
            y_true = np.concatenate(all_true, axis=0)
            y_prob = np.concatenate(all_prob, axis=0)
            va_f1 = multilabel_f1_score(y_true, y_prob, threshold=cfg.threshold)
        else:
            va_f1 = 0.0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_f1"].append(va_f1)

        improved = va_loss < best_val
        if improved:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            torch.save(best_state, cfg.save_path)
        else:
            bad_epochs += 1

        # Early stop conditions:
        # (a) reached target loss, or (b) validation got worse for patience window.
        if va_loss <= cfg.early_stop_target_loss:
            break
        if bad_epochs >= cfg.early_stop_patience:
            break

        print(
            f"[V2] Epoch {epoch:03d}/{cfg.epochs} | "
            f"train={tr_loss:.6f} val={va_loss:.6f} f1={va_f1:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_val_loss": best_val,
        "best_path": cfg.save_path,
    }


def build_loaders_from_dataset(
    ds: Dataset,
    cfg: V2TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Split dataset and build train/validation loaders."""
    n = len(ds)
    if n < 4:
        raise ValueError(f"Need at least 4 windows for train/val split, got {n}.")

    val_len = max(1, int(n * cfg.val_split))
    train_len = n - val_len
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, val_len])
    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg.batch_size, len(train_ds)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(cfg.batch_size, len(val_ds)),
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


# ----------------------------- Inference ---------------------------------

@torch.no_grad()
def infer_single_window(
    model: MOGRUEANet,
    window: np.ndarray,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Run one-window inference.

    Parameters
    ----------
    window : np.ndarray
        Shape: (channels, 120)
    """
    if window.ndim != 2 or window.shape[1] != DEFAULT_WINDOW_SIZE:
        raise ValueError(
            f"window must be (channels, {DEFAULT_WINDOW_SIZE}), got {window.shape}"
        )

    device = torch.device(device)
    model = model.to(device)
    model.eval()
    x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(device)  # (1, C, 120)
    probs = torch.sigmoid(model(x))[0].cpu().numpy()
    return probs.astype(np.float32)


@torch.no_grad()
def benchmark_single_window_latency(
    model: MOGRUEANet,
    n_channels: int,
    *,
    device: torch.device | str = "cpu",
    warmup_runs: int = 30,
    timed_runs: int = 300,
    target_ms: float = 30.0,
) -> dict:
    """Benchmark mean/p95 latency for one (C,120) forward pass."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    x = torch.randn(1, n_channels, DEFAULT_WINDOW_SIZE, dtype=torch.float32, device=device)

    # Warm-up
    for _ in range(max(1, warmup_runs)):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(max(1, timed_runs)):
        t0 = perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    p95_ms = float(np.percentile(arr, 95))
    return {
        "device": device.type,
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "target_ms": float(target_ms),
        "passes_target": bool(mean_ms < target_ms),
    }


# --------------------------- Example Runner ------------------------------

def _example_usage() -> None:
    """Minimal runnable example with synthetic data."""
    rng = np.random.default_rng(7)
    n_ch = 64
    n_samples = 30_000
    n_mu = 6
    channels = 0.03 * rng.standard_normal((n_ch, n_samples)).astype(np.float32)
    pulse_trains = {}
    for i in range(n_mu):
        idx = np.sort(rng.choice(n_samples, size=180, replace=False)).astype(np.int32)
        pulse_trains[f"MU_{i+1}"] = idx

    ds = HDSemgWindowDataset(channels, pulse_trains, n_mu=n_mu)
    cfg = V2TrainingConfig(epochs=3, batch_size=128, save_path="model_weights_v2.pth")
    tr_loader, va_loader = build_loaders_from_dataset(ds, cfg)

    model = MOGRUEANet(in_channels=n_ch, n_mu=ds.n_mu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = train_model_v2(model, tr_loader, va_loader, cfg, device=device)
    print("Training done:", out["best_val_loss"])
    print(benchmark_single_window_latency(model, n_channels=n_ch, device=device))


if __name__ == "__main__":
    _example_usage()
