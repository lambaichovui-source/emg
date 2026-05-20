"""
ui_mainwindow.py

Modular QDockWidget-based UI for the HITL IONM workstation.
This file contains only GUI layout and interaction scaffolding.
"""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import json
import os
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QGraphicsProxyWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QStyle,
    QSlider,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

NUM_CHANNELS = 16
TIMEBASE_OPTIONS = [5, 10, 20, 40]
SCROLL_SCALE = 1000  # ticks per second for time scrollbar
CLINICAL_Y_MIN = -100.0
CLINICAL_Y_MAX = 100.0

LABEL_CHOICES = (
    "MUP",
    "Artifact",
    "Spike",
    "Burst",
    "A-Train (High Risk)",
    "B-Train",
    "C-Train (Polyrhythmic)",
)
LABEL_COLORS = {
    "MUP": (64, 200, 64, 70),
    "Artifact": (255, 140, 0, 80),
    "Spike": (120, 200, 255, 75),
    "Burst": (255, 220, 80, 75),
    "A-Train (High Risk)": (255, 64, 64, 85),
    "B-Train": (180, 120, 255, 75),
    "C-Train (Polyrhythmic)": (255, 120, 200, 75),
}

CHANNEL_COLORS_DARK = [
    "#ff6b6b", "#ffd93d", "#6bff95", "#7aa2ff",
    "#b98cff", "#66e0ff", "#ff9f66", "#9eff66",
    "#ff66d9", "#fca5a5", "#67e8f9", "#fcd34d",
    "#86efac", "#c4b5fd", "#f9a8d4", "#ffffff",
]

CHANNEL_COLORS_LIGHT = [
    "#b91c1c", "#a16207", "#166534", "#1d4ed8",
    "#7e22ce", "#0e7490", "#c2410c", "#3f6212",
    "#be185d", "#991b1b", "#155e75", "#854d0e",
    "#14532d", "#6d28d9", "#9d174d", "#111827",
]


class AnnotatableViewBox(pg.ViewBox):
    """Shift + left drag creates a region on the owning channel."""

    region_requested = pyqtSignal(int, float, float)
    scroll_requested = pyqtSignal(float)

    def __init__(self, channel_idx: int, **kwargs):
        super().__init__(**kwargs)
        self._channel_idx = channel_idx
        self._drag_start_x: float | None = None
        self._temp_region: pg.LinearRegionItem | None = None

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.MouseButton.LeftButton and (
            ev.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            ev.accept()
            x_now = self.mapSceneToView(ev.scenePos()).x()
            if ev.isStart():
                self._drag_start_x = x_now
                self._temp_region = pg.LinearRegionItem(
                    values=(x_now, x_now),
                    brush=pg.mkBrush(255, 255, 255, 20),
                    movable=False,
                )
                self.addItem(self._temp_region)
            elif ev.isFinish() and self._drag_start_x is not None:
                lo, hi = sorted((self._drag_start_x, x_now))
                if self._temp_region is not None:
                    self.removeItem(self._temp_region)
                    self._temp_region = None
                if hi - lo > 0.001:
                    self.region_requested.emit(self._channel_idx, lo, hi)
                self._drag_start_x = None
            elif self._temp_region is not None and self._drag_start_x is not None:
                lo, hi = sorted((self._drag_start_x, x_now))
                self._temp_region.setRegion((lo, hi))
            return

        # Lock default pan/zoom drags in the plot area.
        ev.accept()

    def wheelEvent(self, ev, axis=None):
        # Wheel performs horizontal scrolling only.
        direction = -1.0 if ev.delta() > 0 else 1.0
        self.scroll_requested.emit(direction)
        ev.accept()


class LabeledRegionItem(pg.LinearRegionItem):
    """Region item with label context menu and selection callbacks."""

    delete_requested = pyqtSignal(object)
    label_changed = pyqtSignal(object, str)
    selected_requested = pyqtSignal(object)

    promoted = pyqtSignal(object)

    def __init__(
        self,
        channel_idx: int,
        values: tuple[float, float],
        label: str,
        ai_generated: bool = False,
    ):
        super().__init__(values=values)
        self.channel_idx = channel_idx
        self.label = label
        self.ai_generated = ai_generated
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton
        )
        self._apply_style()

    def set_label(self, label: str) -> None:
        self.label = label
        self._apply_style()
        self.label_changed.emit(self, label)

    def _apply_style(self) -> None:
        clean_label = self.label.replace("AI:", "").strip()
        rgba = LABEL_COLORS.get(clean_label, (130, 130, 130, 70))
        alpha = 60 if self.ai_generated else max(rgba[3], 120)
        self.setBrush(pg.mkBrush(*rgba))
        self.setBrush(pg.mkBrush(rgba[0], rgba[1], rgba[2], alpha))
        self.setHoverBrush(pg.mkBrush(rgba[0], rgba[1], rgba[2], min(255, alpha + 40)))
        pen = pg.mkPen(rgba[0], rgba[1], rgba[2], 180, width=2)
        hover = pg.mkPen(rgba[0], rgba[1], rgba[2], 255, width=3)
        for line in self.lines:
            line.setPen(pen)
            line.setHoverPen(hover)

    def promote_to_human(self) -> None:
        self.label = self.label.replace("AI:", "").strip()
        self.ai_generated = False
        self._apply_style()
        self.promoted.emit(self)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(ev)
            ev.accept()
            return
        self.selected_requested.emit(self)
        super().mouseClickEvent(ev)

    def _show_context_menu(self, ev) -> None:
        menu = QMenu()
        for label in LABEL_CHOICES:
            act = menu.addAction(f"Label: {label}")
            act.triggered.connect(lambda _checked=False, x=label: self.set_label(x))
        if self.ai_generated:
            menu.addSeparator()
            menu.addAction("Promote [✓]").triggered.connect(self.promote_to_human)
        menu.addSeparator()
        del_act = menu.addAction("Delete Region")
        del_act.triggered.connect(lambda: self.delete_requested.emit(self))
        scene = self.scene()
        if scene and scene.views():
            view = scene.views()[0]
            pos = view.mapToGlobal(view.mapFromScene(ev.scenePos()))
            menu.exec(pos)


class MainWindow(QMainWindow):
    """Dock-based main workstation window."""

    annotation_selected = pyqtSignal(object)
    filter_settings_changed = pyqtSignal()
    trace_overlap_changed = pyqtSignal()
    auto_fit_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IONM HITL Workstation")
        self.setMinimumSize(1600, 950)

        self._is_dark = True
        self._timebase_sec = 10.0
        self._x_min = 0.0
        self._x_max = 10.0
        self._y_min = CLINICAL_Y_MIN
        self._y_max = CLINICAL_Y_MAX
        self._scroll_updating = False

        self._channel_checkboxes: list[QCheckBox] = []
        self._plot_items: list[pg.PlotItem] = []
        self._plot_curves: list[pg.PlotDataItem] = []
        self._annotations: list[LabeledRegionItem] = []
        self._annotation_text: dict[LabeledRegionItem, pg.TextItem] = {}
        self._annotation_controls: dict[LabeledRegionItem, QGraphicsProxyWidget] = {}
        self._annotation_row: dict[LabeledRegionItem, int] = {}
        self._row_region: dict[int, LabeledRegionItem] = {}
        self._annotation_metrics: dict[LabeledRegionItem, tuple[float, float, float]] = {}
        self._micro_burst_items: list[tuple[int, pg.LinearRegionItem]] = []
        self._channel_names: list[str] = [f"Ch {i+1}" for i in range(NUM_CHANNELS)]
        self._channel_name_items: list[pg.TextItem] = []
        self._overlap_enabled = False
        self._overlap_prev_curves: list[pg.PlotDataItem] = []
        self._overlap_next_curves: list[pg.PlotDataItem] = []

        self._build_ui()
        self._apply_theme(True)

    # ---------- UI skeleton ----------
    def _build_ui(self) -> None:
        self._build_central_plot_area()
        self._build_left_dock()
        self._build_right_dock()
        self._build_bottom_dock()
        self._build_mup_criteria_dock()
        self._build_status_bar()
        self._show_window_at(0.0)
        # Defer settings restore until after the window is fully painted
        # to avoid QPainter "engine == 0" warnings from _apply_theme.
        QTimer.singleShot(0, self._load_settings)

    def _build_central_plot_area(self) -> None:
        pg.setConfigOptions(antialias=False, useOpenGL=False)
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(4)

        # Container that holds the plot widget + zoom overlay
        plot_container = QWidget()
        plot_container.setObjectName("plotContainer")
        plot_container_layout = QVBoxLayout(plot_container)
        plot_container_layout.setContentsMargins(0, 0, 0, 0)
        plot_container_layout.setSpacing(0)

        self.plot_widget = pg.GraphicsLayoutWidget(plot_container)
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_widget.ci.layout.setSpacing(1)
        plot_container_layout.addWidget(self.plot_widget)

        # Zoom overlay – floated top-right inside plot_container
        self._zoom_overlay = QWidget(plot_container)
        self._zoom_overlay.setObjectName("zoomOverlay")
        self._zoom_overlay.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        zoom_row = QHBoxLayout(self._zoom_overlay)
        zoom_row.setContentsMargins(6, 6, 6, 6)
        zoom_row.setSpacing(4)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_out = QPushButton("−")
        for btn in (self.btn_zoom_in, self.btn_zoom_out):
            btn.setFixedSize(32, 32)
            btn.setObjectName("zoomBtn")
            btn.setStyleSheet(
                "QPushButton#zoomBtn{"
                "background:rgba(40,40,40,200);"
                "color:#fff;"
                "border:1px solid #888;"
                "border-radius:4px;"
                "font-size:14px;"
                "font-weight:bold;"
                "}"
                "QPushButton#zoomBtn:hover{background:rgba(70,70,70,220);}"
                "QPushButton#zoomBtn:pressed{background:rgba(20,20,20,240);}"
            )
        self.btn_zoom_in.clicked.connect(self._on_zoom_in)
        self.btn_zoom_out.clicked.connect(self._on_zoom_out)
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.btn_zoom_out)
        self._zoom_overlay.adjustSize()
        self._zoom_overlay.raise_()

        # Position the overlay once the container is laid out
        plot_container.resizeEvent = self._on_plot_container_resize

        central_layout.addWidget(plot_container, 1)

        self.time_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.setMaximum(int(self._timebase_sec * SCROLL_SCALE))
        self.time_scrollbar.setPageStep(int(self._timebase_sec * SCROLL_SCALE))
        self.time_scrollbar.valueChanged.connect(self._on_time_scrollbar_changed)
        central_layout.addWidget(self.time_scrollbar, 0)
        self.setCentralWidget(central)

        first_plot: pg.PlotItem | None = None
        for i in range(NUM_CHANNELS):
            vb = AnnotatableViewBox(i)
            vb.region_requested.connect(self._on_region_requested)
            vb.scroll_requested.connect(self._on_plot_wheel_scroll)
            plot = self.plot_widget.addPlot(row=i, col=0, viewBox=vb)
            plot.setLabel("left", f"Ch{i+1}", size="8pt")
            plot.getAxis("left").setWidth(45)
            plot.setMouseEnabled(x=False, y=False)
            vb.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            plot.setClipToView(True)
            plot.setDownsampling(auto=True, mode="peak")
            plot.showGrid(x=True, y=False, alpha=0.2)
            plot.setYRange(self._y_min, self._y_max, padding=0.0)
            # Hide Y-axis tick marks and numbers; keep only the axis label.
            left_ax = plot.getAxis("left")
            left_ax.setStyle(showValues=False)
            left_ax.setTicks([])

            if i == 0:
                first_plot = plot
            else:
                plot.setXLink(first_plot)

            if i < NUM_CHANNELS - 1:
                plot.hideAxis("bottom")

            curve = plot.plot(pen=pg.mkPen("#FFFFFF", width=1))
            prev_curve = plot.plot(pen=pg.mkPen(180, 180, 180, 60, width=1))
            next_curve = plot.plot(pen=pg.mkPen(180, 180, 180, 60, width=1))
            prev_curve.setVisible(False)
            next_curve.setVisible(False)
            self._plot_items.append(plot)
            self._plot_curves.append(curve)
            self._overlap_prev_curves.append(prev_curve)
            self._overlap_next_curves.append(next_curve)
            # Channel name is shown only on the left axis label (no floating text).

        self._apply_clinical_grid()

    def _build_left_dock(self) -> None:
        dock = QDockWidget("Control & Artifact Module", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        body = QWidget()
        layout = QVBoxLayout(body)

        self.btn_load_csv = QPushButton("Load CSV")
        layout.addWidget(self.btn_load_csv)
        self.btn_batch_train_folder = QPushButton("Batch Train Folder")
        layout.addWidget(self.btn_batch_train_folder)

        layout.addWidget(self._build_channel_group())

        view_group = QGroupBox("View Controls")
        vlay = QVBoxLayout(view_group)
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(1, 100)
        self.gain_slider.setValue(10)
        self.gain_label = QLabel("Gain: 10")
        self.gain_slider.valueChanged.connect(lambda v: self.gain_label.setText(f"Gain: {v}"))
        vlay.addWidget(self.gain_label)
        vlay.addWidget(self.gain_slider)

        y_row = QHBoxLayout()
        self.input_y_min = QLineEdit(str(int(self._y_min)))
        self.input_y_max = QLineEdit(str(int(self._y_max)))
        self.input_y_min.setMaximumWidth(80)
        self.input_y_max.setMaximumWidth(80)
        self.input_y_min.editingFinished.connect(self._on_y_range_input_changed)
        self.input_y_max.editingFinished.connect(self._on_y_range_input_changed)
        y_row.addWidget(QLabel("Y Min"))
        y_row.addWidget(self.input_y_min)
        y_row.addWidget(QLabel("Y Max"))
        y_row.addWidget(self.input_y_max)
        vlay.addLayout(y_row)

        self.btn_auto_fit_y = QPushButton("Auto Fit Y")
        self.btn_auto_fit_y.clicked.connect(self.auto_fit_requested.emit)
        vlay.addWidget(self.btn_auto_fit_y)

        self.combo_timebase = QComboBox()
        for s in TIMEBASE_OPTIONS:
            self.combo_timebase.addItem(f"{s} sec", s)
        self.combo_timebase.setCurrentIndex(1)
        self.combo_timebase.currentIndexChanged.connect(self._on_timebase_changed)
        vlay.addWidget(QLabel("Timebase"))
        vlay.addWidget(self.combo_timebase)

        self.btn_theme = QPushButton("Switch to Light Mode")
        self.btn_theme.clicked.connect(self._toggle_theme)
        vlay.addWidget(self.btn_theme)
        self.chk_trace_overlap = QCheckBox("Trace Overlap Between Channels")
        self.chk_trace_overlap.setChecked(False)
        self.chk_trace_overlap.toggled.connect(self._on_trace_overlap_toggled)
        vlay.addWidget(self.chk_trace_overlap)
        layout.addWidget(view_group)

        filt_group = QGroupBox("Artifact Reduction")
        flay = QVBoxLayout(filt_group)

        # ── Show Raw (bypass all) ──────────────────────────────────
        self.chk_show_raw = QCheckBox("Show Raw Data (bypass all filters)")
        self.chk_show_raw.toggled.connect(lambda _: self.filter_settings_changed.emit())
        flay.addWidget(self.chk_show_raw)

        # ── Notch + Band-pass (existing) ─────────────────────────
        self.chk_notch = QCheckBox("Notch Filter")
        self.chk_notch.setChecked(True)
        self.chk_notch.toggled.connect(lambda _: self.filter_settings_changed.emit())
        flay.addWidget(self.chk_notch)

        notch_row = QHBoxLayout()
        notch_row.addWidget(QLabel("  Notch (Hz)"))
        self.combo_notch = QComboBox()
        self.combo_notch.addItem("60", 60.0)
        self.combo_notch.addItem("120", 120.0)
        self.combo_notch.currentIndexChanged.connect(lambda _: self.filter_settings_changed.emit())
        notch_row.addWidget(self.combo_notch)
        flay.addLayout(notch_row)

        cut_row = QHBoxLayout()
        cut_row.addWidget(QLabel("  Low-cut"))
        self.input_low_cut = QLineEdit("20")
        self.input_low_cut.setMaximumWidth(60)
        self.input_low_cut.editingFinished.connect(self.filter_settings_changed.emit)
        cut_row.addWidget(self.input_low_cut)
        cut_row.addWidget(QLabel("High-cut"))
        self.input_high_cut = QLineEdit("500")
        self.input_high_cut.setMaximumWidth(60)
        self.input_high_cut.editingFinished.connect(self.filter_settings_changed.emit)
        cut_row.addWidget(self.input_high_cut)
        flay.addLayout(cut_row)

        flay.addSpacing(6)

        # ── Algorithm 1: Baseline Correction (IIR High-Pass) ─────────
        self.chk_baseline = QCheckBox("IIR High-Pass: removes DC drift & baseline wander")
        self.chk_baseline.setChecked(False)
        self.chk_baseline.toggled.connect(lambda _: self.filter_settings_changed.emit())
        flay.addWidget(self.chk_baseline)

        baseline_row = QHBoxLayout()
        baseline_row.addWidget(QLabel("  Cutoff (Hz)"))
        self.input_baseline_cutoff = QLineEdit("15")
        self.input_baseline_cutoff.setMaximumWidth(60)
        self.input_baseline_cutoff.editingFinished.connect(self.filter_settings_changed.emit)
        baseline_row.addWidget(self.input_baseline_cutoff)
        baseline_row.addStretch()
        flay.addLayout(baseline_row)

        flay.addSpacing(6)

        # ── Algorithm 2: SEP Artifact Blanking (Median + Trigger) ───
        self.chk_sep = QCheckBox("Median Filter: flattens ultra-fast spikes")
        self.chk_sep.setChecked(True)
        self.chk_sep.toggled.connect(lambda _: self.filter_settings_changed.emit())
        flay.addWidget(self.chk_sep)

        sep_row = QHBoxLayout()
        sep_row.addWidget(QLabel("  Blank (ms)"))
        self.input_sep_blank_ms = QLineEdit("3.0")
        self.input_sep_blank_ms.setMaximumWidth(60)
        self.input_sep_blank_ms.editingFinished.connect(self.filter_settings_changed.emit)
        sep_row.addWidget(self.input_sep_blank_ms)
        self.chk_sep_trigger = QCheckBox("Trigger")
        self.chk_sep_trigger.setToolTip(
            "If checked, force output = 0 for 'Blank ms' at each SEP trigger onset."
        )
        self.chk_sep_trigger.toggled.connect(lambda _: self.filter_settings_changed.emit())
        sep_row.addWidget(self.chk_sep_trigger)
        flay.addLayout(sep_row)

        flay.addSpacing(6)

        # ── Algorithm 3: Cautery Blanking (Amplitude-Hold) ──────────
        self.chk_cautery = QCheckBox("Amplitude-Hold: reject high amplitude event")
        self.chk_cautery.setChecked(True)
        self.chk_cautery.toggled.connect(lambda _: self.filter_settings_changed.emit())
        flay.addWidget(self.chk_cautery)

        cautery_row1 = QHBoxLayout()
        cautery_row1.addWidget(QLabel("  Threshold (µV)"))
        self.input_cautery_threshold = QLineEdit("2000")
        self.input_cautery_threshold.setMaximumWidth(70)
        self.input_cautery_threshold.editingFinished.connect(self.filter_settings_changed.emit)
        cautery_row1.addWidget(self.input_cautery_threshold)
        cautery_row1.addStretch()
        flay.addLayout(cautery_row1)

        cautery_row2 = QHBoxLayout()
        cautery_row2.addWidget(QLabel("  Hold (ms)"))
        self.input_cautery_hold_ms = QLineEdit("150")
        self.input_cautery_hold_ms.setMaximumWidth(60)
        self.input_cautery_hold_ms.editingFinished.connect(self.filter_settings_changed.emit)
        cautery_row2.addWidget(self.input_cautery_hold_ms)
        cautery_row2.addStretch()
        flay.addLayout(cautery_row2)

        layout.addWidget(filt_group)
        layout.addStretch()

        # ── Auto-save on every control change ─────────────────────────
        self.filter_settings_changed.connect(self._save_settings)
        self.gain_slider.valueChanged.connect(lambda _: self._save_settings())
        self.combo_timebase.currentIndexChanged.connect(lambda _: self._save_settings())
        self.input_y_min.editingFinished.connect(self._save_settings)
        self.input_y_max.editingFinished.connect(self._save_settings)
        self.chk_trace_overlap.toggled.connect(lambda _: self._save_settings())

        dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.left_dock = dock

    def _build_right_dock(self) -> None:
        dock = QDockWidget("Annotation & Analysis Tool", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        body = QWidget()
        layout = QVBoxLayout(body)

        self.tbl_annotations = QTableWidget(0, 5)
        self.tbl_annotations.setHorizontalHeaderLabels(
            ["Channel", "Name", "Duration", "Amplitude", "Frequency"]
        )
        self.tbl_annotations.cellClicked.connect(self._on_annotation_row_clicked)
        layout.addWidget(QLabel("Annotation List"))
        layout.addWidget(self.tbl_annotations)
        self.btn_load_annotations = QPushButton("Load Annotations (JSON)")
        layout.addWidget(self.btn_load_annotations)
        self.btn_save = QPushButton("Save Annotations (JSON)")
        layout.addWidget(self.btn_save)
        self.btn_export_training_csv = QPushButton("Export Training CSV")
        layout.addWidget(self.btn_export_training_csv)
        self.btn_load_training_csv = QPushButton("Load Training CSV")
        layout.addWidget(self.btn_load_training_csv)

        panel = QGroupBox("Analysis Dashboard")
        form = QFormLayout(panel)
        self.lbl_rms = QLabel("-")
        self.lbl_zcr = QLabel("-")
        self.lbl_symmetry = QLabel("-")
        self.lbl_decrescendo = QLabel("-")
        self.lbl_micro_bursts = QLabel("-")
        form.addRow("RMS", self.lbl_rms)
        form.addRow("ZCR", self.lbl_zcr)
        form.addRow("Symmetry Index", self.lbl_symmetry)
        form.addRow("Decrescendo", self.lbl_decrescendo)
        form.addRow("Micro-bursts", self.lbl_micro_bursts)
        layout.addWidget(panel)
        layout.addStretch()

        dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.right_dock = dock

    def _build_bottom_dock(self) -> None:
        dock = QDockWidget("Co-Training & Monitoring Dashboard", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        body = QWidget()
        layout = QVBoxLayout(body)

        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model Selection"))
        self.combo_model_select = QComboBox()
        self.combo_model_select.addItem("MO-GRU-EA (v2)", "v2")
        self.combo_model_select.setEnabled(False)
        self.combo_model_select.currentIndexChanged.connect(lambda _: self._save_settings())
        row_model.addWidget(self.combo_model_select, 1)
        layout.addLayout(row_model)

        row = QHBoxLayout()
        self.btn_train = QPushButton("Start AI Training")
        self.btn_inference = QPushButton("Run Inference")
        self.btn_auto_annotate_test = QPushButton("Auto-Annotate (Test)")
        self.btn_gold_standard = QPushButton("Run All Steps (1→5)")
        self.btn_create_backup = QPushButton("Create Model Backup")
        self.btn_restore_backup = QPushButton("Restore Previous Model")
        self.btn_create_backup.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        self.btn_restore_backup.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack)
        )
        row.addWidget(self.btn_train)
        row.addWidget(self.btn_inference)
        layout.addLayout(row)

        row_auto = QHBoxLayout()
        row_auto.addWidget(self.btn_auto_annotate_test)
        row_auto.addWidget(self.btn_gold_standard)
        layout.addLayout(row_auto)

        gold_box = QGroupBox("Gold Standard — run one step at a time")
        gold_lay = QVBoxLayout(gold_box)
        self.chk_gold_visible_window = QCheckBox(
            "Only analyze visible time window (much faster)"
        )
        self.chk_gold_visible_window.setChecked(True)
        gold_lay.addWidget(self.chk_gold_visible_window)
        self.lbl_gold_cache_status = QLabel("Cache: no steps run yet")
        self.lbl_gold_cache_status.setWordWrap(True)
        gold_lay.addWidget(self.lbl_gold_cache_status)
        row_g1 = QHBoxLayout()
        self.btn_gold_step1 = QPushButton("Step 1")
        self.btn_gold_step2 = QPushButton("Step 2")
        self.btn_gold_step3 = QPushButton("Step 3")
        row_g1.addWidget(self.btn_gold_step1)
        row_g1.addWidget(self.btn_gold_step2)
        row_g1.addWidget(self.btn_gold_step3)
        gold_lay.addLayout(row_g1)
        row_g2 = QHBoxLayout()
        self.btn_gold_step4 = QPushButton("Step 4")
        self.btn_gold_step5 = QPushButton("Step 5")
        row_g2.addWidget(self.btn_gold_step4)
        row_g2.addWidget(self.btn_gold_step5)
        gold_lay.addLayout(row_g2)
        for i, btn in enumerate(
            (
                self.btn_gold_step1,
                self.btn_gold_step2,
                self.btn_gold_step3,
                self.btn_gold_step4,
                self.btn_gold_step5,
            ),
            start=1,
        ):
            btn.setToolTip(
                {
                    1: "Artifact blanking",
                    2: "Energy-linked noise gate (TKEO)",
                    3: "MUP pulse detection",
                    4: "Train clustering + features",
                    5: "Romstöck classify + plot regions",
                }[i]
            )
        layout.addWidget(gold_box)

        row_backup = QHBoxLayout()
        row_backup.addWidget(self.btn_create_backup)
        row_backup.addWidget(self.btn_restore_backup)
        layout.addLayout(row_backup)

        self.lbl_backup_status = QLabel("Last Backup: No Backup Found")
        layout.addWidget(self.lbl_backup_status)

        self.slider_confidence = QSlider(Qt.Orientation.Horizontal)
        self.slider_confidence.setRange(0, 100)
        self.slider_confidence.setValue(50)
        self.lbl_conf = QLabel("Confidence Threshold: 50%")
        self.slider_confidence.valueChanged.connect(self._on_confidence_slider_changed)
        layout.addWidget(self.lbl_conf)
        layout.addWidget(self.slider_confidence)
        self._on_confidence_slider_changed(self.slider_confidence.value())

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.lbl_training_progress = QLabel("Training Progress: -")
        layout.addWidget(self.lbl_training_progress)

        mon = QHBoxLayout()
        self.lbl_ram = QLabel("System RAM Usage: -")
        self.lbl_loss = QLabel("Current Model Loss: -")
        mon.addWidget(self.lbl_ram)
        mon.addWidget(self.lbl_loss)
        layout.addLayout(mon)

        mon2 = QHBoxLayout()
        self.lbl_val_loss = QLabel("Validation Loss: -")
        self.lbl_f1 = QLabel("F1-Score: -")
        mon2.addWidget(self.lbl_val_loss)
        mon2.addWidget(self.lbl_f1)
        layout.addLayout(mon2)

        dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.splitDockWidget(self.right_dock, dock, Qt.Orientation.Vertical)
        self.bottom_dock = dock

    def _build_mup_criteria_dock(self) -> None:
        """Self-explanatory panel for MUP decision criteria + live metrics."""
        dock = QDockWidget("MUP Criteria Monitor", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        body = QWidget()
        layout = QVBoxLayout(body)

        criteria_box = QGroupBox("Decision Logic (Human Readable)")
        cform = QFormLayout(criteria_box)
        self.lbl_mup_window = QLabel("Window: 120 samples")
        self.lbl_mup_step = QLabel("Step: 20 samples")
        self.lbl_mup_sample_rate = QLabel("Sample Rate: 1280 Hz")
        self.lbl_mup_threshold = QLabel("Probability Threshold: 0.50")
        self.lbl_mup_rule = QLabel(
            "Gold Standard: 5 steps (prep → gate → MUP → cluster → classify). Run only what you need."
        )
        self.lbl_mup_rule.setWordWrap(True)
        cform.addRow("Frame Size", self.lbl_mup_window)
        cform.addRow("Frame Step", self.lbl_mup_step)
        cform.addRow("Sampling", self.lbl_mup_sample_rate)
        cform.addRow("Threshold", self.lbl_mup_threshold)
        cform.addRow("Decision", self.lbl_mup_rule)
        layout.addWidget(criteria_box)

        live_box = QGroupBox("Live Model Criteria")
        lform = QFormLayout(live_box)
        self.lbl_mup_mode = QLabel("Mode: Idle")
        self.lbl_mup_train_loss = QLabel("Train Loss: -")
        self.lbl_mup_val_loss = QLabel("Val Loss: -")
        self.lbl_mup_f1 = QLabel("F1: -")
        self.lbl_mup_confidence = QLabel("Mean Confidence: -")
        self.lbl_mup_latency = QLabel("Latency / Window: -")
        self.lbl_mup_decision_status = QLabel("MUP Decision: -")
        self.lbl_neurotonic_status = QLabel("Neurotonic Train: None")
        self.lbl_neurotonic_conf = QLabel("Neurotonic Confidence: -")
        lform.addRow("Pipeline", self.lbl_mup_mode)
        lform.addRow("Training BCE/CE", self.lbl_mup_train_loss)
        lform.addRow("Validation", self.lbl_mup_val_loss)
        lform.addRow("Quality", self.lbl_mup_f1)
        lform.addRow("Inference", self.lbl_mup_confidence)
        lform.addRow("Speed", self.lbl_mup_latency)
        lform.addRow("Current Status", self.lbl_mup_decision_status)
        lform.addRow("Neurotonic", self.lbl_neurotonic_status)
        lform.addRow("Neurotonic Conf", self.lbl_neurotonic_conf)
        layout.addWidget(live_box)
        layout.addStretch()

        dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.splitDockWidget(self.bottom_dock, dock, Qt.Orientation.Vertical)
        self.criteria_dock = dock
        # Sync criteria threshold with current inference confidence slider.
        if hasattr(self, "slider_confidence"):
            self.set_mup_threshold_text(float(self.slider_confidence.value()) / 100.0)

    def _build_channel_group(self) -> QGroupBox:
        group = QGroupBox("Channels")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        box = QWidget()
        v = QVBoxLayout(box)
        for i in range(NUM_CHANNELS):
            cb = QCheckBox(f"Ch {i+1}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_channel_toggled)
            self._channel_checkboxes.append(cb)
            v.addWidget(cb)
        scroll.setWidget(box)
        lay = QVBoxLayout(group)
        lay.addWidget(scroll)
        return group

    def _build_status_bar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_label = QLabel("System Ready")
        sb.addWidget(self.status_label, 1)

    # ---------- Persistent Settings ----------

    _SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

    def _save_settings(self) -> None:
        """Write all UI control values to settings.json."""
        try:
            data = {
                # Theme
                "dark_mode": self._is_dark,
                # View controls
                "gain": self.gain_slider.value(),
                "y_min": self.input_y_min.text(),
                "y_max": self.input_y_max.text(),
                "timebase_idx": self.combo_timebase.currentIndex(),
                "trace_overlap": self.chk_trace_overlap.isChecked(),
                "model_select_idx": self.combo_model_select.currentIndex(),
                # Artifact Reduction
                "show_raw": self.chk_show_raw.isChecked(),
                "enable_notch": self.chk_notch.isChecked(),
                "notch_idx": self.combo_notch.currentIndex(),
                "low_cut": self.input_low_cut.text(),
                "high_cut": self.input_high_cut.text(),
                "enable_baseline": self.chk_baseline.isChecked(),
                "baseline_cutoff": self.input_baseline_cutoff.text(),
                "enable_sep": self.chk_sep.isChecked(),
                "sep_blank_ms": self.input_sep_blank_ms.text(),
                "sep_trigger": self.chk_sep_trigger.isChecked(),
                "enable_cautery": self.chk_cautery.isChecked(),
                "cautery_threshold": self.input_cautery_threshold.text(),
                "cautery_hold_ms": self.input_cautery_hold_ms.text(),
            }
            with open(self._SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Never crash the UI due to settings save failure

    def _load_settings(self) -> None:
        """Restore UI control values from settings.json if it exists."""
        if not os.path.exists(self._SETTINGS_PATH):
            return
        try:
            with open(self._SETTINGS_PATH, "r", encoding="utf-8") as f:
                data: dict = json.load(f)
        except Exception:
            return

        def _tb(widget, val: bool) -> None:
            widget.blockSignals(True); widget.setChecked(bool(val)); widget.blockSignals(False)
        def _ti(widget, val: str) -> None:
            widget.blockSignals(True); widget.setText(str(val)); widget.blockSignals(False)
        def _tc(widget, idx: int) -> None:
            widget.blockSignals(True); widget.setCurrentIndex(int(idx)); widget.blockSignals(False)
        def _ts(widget, val: int) -> None:
            widget.blockSignals(True); widget.setValue(int(val)); widget.blockSignals(False)

        if "dark_mode" in data:
            self._apply_theme(bool(data["dark_mode"]))
        if "gain" in data:
            _ts(self.gain_slider, data["gain"])
            self.gain_label.setText(f"Gain: {self.gain_slider.value()}")
        if "y_min" in data:
            _ti(self.input_y_min, data["y_min"])
        if "y_max" in data:
            _ti(self.input_y_max, data["y_max"])
        if "timebase_idx" in data:
            _tc(self.combo_timebase, data["timebase_idx"])
            self._timebase_sec = float(self.combo_timebase.currentData())
        if "trace_overlap" in data:
            _tb(self.chk_trace_overlap, data["trace_overlap"])
        if "model_select_idx" in data:
            _tc(self.combo_model_select, data["model_select_idx"])
        # Artifact Reduction
        if "show_raw" in data:
            _tb(self.chk_show_raw, data["show_raw"])
        if "enable_notch" in data:
            _tb(self.chk_notch, data["enable_notch"])
        if "notch_idx" in data:
            _tc(self.combo_notch, data["notch_idx"])
        if "low_cut" in data:
            _ti(self.input_low_cut, data["low_cut"])
        if "high_cut" in data:
            _ti(self.input_high_cut, data["high_cut"])
        if "enable_baseline" in data:
            _tb(self.chk_baseline, data["enable_baseline"])
        if "baseline_cutoff" in data:
            _ti(self.input_baseline_cutoff, data["baseline_cutoff"])
        if "enable_sep" in data:
            _tb(self.chk_sep, data["enable_sep"])
        if "sep_blank_ms" in data:
            _ti(self.input_sep_blank_ms, data["sep_blank_ms"])
        if "sep_trigger" in data:
            _tb(self.chk_sep_trigger, data["sep_trigger"])
        if "enable_cautery" in data:
            _tb(self.chk_cautery, data["enable_cautery"])
        if "cautery_threshold" in data:
            _ti(self.input_cautery_threshold, data["cautery_threshold"])
        if "cautery_hold_ms" in data:
            _ti(self.input_cautery_hold_ms, data["cautery_hold_ms"])

    # ---------- Theme ----------
    def _apply_theme(self, dark: bool) -> None:
        self._is_dark = dark
        if dark:
            self.setStyleSheet("QMainWindow,QWidget{background:#1d1d1d;color:#e8e8e8;} QGroupBox{border:1px solid #555; margin-top:8px;} QGroupBox::title{subcontrol-origin:margin;left:8px;} QPushButton{background:#2e2e2e;border:1px solid #666;padding:6px;} QDockWidget::title{background:#2b2b2b;padding:4px;} QTableWidget{background:#121212;gridline-color:#333;}")
            pg.setConfigOption("background", "#000000")
            pg.setConfigOption("foreground", "#d0d0d0")
            colors = CHANNEL_COLORS_DARK
            self.btn_theme.setText("Switch to Light Mode")
        else:
            self.setStyleSheet("QMainWindow,QWidget{background:#f3f3f3;color:#111;} QGroupBox{border:1px solid #aaa; margin-top:8px;} QGroupBox::title{subcontrol-origin:margin;left:8px;} QPushButton{background:#ffffff;border:1px solid #999;padding:6px;} QDockWidget::title{background:#e8e8e8;padding:4px;} QTableWidget{background:#ffffff;gridline-color:#bbb;}")
            pg.setConfigOption("background", "#ffffff")
            pg.setConfigOption("foreground", "#202020")
            colors = CHANNEL_COLORS_LIGHT
            self.btn_theme.setText("Switch to Dark Mode")

        self.plot_widget.setBackground(pg.getConfigOption("background"))
        for i, curve in enumerate(self._plot_curves):
            curve.setPen(pg.mkPen(colors[i % len(colors)], width=1))
            self._plot_items[i].getAxis("left").setTextPen(colors[i % len(colors)])

    def _toggle_theme(self) -> None:
        self._apply_theme(not self._is_dark)
        self._save_settings()

    # ---------- Plot/timebase helpers ----------
    def _apply_clinical_grid(self) -> None:
        # Fixed 10 major divisions across the visible timebase window.
        step = self._timebase_sec / 10.0
        for p in self._plot_items:
            p.showGrid(x=True, y=False, alpha=0.2)
            axis = p.getAxis("bottom")
            axis.setTickSpacing(major=step, minor=step / 2.0)
        self._update_bottom_minute_timeline()

    # ---------- Division-based navigation core ----------

    def _division_sec(self) -> float:
        """Return the width of one division (1/10 of the current timebase)."""
        return self._timebase_sec / 10.0

    def _show_window_at(self, x0: float) -> None:
        """
        Set the view so it starts at x0 and spans exactly _timebase_sec.
        Clamps so the window never exceeds the data extent.
        """
        width = self._timebase_sec
        if self._x_max > self._x_min:
            x0 = max(self._x_min, min(self._x_max - width, x0))
        else:
            x0 = self._x_min
        x1 = x0 + width
        for p in self._plot_items:
            p.setXRange(x0, x1, padding=0.0)
        self._apply_clinical_grid()
        self._update_time_scrollbar()

    def _step_by_divisions(self, divisions: int) -> None:
        """Move the view left (negative) or right (positive) by N divisions."""
        x0, _x1 = self.get_visible_time_range()
        self._show_window_at(x0 + divisions * self._division_sec())

    def _on_timebase_changed(self) -> None:
        """Re-anchor view at the current left edge with the new timebase."""
        self._timebase_sec = float(self.combo_timebase.currentData())
        self._apply_clinical_grid()
        x0, _x1 = self.get_visible_time_range()
        self._show_window_at(x0)

    def get_visible_time_range(self) -> tuple[float, float]:
        if not self._plot_items:
            return 0.0, self._timebase_sec
        x0, x1 = self._plot_items[0].vb.viewRange()[0]
        return float(x0), float(x1)

    def set_time_extent(self, xmin: float, xmax: float) -> None:
        """Called by the controller when new data is loaded."""
        self._x_min = float(xmin)
        self._x_max = float(xmax)
        self._show_window_at(self._x_min)

    # Keep pan_to_timestamp for external callers (e.g. annotation click navigation)
    def pan_to_timestamp(self, ts: float, keep_center: bool = False) -> None:
        """Scroll the view so ts is visible. If keep_center, center on ts."""
        if keep_center:
            x0 = ts - self._timebase_sec / 2.0
        else:
            x0 = ts - self._timebase_sec / 4.0
        self._show_window_at(x0)

    def _update_bottom_minute_timeline(self) -> None:
        if not self._plot_items:
            return
        bottom_axis = self._plot_items[-1].getAxis("bottom")
        xmax = max(self._x_max, 0.0)
        ticks: list[tuple[float, str]] = []
        step_sec = 30.0  # 30-second increments
        t = step_sec
        while t <= xmax + 1e-9:
            total_sec = int(round(t))
            m, s = divmod(total_sec, 60)
            label = f"{m}:{s:02d}"
            ticks.append((t, label))
            t += step_sec
        if not ticks:
            ticks = [(0.0, "0:00")]
        bottom_axis.setTicks([ticks, []])
        bottom_axis.setLabel("Timeline (min)")

    def _position_channel_name_items(self) -> None:
        if not self._plot_items:
            return
        x0, x1 = self.get_visible_time_range()
        x = x0 + max(0.0, (x1 - x0) * 0.01)
        for i, txt in enumerate(self._channel_name_items):
            if i >= len(self._plot_items):
                break
            y_top = self._plot_items[i].vb.viewRange()[1][1]
            txt.setPos(x, y_top * 0.82 if y_top != 0 else 0.1)

    def _update_time_scrollbar(self) -> None:
        """Sync the scrollbar with the current view. Uses divisions as integer steps."""
        if not hasattr(self, "time_scrollbar"):
            return
        self._scroll_updating = True
        div = self._division_sec()
        total = max(0.0, self._x_max - self._x_min)
        x0, _x1 = self.get_visible_time_range()
        # Express everything in division units (integers)
        total_divs = max(10, round(total / div))  # total divisions in the data
        # Scrollbar max = total_divs - 10  (so the last full window fits)
        max_div = max(0, total_divs - 10)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.setMaximum(max_div)
        self.time_scrollbar.setPageStep(10)   # 1 full window = 10 divisions
        self.time_scrollbar.setSingleStep(1)  # 1 arrow click = 1 division
        current_div = round(max(0.0, x0 - self._x_min) / div)
        self.time_scrollbar.setValue(current_div)
        self._scroll_updating = False

    def _on_time_scrollbar_changed(self, value: int) -> None:
        """Scrollbar changed: snap the view to that division boundary."""
        if self._scroll_updating:
            return
        x0 = self._x_min + value * self._division_sec()
        self._show_window_at(x0)

    def _on_plot_container_resize(self, event) -> None:
        """Keep the zoom overlay anchored to the top-right of the plot container."""
        if event is not None:
            QWidget.resizeEvent(self._zoom_overlay.parent(), event)
        if hasattr(self, "_zoom_overlay"):
            overlay_w = self._zoom_overlay.sizeHint().width()
            overlay_h = self._zoom_overlay.sizeHint().height()
            parent = self._zoom_overlay.parent()
            if parent is not None:
                x = parent.width() - overlay_w - 8
                self._zoom_overlay.setGeometry(x, 8, overlay_w, overlay_h)

    def _on_zoom_in(self) -> None:
        """
        Zoom in: halve the timebase (show half the time range, twice the detail).
        The view re-anchors at the current left edge. Minimum timebase is 1 s.
        """
        x0, _x1 = self.get_visible_time_range()
        new_tb = max(1.0, self._timebase_sec / 2.0)
        if new_tb == self._timebase_sec:
            return
        self._timebase_sec = new_tb
        # Sync the combo box without triggering _on_timebase_changed again
        idx = self.combo_timebase.findData(new_tb)
        if idx >= 0:
            self.combo_timebase.blockSignals(True)
            self.combo_timebase.setCurrentIndex(idx)
            self.combo_timebase.blockSignals(False)
        self._apply_clinical_grid()
        self._show_window_at(x0)

    def _on_zoom_out(self) -> None:
        """
        Zoom out: double the timebase (show twice the time range, less detail).
        Capped so we cannot zoom beyond the full data extent.
        """
        x0, _x1 = self.get_visible_time_range()
        data_span = max(self._x_max - self._x_min, 1.0)
        new_tb = min(data_span, self._timebase_sec * 2.0)
        if new_tb == self._timebase_sec:
            return
        self._timebase_sec = new_tb
        idx = self.combo_timebase.findData(new_tb)
        if idx >= 0:
            self.combo_timebase.blockSignals(True)
            self.combo_timebase.setCurrentIndex(idx)
            self.combo_timebase.blockSignals(False)
        self._apply_clinical_grid()
        self._show_window_at(x0)

    def _on_plot_wheel_scroll(self, direction: float) -> None:
        """Scroll wheel: move exactly 1 division."""
        self._step_by_divisions(int(direction))

    def _on_confidence_slider_changed(self, value: int) -> None:
        self.lbl_conf.setText(f"Confidence Threshold: {value}%")
        if hasattr(self, "lbl_mup_threshold"):
            self.set_mup_threshold_text(float(value) / 100.0)

    # ---------- Public API for controller ----------
    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def show_progress(self, value: int, maximum: int = 100) -> None:
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        self.progress_bar.setVisible(value < maximum)

    def set_ram_text(self, text: str) -> None:
        self.lbl_ram.setText(text)

    def set_loss_text(self, text: str) -> None:
        self.lbl_loss.setText(text)

    def set_val_loss_text(self, text: str) -> None:
        self.lbl_val_loss.setText(text)

    def set_f1_text(self, text: str) -> None:
        self.lbl_f1.setText(text)

    def set_training_progress_text(self, text: str) -> None:
        self.lbl_training_progress.setText(text)

    def set_backup_status_text(self, text: str) -> None:
        self.lbl_backup_status.setText(text)

    def set_gold_cache_status(self, text: str) -> None:
        if hasattr(self, "lbl_gold_cache_status"):
            self.lbl_gold_cache_status.setText(text)

    def set_gold_step_buttons_enabled(self, enabled: bool) -> None:
        for btn in (
            getattr(self, "btn_gold_step1", None),
            getattr(self, "btn_gold_step2", None),
            getattr(self, "btn_gold_step3", None),
            getattr(self, "btn_gold_step4", None),
            getattr(self, "btn_gold_step5", None),
            getattr(self, "btn_gold_standard", None),
        ):
            if btn is not None:
                btn.setEnabled(enabled)

    def get_selected_model_key(self) -> str:
        return str(self.combo_model_select.currentData() or "v2")

    def set_mup_sampling_text(self, sample_rate_hz: float) -> None:
        if hasattr(self, "lbl_mup_sample_rate"):
            self.lbl_mup_sample_rate.setText(f"Sample Rate: {sample_rate_hz:.0f} Hz")

    def set_mup_threshold_text(self, threshold: float) -> None:
        if hasattr(self, "lbl_mup_threshold"):
            self.lbl_mup_threshold.setText(f"Probability Threshold: {threshold:.2f}")

    def set_mup_live_metrics(
        self,
        *,
        mode: str | None = None,
        train_loss: float | None = None,
        val_loss: float | None = None,
        f1_score: float | None = None,
        mean_confidence: float | None = None,
        latency_ms: float | None = None,
        decision_status: str | None = None,
    ) -> None:
        if not hasattr(self, "lbl_mup_mode"):
            return
        if mode is not None:
            self.lbl_mup_mode.setText(f"Mode: {mode}")
        if train_loss is not None:
            self.lbl_mup_train_loss.setText(f"Train Loss: {train_loss:.5f}")
        if val_loss is not None:
            self.lbl_mup_val_loss.setText(f"Val Loss: {val_loss:.5f}")
        if f1_score is not None:
            self.lbl_mup_f1.setText(f"F1: {f1_score:.4f}")
        if mean_confidence is not None:
            self.lbl_mup_confidence.setText(f"Mean Confidence: {mean_confidence:.3f}")
        if latency_ms is not None:
            self.lbl_mup_latency.setText(f"Latency / Window: {latency_ms:.2f} ms")
        if decision_status is not None:
            self.lbl_mup_decision_status.setText(f"MUP Decision: {decision_status}")

    def set_neurotonic_status(self, label: str, confidence: float, metrics: dict | None = None) -> None:
        if hasattr(self, "lbl_neurotonic_status"):
            ch = int(metrics.get("channel_idx", -1)) if metrics else -1
            ch_txt = f"Ch{ch + 1} " if ch >= 0 else ""
            self.lbl_neurotonic_status.setText(f"Neurotonic Train: {ch_txt}{label}")
        if hasattr(self, "lbl_neurotonic_conf"):
            self.lbl_neurotonic_conf.setText(f"Neurotonic Confidence: {confidence:.3f}")
        if metrics:
            freq = float(metrics.get("mean_frequency_hz", 0.0))
            amp = float(metrics.get("rolling_amplitude_uv", 0.0))
            self.set_mup_live_metrics(
                mode="Inference",
                mean_confidence=confidence,
                decision_status=f"Train={label} | f={freq:.1f}Hz amp={amp:.1f}uV",
            )

    def update_curve(self, channel: int, x_data, y_data) -> None:
        if 0 <= channel < NUM_CHANNELS:
            self._plot_curves[channel].setData(x_data, y_data)

    def update_overlap_traces(self, x_data, y_matrix) -> None:
        if y_matrix is None:
            return
        n = min(NUM_CHANNELS, int(y_matrix.shape[0]))
        for i in range(NUM_CHANNELS):
            prev_curve = self._overlap_prev_curves[i]
            next_curve = self._overlap_next_curves[i]
            show = self._overlap_enabled and i < n
            prev_curve.setVisible(show and i > 0)
            next_curve.setVisible(show and i < (n - 1))
            if show and i > 0:
                prev_curve.setData(x_data, y_matrix[i - 1])
            else:
                prev_curve.setData([], [])
            if show and i < (n - 1):
                next_curve.setData(x_data, y_matrix[i + 1])
            else:
                next_curve.setData([], [])

    def clear_all_curves(self) -> None:
        for c in self._plot_curves:
            c.setData([], [])
        for c in self._overlap_prev_curves:
            c.setData([], [])
        for c in self._overlap_next_curves:
            c.setData([], [])

    def set_active_channels(self, num_channels: int) -> None:
        for i in range(NUM_CHANNELS):
            active = i < num_channels
            self._channel_checkboxes[i].blockSignals(True)
            self._channel_checkboxes[i].setChecked(active)
            self._channel_checkboxes[i].setEnabled(active)
            self._channel_checkboxes[i].blockSignals(False)
            self._plot_items[i].setVisible(active)

    def set_channel_names(self, names: list[str]) -> None:
        for i in range(NUM_CHANNELS):
            label = names[i] if i < len(names) and names[i] else f"Ch {i+1}"
            self._channel_names[i] = label
            self._channel_checkboxes[i].setText(label)
            self._plot_items[i].setLabel("left", label, size="8pt")

    def set_gain_scale(self, gain_value: int) -> None:
        # Fixed Y range only applies when overlap is off.
        _ = gain_value
        if not self._overlap_enabled:
            for p in self._plot_items:
                p.setYRange(self._y_min, self._y_max, padding=0.0)

    def _on_y_range_input_changed(self) -> None:
        try:
            y_min = float(self.input_y_min.text().strip())
            y_max = float(self.input_y_max.text().strip())
            if y_min >= y_max:
                raise ValueError("Y min must be < Y max")
        except Exception:
            self.input_y_min.setText(f"{self._y_min:g}")
            self.input_y_max.setText(f"{self._y_max:g}")
            self.set_status("Invalid Y range input. Expected Y min < Y max.")
            return
        self.set_fixed_y_range(y_min, y_max)
        self.set_status(f"Y range updated to [{y_min:g}, {y_max:g}]")

    def set_fixed_y_range(self, y_min: float, y_max: float) -> None:
        self._y_min = float(y_min)
        self._y_max = float(y_max)
        self.input_y_min.setText(f"{self._y_min:g}")
        self.input_y_max.setText(f"{self._y_max:g}")
        if not self._overlap_enabled:
            for p in self._plot_items:
                p.setYRange(self._y_min, self._y_max, padding=0.0)

    def _on_trace_overlap_toggled(self, checked: bool) -> None:
        self._overlap_enabled = bool(checked)
        # Always use the fixed clinical Y range (auto-range removed).
        for p in self._plot_items:
            p.disableAutoRange(axis="y")
            p.setYRange(self._y_min, self._y_max, padding=0.0)
        self.trace_overlap_changed.emit()

    def set_analysis_values(self, rms: float, zcr: float, sym: float, dec: float) -> None:
        self.lbl_rms.setText(f"{rms:.6f}")
        self.lbl_zcr.setText(f"{zcr:.6f}")
        self.lbl_symmetry.setText(f"{sym:.6f}")
        self.lbl_decrescendo.setText(f"{dec:.6f}")

    def set_micro_burst_values(self, count: int, threshold: float) -> None:
        self.lbl_micro_bursts.setText(f"{count} (thr={threshold:.4f})")

    def apply_filter_metadata(self, meta: dict) -> None:
        low = float(meta.get("low_cut_hz", 20.0))
        high = float(meta.get("high_cut_hz", 500.0))
        notch = float(meta.get("notch_hz", 60.0))
        self.input_low_cut.setText(f"{low:g}")
        self.input_high_cut.setText(f"{high:g}")
        idx = self.combo_notch.findData(notch)
        if idx >= 0:
            self.combo_notch.setCurrentIndex(idx)

    def get_filter_settings(self) -> dict:
        def _safe_float(txt: str, default: float) -> float:
            try:
                return float(txt.strip())
            except Exception:
                return default

        return {
            # Raw / notch / band-pass
            "show_raw":         self.chk_show_raw.isChecked(),
            "enable_notch":     self.chk_notch.isChecked(),
            "notch_hz":         float(self.combo_notch.currentData() or 60.0),
            "low_cut_hz":       _safe_float(self.input_low_cut.text(), 20.0),
            "high_cut_hz":      _safe_float(self.input_high_cut.text(), 500.0),
            # Algorithm 1 – Baseline Correction (IIR High-Pass)
            "enable_baseline":      self.chk_baseline.isChecked(),
            "baseline_cutoff_hz":   _safe_float(self.input_baseline_cutoff.text(), 15.0),
            # Algorithm 2 – SEP Artifact Reduction
            "enable_sep":       self.chk_sep.isChecked(),
            "sep_blank_ms":     _safe_float(self.input_sep_blank_ms.text(), 3.0),
            "sep_trigger":      self.chk_sep_trigger.isChecked(),
            # Algorithm 3 – Cautery Blanking
            "enable_cautery":       self.chk_cautery.isChecked(),
            "cautery_threshold":    _safe_float(self.input_cautery_threshold.text(), 2000.0),
            "cautery_hold_ms":      _safe_float(self.input_cautery_hold_ms.text(), 150.0),
        }

    # ---------- Annotation API ----------
    def clear_annotations(self) -> None:
        self.clear_micro_burst_overlay()
        for region in list(self._annotations):
            self._remove_annotation(region)
        self.tbl_annotations.setRowCount(0)
        self._annotation_row.clear()
        self._row_region.clear()
        self._annotation_metrics.clear()

    def add_ai_annotation(
        self,
        channel_idx: int,
        start_time: float,
        end_time: float,
        label: str,
    ) -> None:
        if not label.startswith("AI:"):
            label = f"AI: {label}"
        self._create_region(channel_idx, start_time, end_time, label, ai_generated=True)

    def clear_ai_annotations(self) -> None:
        """Remove prior AI/gold-standard regions before adding a new batch."""
        for region in list(self._annotations):
            if region.ai_generated:
                self._remove_annotation(region)

    def add_ai_annotations(self, annotations: list[dict]) -> None:
        """Add many AI regions in one batched pass (avoids GUI freeze / crash)."""
        if not annotations:
            return
        self.clear_ai_annotations()
        self.setUpdatesEnabled(False)
        try:
            for ann in annotations:
                ch_idx = int(ann["channel_idx"])
                self.add_ai_annotation(
                    ch_idx,
                    float(ann["start_time"]),
                    float(ann["end_time"]),
                    str(ann["label"]),
                )
        finally:
            self.setUpdatesEnabled(True)
        self.set_status(f"Added {len(annotations)} gold-standard region(s).")

    def get_annotations_data(self) -> list[dict]:
        out: list[dict] = []
        for region in self._annotations:
            x0, x1 = region.getRegion()
            dur, amp, freq = self._annotation_metrics.get(region, (x1 - x0, 0.0, 0.0))
            clean_label = str(region.label).replace("AI:", "").strip()
            out.append(
                {
                    "channel_idx": int(region.channel_idx),
                    "channel": f"Ch{region.channel_idx+1}",
                    "start_time": round(float(x0), 6),
                    "end_time": round(float(x1), 6),
                    "label": clean_label,
                    "duration": round(float(dur), 6),
                    "amplitude": round(float(amp), 6),
                    "frequency": round(float(freq), 3),
                }
            )
        return sorted(out, key=lambda x: (x["channel"], x["start_time"]))

    def set_annotations_data(self, annotations: list[dict]) -> None:
        """Replace current regions with persisted annotations."""
        self.clear_annotations()
        for ann in annotations:
            ch_idx = int(ann.get("channel_idx", 0))
            ch_idx = max(0, min(NUM_CHANNELS - 1, ch_idx))
            x0 = float(ann.get("start_time", 0.0))
            x1 = float(ann.get("end_time", x0))
            label = str(ann.get("label", "MUP")).strip() or "MUP"
            region = self._create_region(ch_idx, x0, x1, label, ai_generated=False)
            duration = float(ann.get("duration", abs(x1 - x0)))
            amplitude = float(ann.get("amplitude", 0.0))
            frequency = float(ann.get("frequency", 0.0))
            self.update_annotation_metrics(region, duration, amplitude, frequency)

    def update_annotation_metrics(
        self, region: LabeledRegionItem, duration: float, amplitude: float, frequency: float
    ) -> None:
        self._annotation_metrics[region] = (duration, amplitude, frequency)
        row = self._annotation_row.get(region)
        if row is not None:
            self.tbl_annotations.setItem(row, 0, QTableWidgetItem(f"Ch{region.channel_idx + 1}"))
            self.tbl_annotations.setItem(row, 1, QTableWidgetItem(region.label))
            self.tbl_annotations.setItem(row, 2, QTableWidgetItem(f"{duration:.3f}s"))
            self.tbl_annotations.setItem(row, 3, QTableWidgetItem(f"{amplitude:.3f}"))
            self.tbl_annotations.setItem(row, 4, QTableWidgetItem(f"{frequency:.2f}Hz"))
        txt = self._annotation_text.get(region)
        if txt is not None:
            txt.setText(f"({region.label}, {duration:.3f}s, {amplitude:.3f}, {frequency:.2f}Hz)")

    # ---------- Event handlers ----------
    def _on_channel_toggled(self) -> None:
        for i, cb in enumerate(self._channel_checkboxes):
            self._plot_items[i].setVisible(cb.isChecked())

    def _on_region_requested(self, channel_idx: int, x0: float, x1: float) -> None:
        self._create_region(channel_idx, x0, x1, "MUP", ai_generated=False)

    def _create_region(
        self,
        channel_idx: int,
        x0: float,
        x1: float,
        label: str,
        ai_generated: bool,
    ) -> LabeledRegionItem:
        region = LabeledRegionItem(channel_idx, (x0, x1), label, ai_generated=ai_generated)
        region.delete_requested.connect(self._remove_annotation)
        region.label_changed.connect(self._on_region_label_changed)
        region.selected_requested.connect(self._on_region_selected)
        region.promoted.connect(self._on_region_promoted)
        region.sigRegionChanged.connect(lambda: self._on_region_geometry_changed(region))
        region.sigRegionChangeFinished.connect(
            lambda: self.annotation_selected.emit(region)
        )
        self._plot_items[channel_idx].addItem(region)
        self._annotations.append(region)

        y_top = self._plot_items[channel_idx].vb.viewRange()[1][1]
        txt = pg.TextItem("", color=(255, 255, 255))
        txt.setPos(x0, y_top * 0.9 if y_top != 0 else 0.1)
        self._plot_items[channel_idx].addItem(txt)
        self._annotation_text[region] = txt
        self._create_region_controls(region)

        row = self.tbl_annotations.rowCount()
        self.tbl_annotations.insertRow(row)
        self._annotation_row[region] = row
        self._row_region[row] = region
        self.update_annotation_metrics(region, abs(x1 - x0), 0.0, 0.0)
        self.set_status(f"Region created on Ch{channel_idx+1}. Right click to relabel.")
        self.annotation_selected.emit(region)
        return region

    def _on_region_geometry_changed(self, region: LabeledRegionItem) -> None:
        x0, x1 = region.getRegion()
        txt = self._annotation_text.get(region)
        if txt is not None:
            y_top = self._plot_items[region.channel_idx].vb.viewRange()[1][1]
            txt.setPos(min(x0, x1), y_top * 0.9 if y_top != 0 else 0.1)
        dur, amp, freq = self._annotation_metrics.get(region, (0.0, 0.0, 0.0))
        self.update_annotation_metrics(region, abs(x1 - x0), amp, freq)
        self._position_region_controls(region)

    def _on_region_label_changed(self, region: LabeledRegionItem, _label: str) -> None:
        dur, amp, freq = self._annotation_metrics.get(region, (0.0, 0.0, 0.0))
        self.update_annotation_metrics(region, dur, amp, freq)

    def _on_region_promoted(self, region: LabeledRegionItem) -> None:
        proxy = self._annotation_controls.get(region)
        if proxy is not None and proxy.widget() is not None:
            promote_btn = proxy.widget().findChild(QPushButton, "btnPromote")
            if promote_btn is not None:
                promote_btn.hide()
        self._on_region_label_changed(region, region.label)

    def _on_region_selected(self, region: LabeledRegionItem) -> None:
        row = self._annotation_row.get(region)
        if row is not None:
            self.tbl_annotations.selectRow(row)
        self.annotation_selected.emit(region)

    def clear_micro_burst_overlay(self) -> None:
        for ch_idx, item in self._micro_burst_items:
            self._plot_items[ch_idx].removeItem(item)
        self._micro_burst_items.clear()

    def show_micro_burst_overlay(self, channel_idx: int, spans: list[tuple[float, float]]) -> None:
        self.clear_micro_burst_overlay()
        for t0, t1 in spans:
            item = pg.LinearRegionItem(values=(float(t0), float(t1)), movable=False)
            item.setBrush(pg.mkBrush(0, 255, 255, 45))
            pen = pg.mkPen(0, 255, 255, 180, width=1)
            for line in item.lines:
                line.setPen(pen)
            self._plot_items[channel_idx].addItem(item)
            self._micro_burst_items.append((channel_idx, item))

    def _on_annotation_row_clicked(self, row: int, _col: int) -> None:
        region = self._row_region.get(row)
        if region is None:
            return
        x0, x1 = region.getRegion()
        self.pan_to_timestamp((x0 + x1) / 2.0)
        self.annotation_selected.emit(region)

    def _remove_annotation(self, region: LabeledRegionItem) -> None:
        if region not in self._annotations:
            return
        self._annotations.remove(region)
        self._plot_items[region.channel_idx].removeItem(region)
        txt = self._annotation_text.pop(region, None)
        if txt is not None:
            self._plot_items[region.channel_idx].removeItem(txt)
        row = self._annotation_row.pop(region, None)
        proxy = self._annotation_controls.pop(region, None)
        if proxy is not None:
            self._plot_items[region.channel_idx].removeItem(proxy)
        self._annotation_metrics.pop(region, None)
        if row is not None:
            self.tbl_annotations.removeRow(row)
            self._reindex_annotation_rows()
        self.set_status("Region deleted.")

    def _create_region_controls(self, region: LabeledRegionItem) -> None:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)

        btn_delete = QPushButton("X")
        btn_delete.setObjectName("btnDelete")
        btn_delete.setFixedSize(18, 18)
        btn_delete.clicked.connect(lambda: self._remove_annotation(region))
        row.addWidget(btn_delete)

        btn_promote = QPushButton("✓")
        btn_promote.setObjectName("btnPromote")
        btn_promote.setFixedSize(18, 18)
        btn_promote.clicked.connect(region.promote_to_human)
        btn_promote.setVisible(region.ai_generated)
        row.addWidget(btn_promote)

        proxy = QGraphicsProxyWidget()
        proxy.setWidget(container)
        self._plot_items[region.channel_idx].addItem(proxy)
        self._annotation_controls[region] = proxy
        self._position_region_controls(region)

    def _position_region_controls(self, region: LabeledRegionItem) -> None:
        proxy = self._annotation_controls.get(region)
        if proxy is None:
            return
        x0, x1 = region.getRegion()
        x = min(float(x0), float(x1))
        y_top = self._plot_items[region.channel_idx].vb.viewRange()[1][1]
        proxy.setPos(x, y_top * 0.96 if y_top != 0 else 0.1)

    def _reindex_annotation_rows(self) -> None:
        self._annotation_row.clear()
        self._row_region.clear()
        for r in range(self.tbl_annotations.rowCount()):
            if r < len(self._annotations):
                region = self._annotations[r]
                self._annotation_row[region] = r
                self._row_region[r] = region

    # ---------- Keyboard navigation ----------
    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Up:
            self.gain_slider.setValue(min(100, self.gain_slider.value() + 1))
            event.accept()
            return
        if event.key() == Qt.Key.Key_Down:
            self.gain_slider.setValue(max(1, self.gain_slider.value() - 1))
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            # Move exactly 1 division (timebase / 10) per arrow press.
            n = -1 if event.key() == Qt.Key.Key_Left else 1
            self._step_by_divisions(n)
            event.accept()
            return
        super().keyPressEvent(event)
