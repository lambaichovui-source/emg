# Repository Map: IONM EMG AI Assistant

## 1. Overview
The IONM (Intraoperative Neuromonitoring) EMG AI Assistant is a PyQt6-based desktop application designed for processing, analyzing, and annotating electromyography (EMG) signals. It utilizes multiple background threads to handle heavy computations, such as signal filtering, mathematical feature extraction, and AI model inference, ensuring the user interface remains highly responsive.

## 2. Core Components & Relationships

### `main.py` - Application Entry Point & Controller
- **Role**: Initializes the application and contains the `AppController` class.
- **Responsibilities**:
  - Bridges the UI (`MainWindow`) with background workers (`worker_threads.py`).
  - Wires PyQt signals from the UI to corresponding background tasks.
  - Handles plot updates and manages state (e.g., loaded data, currently selected model, CSV metadata).

### `ui_mainwindow.py` - Graphical User Interface
- **Role**: Defines the modular QDockWidget-based UI layout using PyQt6 and `pyqtgraph`.
- **Responsibilities**:
  - Renders the central plot area for multi-channel EMG visualization.
  - Manages interactive annotations (`LabeledRegionItem` and `AnnotatableViewBox`).
  - Contains docks for controls (loading CSVs, filter settings, model selection) and dashboards (metrics, live decision rules).
  - Emits custom signals (e.g., `filter_settings_changed`, `auto_fit_requested`) that are connected to the `AppController` in `main.py`.

### `worker_threads.py` - Background Processing
- **Role**: Provides `QThread` subclasses to offload blocking tasks from the main GUI thread.
- **Key Threads**:
  - `DataLoadingThread`: Loads CSV data via `DataHandler` and extracts initial features.
  - `RefilterThread`: Re-applies signal filters dynamically when user settings change.
  - `AITrainingThreadV2`: Trains the sequence model (`ai_model_v2.py`).
  - `GoldStandardStageThread`: Runs the deterministic evaluation pipeline (`gold_standard_pipeline.py`).
  - `NeurotonicInferenceThread`: Runs inference using the V2 AI model to auto-annotate neurotonic events.

### `data_handler.py` - Data Ingestion & Signal Processing
- **Role**: Handles CSV parsing, memory-efficient data structures, and DSP operations.
- **Responsibilities**:
  - `DataHandler`: Parses custom CSV formats and handles metadata chunking.
  - `SignalPreprocessor`: Applies IIR filters (Notch, Band-pass), baseline corrections, and artifact blanking (SEP, Cautery).
  - Contains Numba-compiled (`@njit`) mathematical operations for ultra-fast feature extraction (e.g., RMS, ZCR, Symmetry Index, Envelope Decrescendo).
  - Provides a thread-safe `RingBuffer` for continuous data flow in inference.

### AI Models & Logic
- **`ai_model_v2.py`**: Contains the `MOGRUEANet` architecture (Multi-Objective GRU Evolutionary Algorithm Network) and PyTorch Dataset definitions for the V2 AI pipeline. It frames signals and extracts MUP predictions.
- **`neurotonic_logic.py`**: A deterministic stage-2 classifier. It takes timestamp peaks (extracted by the neural net or FASTICA) and applies clinical thresholds to classify sequences as Spikes, Bursts, or Trains (A, B, C).
- **`gold_standard_pipeline.py` & `gold_standard_model.py`**: Implements the 5-step mathematical evaluation pipeline (Artifact Blanking -> Energy Gate -> MUP Detect -> Clustering -> Romstöck Classification) used as a baseline standard for evaluating the ML model.

### `annotation_store.py`
- **Role**: Handles reading and writing of standardized JSON and CSV annotations.
- **Responsibilities**: Serializes AI and human-created regions for model training and record keeping.

## 3. UI Hookup & Signal Flow

1. **User Interaction**: The user clicks "Load CSV" in the UI (`ui_mainwindow.py`).
2. **Controller Trigger**: The `btn_load_csv.clicked` signal fires, handled by `AppController._on_load_csv` (`main.py`).
3. **Thread Execution**: `AppController` spawns a `DataLoadingThread` (`worker_threads.py`) passing the file path.
4. **Data Ingestion**: `DataLoadingThread` uses `DataHandler` (`data_handler.py`) to chunk read and preprocess the CSV.
5. **Progress Updates**: As chunks are processed, the thread emits `progress_updated` and `chunk_ready` signals.
6. **UI Update**: `AppController` catches these signals to incrementally update the `pyqtgraph` plots and the progress bar without freezing the main window.
7. **Completion**: The thread emits `loading_finished`, passing the complete `time` and `channels` arrays back to the controller for final rendering.

This pattern is identical for filtering (`RefilterThread`), AI training (`AITrainingThreadV2`), and Auto-Annotation (`NeurotonicInferenceThread`).

## 4. Summary of Data Pipeline
1. **Raw CSV** -> `DataHandler` -> **Raw Chunks**
2. **Raw Chunks** -> `SignalPreprocessor` (Notch, High-pass, Median Filters) -> **Cleaned Chunks**
3. **Cleaned Chunks** -> Numba Feature Extractor -> **Romstöck Features**
4. **Cleaned Chunks** -> Inference Model (`MOGRUEANet`) -> **Pulse Trains**
5. **Pulse Trains** -> `NeurotonicClassifier` -> **Clinical Labels** (e.g., "A-Train (High Risk)") -> UI Rendered Regions.