# Dataset Metrics and Normalization Analysis

This document summarizes the analysis of the raw EMG datasets in the `testing dataset` folder, providing core metrics, thresholds, and normalization strategies for processing physiological signals.

## 1. File Summaries

### File: `Raw1minDrill.csv`
- **Duration**: 59.52 seconds
- **Sample Rate**: 1280.0 Hz
- **Channels**: 8
- **Total Samples**: 76,184
- **Global Amplitude Range**: [-532.57 µV, 550.41 µV]
- **Mean Baseline Noise (MAD)**: 4.03 µV
- **Mean RMS**: 47.72 µV
- **Estimated Global SNR**: 21.47 dB

### File: `Raw1minStim.csv`
- **Duration**: 60.00 seconds
- **Sample Rate**: 1280.0 Hz
- **Channels**: 8
- **Total Samples**: 76,800
- **Global Amplitude Range**: [-534.27 µV, 544.86 µV]
- **Mean Baseline Noise (MAD)**: 2.09 µV
- **Mean RMS**: 40.86 µV
- **Estimated Global SNR**: 25.83 dB

### File: `Raw1minNoevent.csv`
- **Duration**: 60.00 seconds
- **Sample Rate**: 1280.0 Hz
- **Channels**: 8
- **Total Samples**: 76,800
- **Global Amplitude Range**: [-75.43 µV, 20.76 µV]
- **Mean Baseline Noise (MAD)**: 1.51 µV
- **Mean RMS**: 1.71 µV
- **Estimated Global SNR**: 1.07 dB

### File: `Raw5minEDC.csv`
- **Duration**: 299.56 seconds
- **Sample Rate**: 1280.0 Hz
- **Channels**: 8
- **Total Samples**: 383,440
- **Global Amplitude Range**: [-529.99 µV, 541.73 µV]
- **Mean Baseline Noise (MAD)**: 1.57 µV
- **Mean RMS**: 15.78 µV
- **Estimated Global SNR**: 20.04 dB

## 2. Core Metrics & Physiological Normalization

### A. Baseline and Noise Estimation
Based on the data analysis, the baseline noise level (measured via Median Absolute Deviation - MAD) fluctuates depending on the recording. However, a safe assumption for physiological rest state across channels is the calculated average MAD.
- **Threshold Strategy**: To detect active signals over the noise floor, a dynamic threshold of `Threshold = Median + (K * MAD)` should be used, where `K` is typically 3 to 5.
- **Energy-Linked Noise Gate**: As per methodology, apply TKEO (Teager-Kaiser Energy Operator) to highlight the high-frequency/high-amplitude MUPs and calculate MAD on the TKEO signal for the most robust noise gating.

### B. Signal Amplitude & Normalization
The amplitude range varies dramatically (e.g., from small baseline noise to large MUP spikes and artifacts). Raw values are in microvolts (µV).
- **Clipping & Artifact Blanking**: Extreme values (e.g., beyond ±2000 µV or ±5000 µV depending on the cautery threshold) are usually non-physiological artifacts and should be blanked.
- **Normalization Strategy**: When feeding data into AI models (like the MOGRUEANet), use `mapminmax` normalization (scaling to [-1, 1]) per window or z-score normalization (`(x - mean) / std`) to ensure magnitude invariance for morphology analysis.

### C. Windowing for Feature Extraction
For continuous analysis and real-time processing, the data must be chunked into overlapping windows.
- **Typical Window Size**: 256 samples (which corresponds to exactly 0.2 seconds at 1280 Hz) provides a good balance between time resolution and frequency analysis.
- **Step Size**: 128 samples (50% overlap) ensures continuity and captures events spanning window boundaries.
- **MUP Detection Window**: For fine-grained Motor Unit Potential (MUP) guardrails, extract micro-windows of 10-15ms around the TKEO peak to analyze Zero-Crossing Rate (ZCR) and duration.

### D. Classification Thresholds (Romstöck Criteria)
- **Spike**: 1 discrete pulse.
- **Burst**: Multiple pulses, > 50 Hz, total duration < 1.0 second.
- **A-Train**: Duration >= 1.0s, Firing Rate >= 60 Hz (sometimes >= 40 Hz as per modern tuning), and high Rhythmicity (CV <= 0.15).
- **C-Train**: Duration >= 1.0s, high variability (Polyrhythmic, CV > 0.45 or high ISI variance).
- **B-Train**: Duration >= 1.0s, but does not meet A or C criteria.
