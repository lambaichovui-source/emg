import os
import glob
import numpy as np
import pandas as pd
from data_handler import DataHandler, SignalPreprocessor

def analyze_dataset():
    dataset_dir = "testing dataset"
    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))

    analysis_results = {}

    for file in csv_files:
        print(f"Analyzing {file}...")
        try:
            handler = DataHandler()

            # Read metadata
            meta = handler.read_csv_metadata(file)

            # Process chunks to get the full signal
            time_chunks = []
            raw_chunks = []
            for t_arr, ch_arr, _ in handler.iter_signal_chunks(file):
                time_chunks.append(t_arr)
                raw_chunks.append(ch_arr)

            if not time_chunks:
                print(f"  Skipping empty file or parsing failed.")
                continue

            time = np.concatenate(time_chunks)
            raw_channels = np.concatenate(raw_chunks, axis=1)

            num_ch = raw_channels.shape[0]
            num_samples = raw_channels.shape[1]
            sample_rate = meta.get("sample_rate", 1280.0)
            duration = num_samples / sample_rate

            # Apply preprocessing to get cleaner signals
            preprocessor = SignalPreprocessor(sample_rate=sample_rate, num_channels=num_ch)
            filtered_channels = preprocessor.apply_notch(raw_channels)

            # Calculate metrics
            baseline_mad_per_ch = []
            amplitude_range_per_ch = []
            rms_per_ch = []

            for ch in range(num_ch):
                sig = filtered_channels[ch]

                # Baseline estimation (MAD - Median Absolute Deviation)
                # MAD = median(|Xi - median(X)|)
                median_val = np.median(sig)
                mad = np.median(np.abs(sig - median_val))
                baseline_mad_per_ch.append(float(mad))

                # Amplitude Range
                vmax = np.max(sig)
                vmin = np.min(sig)
                amplitude_range_per_ch.append((float(vmin), float(vmax)))

                # RMS
                rms = np.sqrt(np.mean(sig**2))
                rms_per_ch.append(float(rms))

            # Aggregate metrics
            mean_mad = np.mean(baseline_mad_per_ch)
            mean_rms = np.mean(rms_per_ch)
            max_amp = np.max([v[1] for v in amplitude_range_per_ch])
            min_amp = np.min([v[0] for v in amplitude_range_per_ch])

            # Estimate SNR (Signal to Noise Ratio in dB)
            # Signal power ~ RMS^2, Noise power ~ MAD^2
            # Very rough estimation since actual signal vs noise separation is complex
            snr_db = 20 * np.log10(max(mean_rms, 1e-6) / max(mean_mad, 1e-6))

            analysis_results[os.path.basename(file)] = {
                "Metadata": meta,
                "Duration (s)": float(duration),
                "Sample Rate (Hz)": float(sample_rate),
                "Number of Channels": num_ch,
                "Total Samples": num_samples,
                "Mean Baseline Noise (MAD, µV)": float(mean_mad),
                "Mean RMS (µV)": float(mean_rms),
                "Global Amplitude Range (µV)": (float(min_amp), float(max_amp)),
                "Estimated Global SNR (dB)": float(snr_db),
            }

            print(f"  Processed {num_samples} samples across {num_ch} channels.")

        except Exception as e:
            print(f"  Error processing {file}: {e}")

    # Write output to markdown
    with open("dataset_metrics_analysis.md", "w") as f:
        f.write("# Dataset Metrics and Normalization Analysis\n\n")
        f.write("This document summarizes the analysis of the raw EMG datasets in the `testing dataset` folder, providing core metrics, thresholds, and normalization strategies for processing physiological signals.\n\n")

        f.write("## 1. File Summaries\n\n")
        for filename, data in analysis_results.items():
            f.write(f"### File: `{filename}`\n")
            f.write(f"- **Duration**: {data['Duration (s)']:.2f} seconds\n")
            f.write(f"- **Sample Rate**: {data['Sample Rate (Hz)']:.1f} Hz\n")
            f.write(f"- **Channels**: {data['Number of Channels']}\n")
            f.write(f"- **Total Samples**: {data['Total Samples']:,}\n")
            f.write(f"- **Global Amplitude Range**: [{data['Global Amplitude Range (µV)'][0]:.2f} µV, {data['Global Amplitude Range (µV)'][1]:.2f} µV]\n")
            f.write(f"- **Mean Baseline Noise (MAD)**: {data['Mean Baseline Noise (MAD, µV)']:.2f} µV\n")
            f.write(f"- **Mean RMS**: {data['Mean RMS (µV)']:.2f} µV\n")
            f.write(f"- **Estimated Global SNR**: {data['Estimated Global SNR (dB)']:.2f} dB\n")
            f.write("\n")

        # Analysis section
        f.write("## 2. Core Metrics & Physiological Normalization\n\n")

        f.write("### A. Baseline and Noise Estimation\n")
        f.write("Based on the data analysis, the baseline noise level (measured via Median Absolute Deviation - MAD) fluctuates depending on the recording. However, a safe assumption for physiological rest state across channels is the calculated average MAD.\n")
        f.write("- **Threshold Strategy**: To detect active signals over the noise floor, a dynamic threshold of `Threshold = Median + (K * MAD)` should be used, where `K` is typically 3 to 5.\n")
        f.write("- **Energy-Linked Noise Gate**: As per methodology, apply TKEO (Teager-Kaiser Energy Operator) to highlight the high-frequency/high-amplitude MUPs and calculate MAD on the TKEO signal for the most robust noise gating.\n\n")

        f.write("### B. Signal Amplitude & Normalization\n")
        f.write("The amplitude range varies dramatically (e.g., from small baseline noise to large MUP spikes and artifacts). Raw values are in microvolts (µV).\n")
        f.write("- **Clipping & Artifact Blanking**: Extreme values (e.g., beyond ±2000 µV or ±5000 µV depending on the cautery threshold) are usually non-physiological artifacts and should be blanked.\n")
        f.write("- **Normalization Strategy**: When feeding data into AI models (like the MOGRUEANet), use `mapminmax` normalization (scaling to [-1, 1]) per window or z-score normalization (`(x - mean) / std`) to ensure magnitude invariance for morphology analysis.\n\n")

        f.write("### C. Windowing for Feature Extraction\n")
        f.write("For continuous analysis and real-time processing, the data must be chunked into overlapping windows.\n")
        f.write("- **Typical Window Size**: 256 samples (which corresponds to exactly 0.2 seconds at 1280 Hz) provides a good balance between time resolution and frequency analysis.\n")
        f.write("- **Step Size**: 128 samples (50% overlap) ensures continuity and captures events spanning window boundaries.\n")
        f.write("- **MUP Detection Window**: For fine-grained Motor Unit Potential (MUP) guardrails, extract micro-windows of 10-15ms around the TKEO peak to analyze Zero-Crossing Rate (ZCR) and duration.\n\n")

        f.write("### D. Classification Thresholds (Romstöck Criteria)\n")
        f.write("- **Spike**: 1 discrete pulse.\n")
        f.write("- **Burst**: Multiple pulses, > 50 Hz, total duration < 1.0 second.\n")
        f.write("- **A-Train**: Duration >= 1.0s, Firing Rate >= 60 Hz (sometimes >= 40 Hz as per modern tuning), and high Rhythmicity (CV <= 0.15).\n")
        f.write("- **C-Train**: Duration >= 1.0s, high variability (Polyrhythmic, CV > 0.45 or high ISI variance).\n")
        f.write("- **B-Train**: Duration >= 1.0s, but does not meet A or C criteria.\n\n")

    print("Analysis complete. Results written to dataset_metrics_analysis.md")

if __name__ == "__main__":
    analyze_dataset()
