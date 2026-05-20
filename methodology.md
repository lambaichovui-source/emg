Modular Prompts for the IONM Mathematical Engine

Copy and paste these prompts one by one into your AI coding assistant to build the pipeline.

PROMPT 1: The Preprocessor & Energy-Linked Noise Gate

Copy the text below:

Act as an Expert Biomedical Signal Processing Engineer. 

**Objective:**
Write a Python module using NumPy and SciPy to clean a raw EMG signal and apply an "Energy-Linked Noise Gate" to safely zero out baseline fuzz without destroying the zero-crossings of Motor Unit Potentials (MUPs).

**Requirements & Math:**
Create a function `apply_energy_gate(signal, fs)` that performs the following steps:
1. **Filtering:** Apply a 4th-order Butterworth bandpass filter (30 Hz - 500 Hz) and a 60 Hz notch filter to the input signal.
2. **TKEO Calculatiown:** Calculate the Teager-Kaiser Energy Operator: `TKEO[n] = x[n]^2 - x[n-1]*x[n+1]`.
3. **Noise Floor Estimation:** Calculate the Median Absolute Deviation (MAD) of the TKEO signal to robustly estimate the baseline noise level. Define the threshold as `T = 3 * MAD`.
4. **Boolean Mask:** Create a binary mask array where the mask is `1` if `TKEO > T`, and `0` otherwise.
5. **Dilation:** Use `scipy.ndimage.binary_dilation` to expand the `1`s in the mask by 2 milliseconds on both sides. This protects the MUP zero-crossings.
6. **Application:** Multiply the filtered signal by the dilated mask. 

**Output:**
Return the fully gated, clean signal. Please include a small test block that simulates a noisy sine wave with a large spike to verify the baseline is perfectly 0 while the spike remains intact.



PROMPT 2: The MUP Trigger & Morphological Guardrails

Copy the text below:

Act as an Expert Biomedical Signal Processing Engineer.

**Objective:**
I have a cleaned, gated EMG signal. I need a Python function using NumPy/SciPy to find the exact timestamps of Motor Unit Potentials (MUPs) using adaptive statistics and geometric guardrails.

**Requirements & Math:**
Create a function `detect_mup_timestamps(gated_signal, fs)`:
1. **Adaptive Thresholding:** Calculate the Teager-Kaiser Energy Operator (TKEO) of the gated signal. Calculate the rolling mean (μ) and standard deviation (σ) of the TKEO over 1-second windows. Set the detection trigger to `Th = μ + (5 * σ)`.
2. **Peak Detection:** Use `scipy.signal.find_peaks` on the TKEO signal using the dynamic threshold `Th` to find trigger indices.
3. **Morphological Guardrails (The Bouncer):** For every triggered index, extract a 10-millisecond micro-window from the `gated_signal` centered on the peak.
   - Calculate the Zero-Crossing Rate (ZCR) of this micro-window.
   - Reject the peak if ZCR == 0 (baseline shift) or ZCR > 4 (high-frequency artifact noise).
   - Calculate the Duration of the wave (time from the first non-zero sample to the last non-zero sample in the window). Reject if Duration < 2ms or > 15ms.
4. **Output:** Return a NumPy array of the surviving timestamps (in seconds).



PROMPT 3: Train Clustering & Feature Extraction

Copy the text below:

Act as an Expert Biomedical Signal Processing Engineer.

**Objective:**
I have an array of detected MUP timestamps (in seconds) and the original continuous EMG voltage array. I need a Python engine to cluster these spikes into clinical "Events" and extract their mathematical kinematics.

**Requirements & Math:**
Create a function `extract_train_features(timestamps, raw_signal, fs)`:
1. **Clustering:** Iterate through the timestamps. If the Inter-Spike Interval (ISI) between two consecutive spikes is > 0.5 seconds, close the current cluster and start a new one.
2. **For each closed cluster, calculate:**
   - **Duration:** `t_last - t_first`
   - **Firing Rate:** Calculate the array of ISIs, then Instantaneous Frequencies (`1/ISI`). Calculate the `mean` and `median` frequency.
   - **Rhythmicity (CV):** Coefficient of Variation = `std(ISI) / mean(ISI)`.
   - **Tendency (Slope):** Use `scipy.stats.linregress` to plot the Instantaneous Frequencies against time. Return the slope (`m`) in Hz/sec.
   - **Amplitude (Vpp):** For each timestamp in the cluster, extract a 5ms window from the `raw_signal` and calculate Peak-to-Peak voltage `max - min`. Return the median Vpp for the cluster.
3. **Output:** Return a list of dictionaries, where each dictionary contains the mathematical features for one clustered event.



PROMPT 4: The Romstöck Clinical Classifier

Copy the text below:

Act as an Expert Clinical Neuromonitoring Software Engineer.

**Objective:**
I have a dictionary of mathematical features for an EMG event (Duration, Mean Firing Rate, Rhythmicity/CV, Tendency/Slope). I need a pure Python logic function to categorize this event into standard IONM clinical classifications based on the Romstöck criteria.

**Requirements & Logic:**
Create a function `classify_emg_event(features_dict)` that applies the following decision tree and returns a String label:
1. **Spike/Burst Check:** If `Duration < 1.0` second -> Return `"Spike"` (if only 1 interval) or `"Burst"`.
2. **A-Train Check:** If `Duration >= 1.0`, AND `Mean Firing Rate >= 60`, AND `Rhythmicity (CV) <= 0.15` -> Return `"A-Train (High Risk)"`.
3. **C-Train Check:** If `Duration >= 1.0`, AND `Rhythmicity (CV) > 0.45` -> Return `"C-Train (Polyrhythmic)"`.
4. **B-Train Check:** If `Duration >= 1.0` and it does not meet A or C criteria -> Return `"B-Train"`.
5. **Output:** The function must simply take the dictionary and return the correct string label. Please write a small unit test passing mock dictionaries to prove the logic routing works perfectly.

