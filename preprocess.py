import numpy as np
from scipy.signal import butter, filtfilt
import argparse
import os

def load_data(file_path):
    """
    Load the raw electrophysiology data from a .npy file.

    Parameters:
    file_path (str): Path to the input file.

    Returns:
    np.ndarray: Loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return np.load(file_path)

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Design a bandpass filter.

    Parameters:
    lowcut (float): Low cutoff frequency (in Hz).
    highcut (float): High cutoff frequency (in Hz).
    fs (float): Sampling rate (in Hz).
    order (int): Order of the filter.

    Returns:
    tuple: Filter coefficients (b, a).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the data.

    Parameters:
    data (np.ndarray): Input signal.
    lowcut (float): Low cutoff frequency (in Hz).
    highcut (float): High cutoff frequency (in Hz).
    fs (float): Sampling rate (in Hz).
    order (int): Order of the filter.

    Returns:
    np.ndarray: Filtered data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def detect_artifacts(data, threshold_factor=5):
    """
    Detect artifacts in the data based on an amplitude threshold.

    Parameters:
    data (np.ndarray): Input signal.
    threshold_factor (float): Multiplier for the standard deviation to set the artifact detection threshold.

    Returns:
    np.ndarray: Indices of detected artifacts.
    """
    threshold = threshold_factor * np.std(data)
    return np.where(np.abs(data) > threshold)[0]

def replace_artifacts(data, artifact_indices):
    """
    Replace detected artifacts with interpolated values.

    Parameters:
    data (np.ndarray): Input signal.
    artifact_indices (np.ndarray): Indices of detected artifacts.

    Returns:
    np.ndarray: Cleaned data with artifacts replaced.
    """
    cleaned_data = data.copy()
    for idx in artifact_indices:
        if idx > 0 and idx < len(data) - 1:
            cleaned_data[idx] = (data[idx - 1] + data[idx + 1]) / 2
    return cleaned_data

def save_data(data, output_path):
    """
    Save the processed data to a .npy file.

    Parameters:
    data (np.ndarray): Processed signal.
    output_path (str): Path to save the output file.
    """
    np.save(output_path, data)

def main(args):
    # Load the raw data
    print(f"Loading data from {args.input}")
    data = load_data(args.input)

    # Apply bandpass filtering
    print(f"Applying bandpass filter: {args.lowcut}-{args.highcut} Hz")
    filtered_data = bandpass_filter(data, args.lowcut, args.highcut, args.fs)

    # Detect and replace artifacts
    print(f"Detecting artifacts with threshold factor: {args.threshold}")
    artifact_indices = detect_artifacts(filtered_data, args.threshold)
    print(f"Detected {len(artifact_indices)} artifacts. Replacing artifacts...")
    cleaned_data = replace_artifacts(filtered_data, artifact_indices)

    # Save the cleaned data
    print(f"Saving cleaned data to {args.output}")
    save_data(cleaned_data, args.output)
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess electrophysiology data (filtering and artifact removal).")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .npy file")
    parser.add_argument("--output", type=str, default="data/processed/cleaned_data.npy", help="Path to save the output .npy file")
    parser.add_argument("--lowcut", type=float, default=300.0, help="Low cutoff frequency for bandpass filter (Hz)")
    parser.add_argument("--highcut", type=float, default=3000.0, help="High cutoff frequency for bandpass filter (Hz)")
    parser.add_argument("--fs", type=float, default=20000.0, help="Sampling rate (Hz)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Threshold factor for artifact detection (multiplier of standard deviation)")
    args = parser.parse_args()

    main(args)
