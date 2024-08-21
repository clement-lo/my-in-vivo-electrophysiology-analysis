import numpy as np
from scipy.signal import butter, filtfilt
import argparse

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=300.0, highcut=3000.0, fs=20000.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess electrophysiology data.")
    parser.add_argument("--input", type=str, required=True, help="Path to the raw data file")
    parser.add_argument("--output", type=str, default="data/processed/filtered_data.npy", help="Path to save the filtered data")
    args = parser.parse_args()

    data = np.load(args.input)
    filtered_data = bandpass_filter(data)
    np.save(args.output, filtered_data)