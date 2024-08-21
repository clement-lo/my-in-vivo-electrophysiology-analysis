import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.signal import welch, spectrogram
import argparse
import os

def load_data(file_path):
    """
    Load preprocessed electrophysiology data from a .npy file.

    Parameters:
    file_path (str): Path to the input file.

    Returns:
    np.ndarray: Loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return np.load(file_path)

def detect_spikes(data, threshold_factor=-5):
    """
    Detect spikes in the signal using a simple threshold.

    Parameters:
    data (np.ndarray): Input signal.
    threshold_factor (float): Multiplier of the standard deviation to set the spike detection threshold.

    Returns:
    np.ndarray: Indices of detected spikes.
    """
    threshold = threshold_factor * np.std(data)
    spike_indices = np.where(data < threshold)[0]
    return spike_indices

def extract_spike_waveforms(data, spike_indices, window_size=30):
    """
    Extract spike waveforms around detected spike indices.

    Parameters:
    data (np.ndarray): Input signal.
    spike_indices (np.ndarray): Indices of detected spikes.
    window_size (int): Number of samples before and after the spike to extract.

    Returns:
    np.ndarray: Extracted spike waveforms.
    """
    spike_waveforms = np.array([data[i-window_size:i+window_size] for i in spike_indices if i > window_size and i < len(data) - window_size])
    return spike_waveforms

def perform_spike_sorting(spike_waveforms, n_clusters=3):
    """
    Perform spike sorting using PCA for feature extraction and k-means clustering.

    Parameters:
    spike_waveforms (np.ndarray): Extracted spike waveforms.
    n_clusters (int): Number of clusters for k-means.

    Returns:
    np.ndarray: Cluster labels for each spike.
    np.ndarray: PCA features of the spike waveforms.
    """
    pca = PCA(n_components=2)
    spike_features = pca.fit_transform(spike_waveforms)
    
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(spike_features)
    
    return labels, spike_features

def analyze_lfp(data, fs=20000):
    """
    Analyze local field potentials (LFPs) by computing the power spectral density (PSD).

    Parameters:
    data (np.ndarray): Input signal (LFP).
    fs (float): Sampling rate (Hz).

    Returns:
    np.ndarray: Frequencies.
    np.ndarray: Power spectral density.
    """
    frequencies, psd = welch(data, fs=fs, nperseg=2048)
    return frequencies, psd

def compute_isi(spike_indices, fs=20000):
    """
    Compute inter-spike intervals (ISIs).

    Parameters:
    spike_indices (np.ndarray): Indices of detected spikes.
    fs (float): Sampling rate (Hz).

    Returns:
    np.ndarray: Inter-spike intervals (in seconds).
    """
    isi = np.diff(spike_indices) / fs
    return isi

def compute_spectrogram(data, fs=20000):
    """
    Compute the spectrogram for time-frequency analysis.

    Parameters:
    data (np.ndarray): Input signal.
    fs (float): Sampling rate (Hz).

    Returns:
    np.ndarray: Frequencies.
    np.ndarray: Times.
    np.ndarray: Spectrogram (power).
    """
    frequencies, times, Sxx = spectrogram(data, fs=fs, nperseg=1024)
    return frequencies, times, Sxx

def save_results(output_dir, spike_features, labels, frequencies, psd, isi, spectrogram_data):
    """
    Save the analysis results to specified output files.

    Parameters:
    output_dir (str): Directory to save output files.
    spike_features (np.ndarray): PCA features of the spikes.
    labels (np.ndarray): Cluster labels for the spikes.
    frequencies (np.ndarray): Frequencies for LFP analysis.
    psd (np.ndarray): Power spectral density for LFP analysis.
    isi (np.ndarray): Inter-spike intervals.
    spectrogram_data (tuple): Time-frequency analysis results (frequencies, times, spectrogram).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, 'spike_features.npy'), spike_features)
    np.save(os.path.join(output_dir, 'spike_labels.npy'), labels)
    np.save(os.path.join(output_dir, 'lfp_frequencies.npy'), frequencies)
    np.save(os.path.join(output_dir, 'lfp_psd.npy'), psd)
    np.save(os.path.join(output_dir, 'isi.npy'), isi)
    np.save(os.path.join(output_dir, 'spectrogram_frequencies.npy'), spectrogram_data[0])
    np.save(os.path.join(output_dir, 'spectrogram_times.npy'), spectrogram_data[1])
    np.save(os.path.join(output_dir, 'spectrogram_power.npy'), spectrogram_data[2])

def main(args):
    # Load the preprocessed data
    print(f"Loading data from {args.input}")
    data = load_data(args.input)

    # Spike detection and sorting
    print(f"Detecting spikes with threshold factor: {args.threshold}")
    spike_indices = detect_spikes(data, args.threshold)
    spike_waveforms = extract_spike_waveforms(data, spike_indices)
    print(f"Performing spike sorting with {args.n_clusters} clusters")
    labels, spike_features = perform_spike_sorting(spike_waveforms, args.n_clusters)

    # LFP analysis
    print("Analyzing local field potentials (LFPs)")
    frequencies, psd = analyze_lfp(data, args.fs)

    # Compute inter-spike intervals (ISI)
    print("Computing inter-spike intervals (ISI)")
    isi = compute_isi(spike_indices, args.fs)

    # Time-frequency analysis (Spectrogram)
    print("Computing spectrogram for time-frequency analysis")
    spectrogram_data = compute_spectrogram(data, args.fs)

    # Save all results
    print(f"Saving results to {args.output_dir}")
    save_results(args.output_dir, spike_features, labels, frequencies, psd, isi, spectrogram_data)
    print("Analysis complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze electrophysiology data (spike sorting, LFP analysis, and metrics extraction).")
    parser.add_argument("--input", type=str, required=True, help="Path to the input preprocessed .npy file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the output files")
    parser.add_argument("--threshold", type=float, default=-5.0, help="Threshold factor for spike detection (negative multiplier of standard deviation)")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for spike sorting")
    parser.add_argument("--fs", type=float, default=20000.0, help="Sampling rate (Hz)")
    args = parser.parse_args()

    main(args)
