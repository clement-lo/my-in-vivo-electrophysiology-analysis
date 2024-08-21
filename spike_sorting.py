import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spike sorting for electrophysiology data.")
    parser.add_argument("--input", type=str, required=True, help="Path to the filtered data file")
    parser.add_argument("--output_features", type=str, default="data/processed/spike_features.npy", help="Path to save spike features")
    parser.add_argument("--output_labels", type=str, default="data/processed/spike_labels.npy", help="Path to save spike labels")
    args = parser.parse_args()

    data = np.load(args.input)
    threshold = 5 * np.std(data)
    spike_indices = np.where(data < -threshold)[0]
    window_size = 30
    spike_waveforms = np.array([data[i-window_size:i+window_size] for i in spike_indices])

    pca = PCA(n_components=2)
    spike_features = pca.fit_transform(spike_waveforms)

    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(spike_features)

    np.save(args.output_features, spike_features)
    np.save(args.output_labels, labels)