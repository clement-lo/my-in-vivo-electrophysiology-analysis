# advanced_spike_sorting.py

# Import necessary libraries
import neo  # For data handling
import spikeinterface as si  # Core module for SpikeInterface
import spikeinterface.extractors as se  # For data loading and extraction
import spikeinterface.preprocessing as sp  # For data preprocessing
import spikeinterface.sorters as ss  # For spike sorting algorithms
import spikeinterface.postprocessing as spost  # For postprocessing sorted data
import spikeinterface.qualitymetrics as sq  # For quality control metrics
import elephant.spike_train_correlation as escorr  # For spike train correlations
import elephant.connectivity as econn  # For network connectivity analysis
import pyspike as ps  # For synchrony and burst detection
import quantities as pq  # For unit handling
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For static visualization
import plotly.express as px  # For interactive visualization
from sklearn.decomposition import PCA  # For dimensionality reduction
from sklearn.manifold import TSNE  # For t-SNE visualization
from umap import UMAP  # For UMAP visualization
from sklearn.cluster import AffinityPropagation, DBSCAN, AgglomerativeClustering  # For clustering algorithms
import hdbscan  # For HDBSCAN clustering
from neo.io import NeuralynxIO, BlackrockIO  # Example IOs for Neo data loading

# 1. Data Handling Module
def load_mea_data(file_path, io_type='NeuralynxIO'):
    """
    Load MEA data using Neo and convert to SpikeInterface format.
    
    Args:
    - file_path (str): Path to the file containing raw data.
    - io_type (str): Type of Neo IO to use ('NeuralynxIO', 'BlackrockIO', etc.).
    
    Returns:
    - recording (si.BaseRecording): Loaded data in SpikeInterface's RecordingExtractor format.
    """
    try:
        io_types = {
            'NeuralynxIO': NeuralynxIO(dirname=file_path),
            'BlackrockIO': BlackrockIO(filename=file_path)
        }
        
        if io_type not in io_types:
            raise ValueError(f"Unsupported IO type: {io_type}")
        
        reader = io_types[io_type]
        block = reader.read_block()
        analog_signal = block.segments[0].analogsignals[0]
        recording = se.NeoRecordingExtractor(analog_signal)
        return recording
    except Exception as e:
        print(f"Error loading MEA data: {e}")
        raise

# 2. Preprocessing Module
def preprocess_data(recording, freq_min=300, freq_max=6000, noise_reduction='CAR'):
    """
    Apply bandpass filter, normalization, and optional noise reduction techniques.
    
    Args:
    - recording (si.BaseRecording): Loaded data in SpikeInterface's RecordingExtractor format.
    - freq_min (int): Minimum frequency for bandpass filter.
    - freq_max (int): Maximum frequency for bandpass filter.
    - noise_reduction (str): Noise reduction technique ('CAR', 'ICA', etc.).
    
    Returns:
    - recording_preprocessed (si.BaseRecording): Preprocessed data.
    """
    try:
        recording_filtered = sp.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
        
        if noise_reduction == 'CAR':
            recording_filtered = sp.common_reference(recording_filtered, reference='median')
        elif noise_reduction == 'ICA':
            recording_filtered = sp.ica(recording_filtered)
        
        recording_normalized = sp.zscore(recording_filtered)
        return recording_normalized
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        raise

# 3. Dimensionality Reduction Module
def reduce_dimensionality(features, method='pca'):
    """
    Apply PCA, t-SNE, or UMAP for dimensionality reduction.
    
    Args:
    - features (np.ndarray): Feature matrix.
    - method (str): Dimensionality reduction method ('pca', 'tsne', 'umap').
    
    Returns:
    - reduced_features (np.ndarray): Reduced feature matrix.
    """
    try:
        if method == 'pca':
            pca = PCA(n_components=3)
            reduced_features = pca.fit_transform(features)
        elif method == 'tsne':
            tsne = TSNE(n_components=3)
            reduced_features = tsne.fit_transform(features)
        elif method == 'umap':
            umap = UMAP(n_components=3)
            reduced_features = umap.fit_transform(features)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        return reduced_features
    except Exception as e:
        print(f"Error in dimensionality reduction: {e}")
        raise

# 4. Spike Sorting Module
def perform_spike_sorting(recording, sorter_name='kilosort2', custom_params=None):
    """
    Perform spike sorting on MEA data using advanced sorters.
    
    Args:
    - recording (si.BaseRecording): Preprocessed recording data.
    - sorter_name (str): Name of the spike sorting algorithm (e.g., 'kilosort2').
    - custom_params (dict): Optional custom parameters for the sorting algorithm.
    
    Returns:
    - sorting (si.BaseSorting): Sorted spike data.
    """
    try:
        sorter_params = custom_params if custom_params else ss.get_default_params(sorter_name)
        sorting = ss.run_sorter(sorter_name, recording, output_folder='sorting_output', **sorter_params)
        return sorting
    except Exception as e:
        print(f"Error in spike sorting: {e}")
        raise

# 5. Postprocessing and Quality Metrics Module
def postprocess_sorting(sorting, recording):
    """
    Extract waveforms and compute quality metrics for sorted units.
    
    Args:
    - sorting (si.BaseSorting): Sorted spike data.
    - recording (si.BaseRecording): Preprocessed recording data.
    
    Returns:
    - waveform_extractor (si.WaveformExtractor): Extracted waveforms.
    - quality_metrics (dict): Quality metrics for each sorted unit.
    """
    try:
        waveform_extractor = spost.WaveformExtractor.create(recording, sorting, folder='waveforms', remove_existing_folder=True)
        waveform_extractor.set_params(ms_before=1.5, ms_after=2.5)
        waveform_extractor.run()
        quality_metrics = sq.compute_quality_metrics(waveform_extractor, metric_names=['snr', 'isi_violation', 'firing_rate'])
        return waveform_extractor, quality_metrics
    except Exception as e:
        print(f"Error in postprocessing sorting: {e}")
        raise

# 6. Clustering and Validation Module
def cluster_spikes(features, method='HDBSCAN', **kwargs):
    """
    Cluster spikes using specified clustering algorithm.
    
    Args:
    - features (np.ndarray): Feature matrix for clustering.
    - method (str): Clustering method ('AffinityPropagation', 'HDBSCAN', etc.).
    - kwargs: Additional parameters for clustering methods.
    
    Returns:
    - labels (np.ndarray): Cluster labels for each spike.
    """
    try:
        if method == 'AffinityPropagation':
            clustering = AffinityPropagation().fit(features)
        elif method == 'HDBSCAN':
            clustering = hdbscan.HDBSCAN().fit(features)
        elif method == 'DBSCAN':
            db = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
            clustering = db.fit(features)
        elif method == 'AgglomerativeClustering':
            agc = AgglomerativeClustering(n_clusters=kwargs.get('n_clusters', 3))
            clustering = agc.fit(features)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        return clustering.labels_
    except Exception as e:
        print(f"Error in clustering spikes: {e}")
        raise

# 7. Spike Train and Connectivity Analysis Module
def compute_spike_train_correlations(sorting):
    """
    Compute cross-correlograms for spike trains using Elephant.
    
    Args:
    - sorting (si.BaseSorting): Sorted spike data.
    
    Returns:
    - correlations (np.ndarray): Cross-correlogram matrix.
    """
    try:
        spike_trains = [sorting.get_unit_spike_train(unit_id) for unit_id in sorting.unit_ids]
        correlations = escorr.corrcoef(spike_trains)
        return correlations
    except Exception as e:
        print(f"Error in computing spike train correlations: {e}")
        raise

def compute_connectivity_matrix(spike_trains):
    """
    Compute network connectivity using Granger causality and Directed Transfer Function.
    
    Args:
    - spike_trains (list): List of spike trains.
    
    Returns:
    - conn_matrix (np.ndarray): Network connectivity matrix.
    """
    try:
        conn_matrix = econn.granger_causality(spike_trains)
        return conn_matrix
    except Exception as e:
        print(f"Error in computing connectivity matrix: {e}")
        raise

# 8. Visualization Module
def plot_clusters(reduced_features, labels):
    """
    Visualize clustering results in 3D.
    
    Args:
    - reduced_features (np.ndarray): Reduced feature matrix.
    - labels (np.ndarray): Cluster labels.
    """
    try:
        fig = px.scatter_3d(x=reduced_features[:, 0], y=reduced_features[:, 1], z=reduced_features[:, 2], color=labels)
        fig.update_layout(title='3D Clustering Visualization', xaxis_title='Component 1', yaxis_title='Component 2', zaxis_title='Component 3')
        fig.show()
    except Exception as e:
        print(f"Error in plotting clusters: {e}")
        raise

# Main function
def main(file_path, io_type='NeuralynxIO'):
    """
    Main function to perform advanced spike sorting and clustering analysis.
    
    Args:
    - file_path (str): Path to the data file.
    - io_type (str): Type of Neo IO to use for data loading.
    """
    # Step 1: Load Data
    recording = load_mea_data(file_path, io_type)

    # Step 2: Preprocess Data
    recording_preprocessed = preprocess_data(recording)

    # Step 3: Perform Spike Sorting
    sorting = perform_spike_sorting(recording_preprocessed)

    # Step 4: Postprocess Sorting and Compute Quality Metrics
    waveform_extractor, quality_metrics = postprocess_sorting(sorting, recording_preprocessed)
    print("Quality Metrics:", quality_metrics)

    # Step 5: Feature Extraction and Dimensionality Reduction
    features = waveform_extractor.get_waveforms().reshape(len(sorting.unit_ids), -1)
    reduced_features = reduce_dimensionality(features, method='pca')

    # Step 6: Cluster Spikes
    labels = cluster_spikes(reduced_features, method='HDBSCAN')
    print("Cluster Labels:", labels)

    # Step 7: Analyze Spike Train Correlations
    correlations = compute_spike_train_correlations(sorting)
    print("Spike Train Correlation Matrix:", correlations)

    # Step 8: Visualize Clustering Results
    plot_clusters(reduced_features, labels)

if __name__ == "__main__":
    # Example file path for demonstration purposes
    example_file_path = 'data/sample_mea_data'
    main(example_file_path, io_type='NeuralynxIO')