# single_unit_multi_unit_activity_analysis.py

# Import necessary libraries
import neo  # For data handling
import spikeinterface as si  # Core module for SpikeInterface
import spikeinterface.extractors as se  # For data loading and extraction
import spikeinterface.preprocessing as sp  # For data preprocessing
import spikeinterface.sorters as ss  # For spike sorting algorithms
import spikeinterface.postprocessing as spost  # For postprocessing sorted data
import spikeinterface.qualitymetrics as sq  # For quality control metrics
import elephant  # For advanced analysis on spike trains
import elephant.statistics as es  # For statistical measures like firing rates
import elephant.sta as esta  # For spike-triggered averaging
import elephant.conversion as econv  # For converting spike trains
import elephant.spike_train_correlation as escorr  # For correlation analysis
import elephant.spectral as esp  # For spectral analysis
import pyspike as ps  # For synchrony and burst detection
import quantities as pq  # For unit handling
import matplotlib.pyplot as plt  # For static visualization
import plotly.express as px  # For interactive visualization
import numpy as np  # For numerical operations
from neo.io import NeuralynxIO, BlackrockIO, NixIO  # Example IO for Neo data loading
from sklearn.decomposition import PCA  # For dimensionality reduction
from sklearn.mixture import GaussianMixture  # For GMM clustering
from sklearn.cluster import DBSCAN, AgglomerativeClustering  # For clustering

# 1. Data Handling Module
def load_data(file_path, io_type='NeuralynxIO'):
    """
    Load electrophysiological data using Neo and convert to SpikeInterface format.
    
    Args:
    - file_path (str): Path to the file containing raw data.
    - io_type (str): Type of Neo IO to use ('NeuralynxIO', 'BlackrockIO', 'NixIO', etc.).
    
    Returns:
    - recording (si.BaseRecording): Loaded data in SpikeInterface's RecordingExtractor format.
    """
    io_types = {
        'NeuralynxIO': NeuralynxIO(dirname=file_path),
        'BlackrockIO': BlackrockIO(filename=file_path),
        'NixIO': NixIO(filename=file_path)
    }
    
    if io_type not in io_types:
        raise ValueError(f"Unsupported IO type: {io_type}")
    
    reader = io_types[io_type]
    block = reader.read_block()
    segment = block.segments[0]
    analog_signal = segment.analogsignals[0]
    recording = se.NeoRecordingExtractor(analog_signal)
    return recording

# 2. Preprocessing Module
def preprocess_data(recording, freq_min=300, freq_max=3000, notch_freq=None, common_ref_type='median'):
    """
    Preprocess the loaded data by applying bandpass filtering, optional notch filtering, and common reference.
    
    Args:
    - recording (si.BaseRecording): Loaded data in SpikeInterface's RecordingExtractor format.
    - freq_min (int): Minimum frequency for bandpass filter.
    - freq_max (int): Maximum frequency for bandpass filter.
    - notch_freq (float): Frequency for notch filter to remove powerline noise. If None, skip.
    - common_ref_type (str): Type of common reference ('median', 'average', etc.).
    
    Returns:
    - recording_preprocessed (si.BaseRecording): Preprocessed data.
    """
    # Apply bandpass filter
    recording_bp = sp.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
    
    # Apply notch filter if specified
    if notch_freq:
        recording_notch = sp.notch_filter(recording_bp, freq=notch_freq)
    else:
        recording_notch = recording_bp
    
    # Apply common reference
    recording_cmr = sp.common_reference(recording_notch, reference=common_ref_type)
    
    return recording_cmr

# 3. Spike Sorting Module
def sort_spikes(recording, sorter_name='kilosort2', custom_params=None):
    """
    Perform spike sorting on the preprocessed data with configurable parameters.
    
    Args:
    - recording (si.BaseRecording): Preprocessed recording data.
    - sorter_name (str): Name of the sorting algorithm to use (e.g., 'kilosort2').
    - custom_params (dict): Optional custom parameters for the sorting algorithm.
    
    Returns:
    - sorting (si.BaseSorting): Sorted spike data.
    """
    sorter_params = custom_params if custom_params else ss.get_default_params(sorter_name)
    sorting = ss.run_sorter(sorter_name, recording, output_folder='sorting_output', **sorter_params)
    return sorting

# 4. Postprocessing and Feature Extraction Module
def postprocess_sorting(sorting, recording):
    """
    Extract features and waveforms from sorted spikes.
    
    Args:
    - sorting (si.BaseSorting): Sorted spike data.
    - recording (si.BaseRecording): Preprocessed recording data.
    
    Returns:
    - waveform_extractor (si.WaveformExtractor): Extracted waveforms.
    """
    waveform_extractor = spost.WaveformExtractor.create(recording, sorting, folder='waveforms', remove_existing_folder=True)
    waveform_extractor.set_params(ms_before=1.5, ms_after=2.5)
    waveform_extractor.run()
    return waveform_extractor

def extract_features(waveform_extractor, method='pca', n_components=3):
    """
    Extract features from the sorted spike waveforms for clustering using PCA.
    
    Args:
    - waveform_extractor (si.WaveformExtractor): Extracted waveforms.
    - method (str): Method of feature extraction ('pca', 'waveform').
    - n_components (int): Number of PCA components.
    
    Returns:
    - features (np.ndarray): Feature matrix.
    """
    waveforms = waveform_extractor.get_waveforms()
    if method == 'pca':
        pca = PCA(n_components=n_components)
        features = pca.fit_transform(waveforms.reshape(waveforms.shape[0], -1))
    else:
        # Simple feature extraction: mean and std of waveforms
        spike_width = np.mean(np.abs(waveforms), axis=(1, 2))
        spike_amplitude = np.std(waveforms, axis=(1, 2))
        features = np.column_stack((spike_width, spike_amplitude))
    
    return features

# 5. Clustering Module
def cluster_spikes(features, method='gmm', **kwargs):
    """
    Cluster spikes using specified clustering algorithm.
    
    Args:
    - features (np.ndarray): Feature matrix for clustering.
    - method (str): Clustering method ('gmm', 'dbscan', 'hdbscan', etc.).
    - kwargs: Additional parameters for clustering methods.
    
    Returns:
    - labels (np.ndarray): Cluster labels for each spike.
    """
    if method == 'gmm':
        gmm = GaussianMixture(n_components=kwargs.get('n_components', 3))
        labels = gmm.fit_predict(features)
    elif method == 'dbscan':
        db = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
        labels = db.fit_predict(features)
    elif method == 'agglomerative':
        agc = AgglomerativeClustering(n_clusters=kwargs.get('n_clusters', 3))
        labels = agc.fit_predict(features)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    return labels

# 6. Advanced Spike Train Analysis Module
def analyze_spike_trains(sorting, method='burst_detection', **kwargs):
    """
    Analyze spike trains for burst detection, synchrony, and other measures.
    
    Args:
    - sorting (si.BaseSorting): Sorted spike data.
    - method (str): Analysis method ('burst_detection', 'synchrony', etc.).
    - kwargs: Additional parameters for analysis methods.
    
    Returns:
    - analysis_result: Result of the spike train analysis.
    """
    binned_spiketrains = [econv.BinnedSpikeTrain(sorting.get_unit_spike_train(unit_id) * pq.s, binsize=5 * pq.ms) for unit_id in sorting.unit_ids]
    
    if method == 'burst_detection':
        burst_times = estg.burst_detection(sorting)
        return burst_times
    elif method == 'synchrony':
        synchrony_index = ps.spike_train_synchrony(binned_spiketrains)
        return synchrony_index
    elif method == 'cross_correlation':
        correlation_matrix = escorr.corrcoef(binned_spiketrains)
        return correlation_matrix
    else:
        raise ValueError(f"Unsupported analysis method: {method}")

# 7. Spike-Triggered Averaging (STA) and Receptive Field Mapping
def perform_sta(spike_times, stimulus_times, window=(-0.1, 0.1)):
    """
    Perform Spike-Triggered Averaging (STA).
    
    Args:
    - spike_times (list): Spike times.
    - stimulus_times (list): Stimulus times.
    - window (tuple): Time window around the stimulus for averaging (start, end).
    
    Returns:
    - sta (numpy.ndarray): Spike-triggered average.
    """
    sta = esta.spike_triggered_average(spike_times, stimulus_times, window=window)
    return sta

# 8. Visualization Module
def plot_spike_trains(spike_train_profiles):
    """
    Plot spike trains of isolated single units.
    
    Args:
    - spike_train_profiles (dict): Spike train data for each unit.
    """
    for unit, train in spike_train_profiles.items():
        plt.eventplot(train, label=f'Unit {unit}')
    plt.xlabel('Time (s)')
    plt.ylabel('Unit')
    plt.title('Spike Trains of Isolated Units')
    plt.legend()
    plt.show()

def plot_sta(sta):
    """
    Plot Spike-Triggered Average (STA).
    
    Args:
    - sta (numpy.ndarray): Spike-triggered average.
    """
    plt.plot(sta)
    plt.xlabel('Time (s)')
    plt.ylabel('STA Response')
    plt.title('Spike-Triggered Average')
    plt.show()

def plot_cluster_features(features, labels):
    """
    Plot the clustered features using PCA or other feature extraction method.
    
    Args:
    - features (np.ndarray): Feature matrix.
    - labels (np.ndarray): Cluster labels.
    """
    fig = px.scatter(x=features[:, 0], y=features[:, 1], color=labels)
    fig.update_layout(title='Spike Clustering', xaxis_title='Feature 1', yaxis_title='Feature 2')
    fig.show()

# Main function
def main(file_path, stimulus_times):
    """
    Main function to perform Single-Unit and Multi-Unit Activity Analysis.
    
    Args:
    - file_path (str): Path to the data file.
    - stimulus_times (list): List of stimulus times for Spike-Triggered Averaging (STA).
    """
    # Step 1: Load Data
    recording = load_data(file_path)
    
    # Step 2: Preprocess Data
    recording_preprocessed = preprocess_data(recording)

    # Step 3: Perform Spike Sorting
    sorting = sort_spikes(recording_preprocessed)

    # Step 4: Postprocess Sorting and Extract Features
    waveform_extractor = postprocess_sorting(sorting, recording_preprocessed)
    features = extract_features(waveform_extractor)

    # Step 5: Cluster Spikes
    labels = cluster_spikes(features, method='gmm')
    print("Cluster Labels:", labels)

    # Step 6: Analyze Spike Trains
    burst_times = analyze_spike_trains(sorting, method='burst_detection')
    print("Burst Times:", burst_times)

    # Step 7: Perform STA and Receptive Field Mapping
    spike_times = [sorting.get_unit_spike_train(unit_id) for unit_id in sorting.unit_ids]
    sta = perform_sta(spike_times, stimulus_times)
    print("Spike-Triggered Average (STA):", sta)

    # Step 8: Visualize Results
    plot_spike_trains(spike_times)
    plot_sta(sta)
    plot_cluster_features(features, labels)

if __name__ == "__main__":
    # Example file path and stimulus times for demonstration purposes
    example_file_path = 'data/sample_data'  # Adjust the path for your dataset
    example_stimulus_times = [0.5, 1.5, 2.5]  # Example stimulus times in seconds
    main(example_file_path, example_stimulus_times)
