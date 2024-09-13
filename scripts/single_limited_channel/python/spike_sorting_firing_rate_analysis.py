# spike_sorting_firing_rate_analysis.py

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
import elephant.spike_train_correlation as escorr  # For correlation analysis
import elephant.spike_train_generation as estg  # For spike train generation and burst detection
import elephant.time_histogram as eth  # For PSTH analysis
import quantities as pq  # For unit handling
import matplotlib.pyplot as plt  # For static visualization
import plotly.express as px  # For interactive visualization with Plotly
from neo.io import NeuralynxIO, BlackrockIO, NixIO  # Multiple IO options for Neo data loading
import numpy as np  # For numerical operations
from scipy.cluster.vq import kmeans  # K-means clustering
from sklearn.mixture import GaussianMixture  # GMM clustering
from sklearn.cluster import DBSCAN  # DBSCAN clustering

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
    if io_type == 'NeuralynxIO':
        reader = NeuralynxIO(dirname=file_path)
    elif io_type == 'BlackrockIO':
        reader = BlackrockIO(filename=file_path)
    elif io_type == 'NixIO':
        reader = NixIO(filename=file_path)
    else:
        raise ValueError("Unsupported IO type.")
    
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
    recording_bp = sp.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
    
    if notch_freq:
        recording_notch = sp.notch_filter(recording_bp, freq=notch_freq)
        recording_cmr = sp.common_reference(recording_notch, reference=common_ref_type)
    else:
        recording_cmr = sp.common_reference(recording_bp, reference=common_ref_type)
    
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

# 4. Postprocessing Module
def postprocess_sorting(sorting, recording, ms_before=1.5, ms_after=2.5, noise_reduction=True):
    """
    Postprocess the sorted spikes to extract features and waveforms, with optional noise reduction.
    
    Args:
    - sorting (si.BaseSorting): Sorted spike data.
    - recording (si.BaseRecording): Preprocessed recording data.
    - ms_before (float): Time in ms before the spike peak to include in the waveform.
    - ms_after (float): Time in ms after the spike peak to include in the waveform.
    - noise_reduction (bool): Whether to apply noise reduction during waveform extraction.
    
    Returns:
    - waveform_extractor (si.WaveformExtractor): Extracted waveforms.
    """
    waveform_extractor = spost.WaveformExtractor.create(recording, sorting, folder='waveforms', remove_existing_folder=True)
    waveform_extractor.set_params(ms_before=ms_before, ms_after=ms_after)
    
    if noise_reduction:
        # Optional noise reduction step
        waveform_extractor = spost.align_spike_waveforms(waveform_extractor)
    
    waveform_extractor.run()
    return waveform_extractor

def compute_quality_metrics(waveform_extractor):
    """
    Compute quality metrics for sorted units, including advanced metrics like isolation distance.
    
    Args:
    - waveform_extractor (si.WaveformExtractor): Extracted waveforms.
    
    Returns:
    - metrics (dict): Quality metrics for each sorted unit.
    """
    metrics = sq.compute_quality_metrics(waveform_extractor, metric_names=['snr', 'isi_violation', 'firing_rate', 'isolation_distance', 'd_prime'])
    return metrics

# 5. Feature Extraction and Clustering Module
def extract_features(waveform_extractor, method='PCA'):
    """
    Extract features from the sorted spike waveforms for clustering using various methods.
    
    Args:
    - waveform_extractor (si.WaveformExtractor): Extracted waveforms.
    - method (str): Feature extraction method ('PCA', 'template_matching', 'wavelet_transform').
    
    Returns:
    - features (np.ndarray): Feature matrix.
    """
    waveforms = waveform_extractor.get_waveforms()
    
    if method == 'PCA':
        # Example: PCA-based feature extraction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        features = pca.fit_transform(waveforms.reshape(waveforms.shape[0], -1))
    
    elif method == 'template_matching':
        # Template matching-based feature extraction
        template = np.mean(waveforms, axis=0)
        features = np.dot(waveforms.reshape(waveforms.shape[0], -1), template.flatten())
    
    elif method == 'wavelet_transform':
        # Wavelet transform-based feature extraction
        import pywt
        coeffs = [pywt.wavedec(waveform, 'db4', level=3) for waveform in waveforms]
        features = np.array([np.concatenate([c.flatten() for c in coeff]) for coeff in coeffs])
    
    else:
        raise ValueError("Unsupported feature extraction method.")
    
    return features

def cluster_spikes(features, method='kmeans'):
    """
    Cluster spikes using specified clustering algorithm.
    
    Args:
    - features (np.ndarray): Feature matrix for clustering.
    - method (str): Clustering method ('kmeans', 'gmm', 'dbscan', 'hierarchical').
    
    Returns:
    - labels (np.ndarray): Cluster labels for each spike.
    """
    if method == 'kmeans':
        centroids, labels = kmeans(features, 3)
    elif method == 'gmm':
        gmm = GaussianMixture(n_components=3).fit(features)
        labels = gmm.predict(features)
    elif method == 'dbscan':
        db = DBSCAN(eps=0.5, min_samples=5).fit(features)
        labels = db.labels_
    elif method == 'hierarchical':
        from scipy.cluster.hierarchy import linkage, fcluster
        Z = linkage(features, 'ward')
        labels = fcluster(Z, t=3, criterion='maxclust')
    else:
        raise ValueError("Unsupported clustering method.")
    
    return labels

# 6. Spike Train Analysis Module
def burst_detection(spike_train, method='log_surprise'):
    """
    Detect bursts in spike train data using specified burst detection algorithm.
    
    Args:
    - spike_train (elephant.SpikeTrain): Spike train data.
    - method (str): Burst detection method ('log_surprise', 'interval_method').
    
    Returns:
    - bursts (list): List of bursts detected.
    """
    if method == 'log_surprise':
        bursts = estg.burst_detection(spike_train, method='log_surprise')
    elif method == 'interval_method':
        bursts = estg.burst_detection(spike_train, method='interval_method', isi_thresh=0.1 * pq.s)
    else:
        raise ValueError("Unsupported burst detection method.")
    return bursts

def cross_correlation_analysis(sorting, bin_size=5 * pq.ms):
    """
    Perform cross-correlation analysis between spike trains of different units.
    
    Args:
    - sorting (si.BaseSorting): Sorted spike data.
    - bin_size (Quantity): Bin size for spike train binning.
    
    Returns:
    - cross_corr_matrix (np.ndarray): Cross-correlation matrix between spike trains.
    """
    binned_spiketrains = [es.BinnedSpikeTrain(sorting.get_unit_spike_train(unit_id) * pq.s, binsize=bin_size) for unit_id in sorting.unit_ids]
    cross_corr_matrix = escorr.corrcoef(binned_spiketrains)
    return cross_corr_matrix

def jpsth_analysis(spike_trains):
    """
    Perform Joint Peri-Stimulus Time Histogram (JPSTH) analysis on spike trains.
    
    Args:
    - spike_trains (list): List of spike trains to analyze.
    
    Returns:
    - jpsth (np.ndarray): JPSTH matrix.
    """
    from elephant.conversion import BinnedSpikeTrain
    binned_sts = [BinnedSpikeTrain(st, binsize=5 * pq.ms) for st in spike_trains]
    jpsth = es.joint_peri_stimulus_time_histogram(binned_sts[0], binned_sts[1], bin_size=5 * pq.ms)
    return jpsth

# 7. Visualization Module

def plot_waveform_overlays(waveform_extractor):
    """
    Plot overlays of spike waveforms for visual inspection.
    
    Args:
    - waveform_extractor (si.WaveformExtractor): Extracted waveforms.
    """
    waveforms = waveform_extractor.get_waveforms()
    plt.figure(figsize=(10, 6))
    for i in range(waveforms.shape[0]):
        plt.plot(waveforms[i].T, color='gray', alpha=0.3)
    plt.title('Overlay of Spike Waveforms')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_clustering_dendrogram(features):
    """
    Plot dendrogram of clustered spike features.
    
    Args:
    - features (np.ndarray): Feature matrix for clustering.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    Z = linkage(features, 'ward')
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Feature Index')
    plt.ylabel('Distance')
    plt.show()

def interactive_cross_correlation_plot(cross_corr_matrix):
    """
    Create an interactive cross-correlation plot using Plotly.
    
    Args:
    - cross_corr_matrix (np.ndarray): Cross-correlation matrix.
    """
    fig = px.imshow(cross_corr_matrix, color_continuous_scale='Viridis')
    fig.update_layout(title='Cross-Correlation Matrix of Spike Trains', xaxis_title='Unit ID', yaxis_title='Unit ID')
    fig.show()

def plot_autocorrelogram(sorting, unit_id, bin_size=5 * pq.ms, max_lag=50 * pq.ms):
    """
    Plot the autocorrelogram for a given unit's spike train.
    
    Args:
    - sorting (si.BaseSorting): Sorted spike data.
    - unit_id (int): ID of the unit for which to compute the autocorrelogram.
    - bin_size (Quantity): Bin size for autocorrelogram.
    - max_lag (Quantity): Maximum lag time for the autocorrelogram.
    """
    spike_train = sorting.get_unit_spike_train(unit_id) * pq.s
    binned_spike_train = es.BinnedSpikeTrain(spike_train, binsize=bin_size)
    auto_corr = escorr.autocorrelogram(binned_spike_train, window_size=max_lag)
    
    plt.figure(figsize=(8, 6))
    plt.bar(auto_corr.times, auto_corr.magnitude.flatten(), width=bin_size.rescale('s').magnitude)
    plt.xlabel('Lag (s)')
    plt.ylabel('Count')
    plt.title(f'Autocorrelogram of Unit {unit_id}')
    plt.show()

def interactive_psth_plot(sorting, stimulus_times, bin_size=10 * pq.ms):
    """
    Create an interactive peri-stimulus time histogram (PSTH) plot using Plotly.
    
    Args:
    - sorting (si.BaseSorting): Sorted spike data.
    - stimulus_times (array-like): Times of stimulus presentation.
    - bin_size (Quantity): Bin size for PSTH.
    """
    psth_dict = {}
    for unit_id in sorting.unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id) * pq.s
        psth = eth.time_histogram([spike_train], binsize=bin_size, t_start=stimulus_times[0], t_stop=stimulus_times[-1])
        psth_dict[unit_id] = psth.magnitude.flatten()
    
    fig = go.Figure()
    for unit_id, psth_data in psth_dict.items():
        fig.add_trace(go.Bar(x=np.arange(len(psth_data)) * bin_size.rescale('s').magnitude, y=psth_data, name=f'Unit {unit_id}'))
    
    fig.update_layout(title="Interactive PSTH", xaxis_title="Time (s)", yaxis_title="Spike Count")
    fig.show()

def plot_burst_detection(spike_train, bursts):
    """
    Visualize detected bursts within a spike train.
    
    Args:
    - spike_train (elephant.SpikeTrain): Spike train data.
    - bursts (list): List of detected bursts.
    """
    plt.figure(figsize=(10, 6))
    plt.eventplot(spike_train.times, lineoffsets=1, color='black')
    for burst in bursts:
        plt.axvspan(burst[0].rescale('s').magnitude, burst[1].rescale('s').magnitude, color='red', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.title('Burst Detection in Spike Train')
    plt.show()