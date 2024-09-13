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
import elephant.spectral as espt  # For spectral analysis
import elephant.time_histogram as eth  # For PSTH analysis
import quantities as pq  # For unit handling
import matplotlib.pyplot as plt  # For static visualization
import plotly.express as px  # For interactive visualization with Plotly
from neo.io import NeuralynxIO  # Example IO for Neo data loading
import numpy as np  # For numerical operations

# 1. Data Handling Module
def load_data(file_path):
    """
    Load electrophysiological data using Neo and convert to SpikeInterface format.
    
    Args:
    - file_path (str): Path to the file containing raw data.
    
    Returns:
    - recording (si.BaseRecording): Loaded data in SpikeInterface's RecordingExtractor format.
    """
    reader = NeuralynxIO(dirname=file_path)
    block = reader.read_block()
    segment = block.segments[0]
    analog_signal = segment.analogsignals[0]
    recording = se.NeoRecordingExtractor(analog_signal)
    return recording

# 2. Preprocessing Module
def preprocess_data(recording, freq_min=300, freq_max=3000, common_ref_type='median'):
    """
    Preprocess the loaded data by applying bandpass filtering and common reference.
    
    Args:
    - recording (si.BaseRecording): Loaded data in SpikeInterface's RecordingExtractor format.
    - freq_min (int): Minimum frequency for bandpass filter.
    - freq_max (int): Maximum frequency for bandpass filter.
    - common_ref_type (str): Type of common reference ('median', 'average', etc.).
    
    Returns:
    - recording_preprocessed (si.BaseRecording): Preprocessed data.
    """
    recording_bp = sp.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
    recording_cmr = sp.common_reference(recording_bp, reference=common_ref_type)
    return recording_cmr

# 3. Spike Sorting Module
def sort_spikes(recording, sorter_name='kilosort2'):
    """
    Perform spike sorting on the preprocessed data.
    
    Args:
    - recording (si.BaseRecording): Preprocessed recording data.
    - sorter_name (str): Name of the sorting algorithm to use (e.g., 'kilosort2').
    
    Returns:
    - sorting (si.BaseSorting): Sorted spike data.
    """
    sorter_params = ss.get_default_params(sorter_name)
    sorting = ss.run_sorter(sorter_name, recording, output_folder='sorting_output', **sorter_params)
    return sorting

# 4. Postprocessing Module
def postprocess_sorting(sorting, recording):
    """
    Postprocess the sorted spikes to extract features and waveforms.
    
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

def compute_quality_metrics(waveform_extractor):
    """
    Compute quality metrics for sorted units.
    
    Args:
    - waveform_extractor (si.WaveformExtractor): Extracted waveforms.
    
    Returns:
    - metrics (dict): Quality metrics for each sorted unit.
    """
    metrics = sq.compute_quality_metrics(waveform_extractor, metric_names=['snr', 'isi_violation', 'firing_rate'])
    return metrics

# 5. Advanced Analysis Module
def calculate_firing_rate(sorting, bin_size=100 * pq.ms):
    """
    Calculate the mean firing rate from the sorted spike data.
    
    Args:
    - sorting (si.BaseSorting): Sorted spike data.
    - bin_size (Quantity): Time bin size for firing rate calculation.
    
    Returns:
    - firing_rates (dict): Dictionary of firing rates for each unit.
    """
    firing_rates = {}
    for unit_id in sorting.unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id) * pq.s
        rate = es.mean_firing_rate(spike_train, t_start=0 * pq.s, t_stop=max(spike_train), bin_size=bin_size)
        firing_rates[unit_id] = rate
    return firing_rates

def analyze_spike_train_correlation(sorting, method='pearson'):
    """
    Analyze spike train correlations between units.
    
    Args:
    - sorting (si.BaseSorting): Sorted spike data.
    - method (str): Correlation method ('pearson', 'spearman', etc.).
    
    Returns:
    - correlation_matrix (np.ndarray): Correlation matrix of spike trains.
    """
    binned_spiketrains = [es.BinnedSpikeTrain(sorting.get_unit_spike_train(unit_id) * pq.s, binsize=5 * pq.ms) for unit_id in sorting.unit_ids]
    correlation_matrix = escorr.corrcoef(binned_spiketrains, method=method)
    return correlation_matrix

def perform_time_frequency_analysis(spike_train, fs=1000):
    """
    Perform time-frequency analysis using spectral methods.
    
    Args:
    - spike_train (elephant.SpikeTrain): Spike train data.
    - fs (int): Sampling frequency.
    
    Returns:
    - freqs (np.ndarray): Frequency bins.
    - psd (np.ndarray): Power spectral density.
    """
    psd, freqs = espt.welch_psd(spike_train, fs=fs)
    return freqs, psd

# 6. Visualization Module
def plot_raster(sorting):
    """
    Plot raster plot of the spike sorting results.
    
    Args:
    - sorting (si.BaseSorting): Sorted spike data.
    """
    spike_times = [sorting.get_unit_spike_train(unit_id) for unit_id in sorting.unit_ids]
    plt.eventplot(spike_times)
    plt.xlabel('Time (s)')
    plt.ylabel('Units')
    plt.title('Raster Plot')
    plt.show()

def plot_firing_rate_histogram(firing_rates):
    """
    Plot histogram of firing rates using Plotly for interactive exploration.
    
    Args:
    - firing_rates (dict): Firing rates of units.
    """
    fig = px.histogram(x=list(firing_rates.values()), labels={'x': 'Firing Rate (Hz)'})
    fig.update_layout(title="Firing Rate Histogram", xaxis_title="Firing Rate (Hz)", yaxis_title="Count")
    fig.show()

def plot_correlation_matrix(correlation_matrix):
    """
    Plot correlation matrix using Matplotlib or Plotly.
    
    Args:
    - correlation_matrix (np.ndarray): Correlation matrix of spike trains.
    """
    plt.imshow(correlation_matrix, cmap='viridis', interpolation='none')
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Spike Train Correlation Matrix')
    plt.show()

# Main function
def main(file_path):
    """
    Main function to perform spike sorting and firing rate analysis.
    
    Args:
    - file_path (str): Path to the data file.
    """
    # Step 1: Load Data
    recording = load_data(file_path)
    
    # Step 2: Preprocess Data
    recording_preprocessed = preprocess_data(recording)

    # Step 3: Perform Spike Sorting
    sorting = sort_spikes(recording_preprocessed)

    # Step 4: Postprocess Sorting and Compute Quality Metrics
    waveform_extractor = postprocess_sorting(sorting, recording_preprocessed)
    quality_metrics = compute_quality_metrics(waveform_extractor)
    print("Quality Metrics:", quality_metrics)

    # Step 5: Calculate Firing Rates
    firing_rates = calculate_firing_rate(sorting)
    print("Firing Rates (Hz):", firing_rates)

    # Step 6: Analyze Spike Train Correlations
    correlation_matrix = analyze_spike_train_correlation(sorting)
    print("Correlation Matrix:", correlation_matrix)

    # Step 7: Perform Time-Frequency Analysis
    example_spike_train = es.BinnedSpikeTrain(sorting.get_unit_spike_train(sorting.unit_ids[0]) * pq.s, binsize=5 * pq.ms)
    freqs, psd = perform_time_frequency_analysis(example_spike_train)
    print("Power Spectral Density (PSD):", psd)
    print("Frequencies:", freqs)

    # Step 8: Visualize Results
    plot_raster(sorting)
    plot_firing_rate_histogram(firing_rates)
    plot_correlation_matrix(correlation_matrix)

if __name__ == "__main__":
    # Example file path for demonstration purposes
    example_file_path = 'data/sample_data'  # Adjust the path for your dataset
    main(example_file_path)