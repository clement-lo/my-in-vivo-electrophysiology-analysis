# lfp_analysis.py

# Import necessary libraries
import neo  # For data handling
import spikeinterface as si  # Core module for SpikeInterface
import spikeinterface.extractors as se  # For data loading and extraction
import spikeinterface.preprocessing as sp  # For data preprocessing
import elephant  # For advanced LFP analysis
import elephant.spectral as espt  # For spectral analysis
import elephant.coherence as ecoh  # For coherence analysis
import elephant.phase_analysis as ephase  # For PAC/CFC analysis
import quantities as pq  # For unit handling
import matplotlib.pyplot as plt  # For static visualization
import plotly.graph_objects as go  # For interactive visualization with Plotly
import numpy as np  # For numerical operations
from neo.io import NeuralynxIO  # Example IO for Neo data loading

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
def preprocess_data(recording, freq_min=1, freq_max=100, notch_freq=None):
    """
    Preprocess the loaded data by applying bandpass filtering and optional notch filtering.
    
    Args:
    - recording (si.BaseRecording): Loaded data in SpikeInterface's RecordingExtractor format.
    - freq_min (float): Minimum frequency for bandpass filter.
    - freq_max (float): Maximum frequency for bandpass filter.
    - notch_freq (float): Frequency for notch filter to remove powerline noise. If None, skip.
    
    Returns:
    - recording_preprocessed (si.BaseRecording): Preprocessed LFP data.
    """
    # Bandpass filter for LFP
    recording_bp = sp.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
    
    # Optional notch filter
    if notch_freq:
        recording_notch = sp.notch_filter(recording_bp, freq=notch_freq)
        return recording_notch
    return recording_bp

# 3. Time-Frequency Analysis Module
def time_frequency_analysis(analog_signal, fs=1000):
    """
    Perform time-frequency analysis using spectral methods.
    
    Args:
    - analog_signal (neo.AnalogSignal): LFP data in Neo's AnalogSignal format.
    - fs (int): Sampling frequency.
    
    Returns:
    - freqs (np.ndarray): Frequency bins.
    - psd (np.ndarray): Power spectral density.
    """
    psd, freqs = espt.welch_psd(analog_signal, fs=fs)
    return freqs, psd

# 4. Coherence Analysis Module
def coherence_analysis(analog_signal1, analog_signal2):
    """
    Assess coherence between two LFP signals.
    
    Args:
    - analog_signal1 (neo.AnalogSignal): First LFP signal.
    - analog_signal2 (neo.AnalogSignal): Second LFP signal.
    
    Returns:
    - coherency (np.ndarray): Coherence values.
    - freqs (np.ndarray): Frequency bins.
    """
    coherency, freqs = ecoh.coherence(analog_signal1, analog_signal2)
    return coherency, freqs

# 5. Phase-Amplitude Coupling (PAC) Analysis Module
def pac_analysis(analog_signal, low_freq, high_freq):
    """
    Investigate Phase-Amplitude Coupling (PAC) in LFP signals.
    
    Args:
    - analog_signal (neo.AnalogSignal): LFP data in Neo's AnalogSignal format.
    - low_freq (tuple): Low-frequency range for phase extraction.
    - high_freq (tuple): High-frequency range for amplitude extraction.
    
    Returns:
    - pac (float): Modulation index (MI) for PAC.
    """
    pac = ephase.phase_amplitude_coupling(analog_signal, low_freq, high_freq)
    return pac

# 6. Visualization Module
def plot_power_spectral_density(freqs, psd):
    """
    Plot power spectral density using Matplotlib.
    
    Args:
    - freqs (np.ndarray): Frequency bins.
    - psd (np.ndarray): Power spectral density.
    """
    plt.figure()
    plt.semilogy(freqs, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Power Spectral Density of LFP')
    plt.show()

def plot_coherence(freqs, coherency):
    """
    Plot coherence between two LFP signals.
    
    Args:
    - freqs (np.ndarray): Frequency bins.
    - coherency (np.ndarray): Coherence values.
    """
    plt.figure()
    plt.plot(freqs, coherency)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.title('Coherence Analysis')
    plt.show()

def plot_pac(pac):
    """
    Visualize Phase-Amplitude Coupling (PAC) using Plotly.
    
    Args:
    - pac (float): Modulation index for PAC.
    """
    fig = go.Figure(data=go.Heatmap(z=pac))
    fig.update_layout(title="Phase-Amplitude Coupling (PAC)", xaxis_title="Phase Frequency (Hz)", yaxis_title="Amplitude Frequency (Hz)")
    fig.show()

# Main function
def main(file_path):
    """
    Main function to perform LFP analysis.
    
    Args:
    - file_path (str): Path to the data file.
    """
    # Step 1: Load Data
    recording = load_data(file_path)
    
    # Step 2: Preprocess Data
    recording_preprocessed = preprocess_data(recording, freq_min=1, freq_max=100, notch_freq=50)

    # Step 3: Perform Time-Frequency Analysis
    analog_signal = recording_preprocessed.get_traces().T
    freqs, psd = time_frequency_analysis(analog_signal)
    plot_power_spectral_density(freqs, psd)

    # Step 4: Coherence Analysis
    coherency, freqs_coherence = coherence_analysis(analog_signal, analog_signal)  # Example with the same signal
    plot_coherence(freqs_coherence, coherency)

    # Step 5: PAC Analysis
    pac = pac_analysis(analog_signal, (4, 8), (30, 100))  # Example frequencies for phase and amplitude
    plot_pac(pac)

if __name__ == "__main__":
    # Example file path for demonstration purposes
    example_file_path = 'data/sample_lfp_data'  # Adjust the path for your dataset
    main(example_file_path)