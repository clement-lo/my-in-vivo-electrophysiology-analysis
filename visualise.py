import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_data(file_path):
    """
    Load data from a .npy file.

    Parameters:
    file_path (str): Path to the input file.

    Returns:
    np.ndarray: Loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return np.load(file_path)

def plot_raster(spike_times, spike_labels, num_neurons, output_path):
    """
    Generate and save a raster plot.

    Parameters:
    spike_times (np.ndarray): Spike times (indices).
    spike_labels (np.ndarray): Labels corresponding to different neurons.
    num_neurons (int): Number of neurons to display.
    output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 6))
    for neuron_id in range(num_neurons):
        neuron_spike_times = spike_times[spike_labels == neuron_id]
        plt.scatter(neuron_spike_times / 20000.0, np.ones_like(neuron_spike_times) * neuron_id, s=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron ID')
    plt.title('Raster Plot')
    plt.savefig(output_path)
    plt.close()

def plot_psth(spike_times, spike_labels, stimulus_times, window, bin_size, output_path):
    """
    Generate and save a peri-stimulus time histogram (PSTH).

    Parameters:
    spike_times (np.ndarray): Spike times (indices).
    spike_labels (np.ndarray): Labels corresponding to different neurons.
    stimulus_times (np.ndarray): Times of stimulus events.
    window (tuple): Time window around the stimulus (in seconds).
    bin_size (float): Size of bins for the histogram (in seconds).
    output_path (str): Path to save the plot.
    """
    fs = 20000  # Assuming 20 kHz sampling rate
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    psth = np.zeros(len(bins) - 1)

    for stim_time in stimulus_times:
        for neuron_id in range(np.max(spike_labels) + 1):
            neuron_spikes = spike_times[(spike_labels == neuron_id) & (spike_times > stim_time + window[0] * fs) & (spike_times < stim_time + window[1] * fs)]
            aligned_spikes = (neuron_spikes - stim_time) / fs
            psth += np.histogram(aligned_spikes, bins=bins)[0]

    psth = psth / (len(stimulus_times) * bin_size)  # Normalize by number of stimuli and bin size

    plt.figure(figsize=(12, 6))
    plt.bar(bins[:-1], psth, width=bin_size)
    plt.xlabel('Time (s) relative to stimulus')
    plt.ylabel('Spike Rate (Hz)')
    plt.title('Peri-Stimulus Time Histogram (PSTH)')
    plt.savefig(output_path)
    plt.close()

def plot_erp(lfp_data, stimulus_times, window, output_path):
    """
    Generate and save an event-related potential (ERP) plot.

    Parameters:
    lfp_data (np.ndarray): LFP data.
    stimulus_times (np.ndarray): Times of stimulus events.
    window (tuple): Time window around the stimulus (in seconds).
    output_path (str): Path to save the plot.
    """
    fs = 20000  # Assuming 20 kHz sampling rate
    num_samples = int((window[1] - window[0]) * fs)
    erp = np.zeros(num_samples)

    for stim_time in stimulus_times:
        start_idx = int(stim_time + window[0] * fs)
        end_idx = start_idx + num_samples
        erp += lfp_data[start_idx:end_idx]

    erp /= len(stimulus_times)
    time = np.linspace(window[0], window[1], num_samples)

    plt.figure(figsize=(12, 6))
    plt.plot(time, erp)
    plt.xlabel('Time (s) relative to stimulus')
    plt.ylabel('Potential (µV)')
    plt.title('Event-Related Potential (ERP)')
    plt.savefig(output_path)
    plt.close()

def plot_spectrogram(lfp_data, output_path):
    """
    Generate and save a spectrogram.

    Parameters:
    lfp_data (np.ndarray): LFP data.
    output_path (str): Path to save the plot.
    """
    fs = 20000  # Assuming 20 kHz sampling rate
    frequencies, times, Sxx = spectrogram(lfp_data, fs=fs, nperseg=1024)

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Spectrogram (Time-Frequency Analysis)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim([0, 100])
    plt.colorbar(label='Power (dB)')
    plt.savefig(output_path)
    plt.close()

def main(args):
    # Load the required data
    spike_times = load_data(args.spike_times)
    spike_labels = load_data(args.spike_labels)
    lfp_data = load_data(args.lfp_data)
    stimulus_times = load_data(args.stimulus_times)

    # Generate and save visualizations
    plot_raster(spike_times, spike_labels, args.num_neurons, os.path.join(args.output_dir, 'raster_plot.png'))
    plot_psth(spike_times, spike_labels, stimulus_times, args.window, args.bin_size, os.path.join(args.output_dir, 'psth.png'))
    plot_erp(lfp_data, stimulus_times, args.window, os.path.join(args.output_dir, 'erp.png'))
    plot_spectrogram(lfp_data, os.path.join(args.output_dir, 'spectrogram.png'))

    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations for electrophysiology data (raster plots, PSTH, ERP, and spectrogram).")
    parser.add_argument("--spike_times", type=str, required=True, help="Path to the spike times .npy file")
    parser.add_argument("--spike_labels", type=str, required=True, help="Path to the spike labels .npy file")
    parser.add_argument("--lfp_data", type=str, required=True, help="Path to the LFP data .npy file")
    parser.add_argument("--stimulus_times", type=str, required=True, help="Path to the stimulus times .npy file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the plots")
    parser.add_argument("--num_neurons", type=int, default=5, help="Number of neurons to display in the raster plot")
    parser.add_argument("--window", type=float, nargs=2, default=(-0.1, 0.5), help="Time window around stimulus for PSTH and ERP (in seconds)")
    parser.add_argument("--bin_size", type=float, default=0.01, help="Bin size for PSTH (in seconds)")
    args = parser.parse_args()

    main(args)
