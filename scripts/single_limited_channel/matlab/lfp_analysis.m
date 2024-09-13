% lfp_analysis.m

%% Local Field Potential (LFP) Analysis
% This script performs LFP analysis using MATLAB, including:
% - Time-Frequency Analysis: STFT, Wavelet Transform
% - Coherence Analysis
% - Phase-Amplitude Coupling (PAC)
% - Cross-Frequency Coupling (CFC)
% - Visualizations for each analysis

%% 1. Load Data
function [data, fs] = load_data(file_path)
    % Load electrophysiological data using MATLAB's built-in functions
    loaded_data = load(file_path);
    % Ensure the actual variable names match your data structure
    data = loaded_data.lfp_data;  % Replace with your data variable name
    fs = loaded_data.fs;  % Replace with your sampling frequency variable name
end

%% 2. Preprocess Data
function data_filtered = preprocess_data(data, fs, freq_min, freq_max, notch_freq)
    % Preprocess the loaded data by applying bandpass and notch filters.
    [b, a] = butter(2, [freq_min, freq_max] / (fs / 2), 'bandpass');
    data_filtered = filtfilt(b, a, data);

    % Optional notch filter
    if ~isempty(notch_freq)
        [b_notch, a_notch] = iirnotch(notch_freq / (fs / 2), notch_freq / (fs * 35));
        data_filtered = filtfilt(b_notch, a_notch, data_filtered);
    end
end

%% 3. Time-Frequency Analysis (STFT)
function [S, F, T] = time_frequency_analysis_stft(data, fs, nperseg)
    % Perform time-frequency analysis using Short-Time Fourier Transform (STFT).
    window = hann(nperseg);
    overlap = round(nperseg * 0.5);
    [S, F, T] = stft(data, fs, 'Window', window, 'OverlapLength', overlap, 'FFTLength', nperseg);
    S = abs(S);
end

%% 4. Time-Frequency Analysis (Wavelet Transform)
function [cfs, freqs, T] = time_frequency_analysis_wavelet(data, fs)
    % Perform time-frequency analysis using Wavelet Transform.
    scales = 1:1:128;  % Example scales
    [cfs, freqs] = cwt(data, 'amor', fs, 'VoicesPerOctave', 12);
    T = (0:length(data)-1) / fs;  % Time vector corresponding to the data length
end

%% 5. Coherence Analysis
function [coherency, freqs] = coherence_analysis(data1, data2, fs)
    [coherency, freqs] = mscohere(data1, data2, [], [], [], fs);
end

%% 6. Phase-Amplitude Coupling (PAC) Analysis
function mi = pac_analysis(data, fs, low_freq, high_freq)
    % Investigate Phase-Amplitude Coupling (PAC) in LFP signals.
    low_freq_band = bandpass(data, low_freq, fs);
    phase_data = angle(hilbert(low_freq_band));
    high_freq_band = bandpass(data, high_freq, fs);
    amplitude_data = abs(hilbert(high_freq_band));
    % Use standard PAC analysis method, such as Modulation Index by Tort et al.
    % (Example simplified calculation here)
    mi = mean(amplitude_data .* exp(1j * phase_data));
end

%% 7. Cross-Frequency Coupling (CFC) Analysis
function cfc_matrix = cfc_analysis(data, fs, phase_freqs, amplitude_freqs)
    num_phase_freqs = length(phase_freqs);
    num_amplitude_freqs = length(amplitude_freqs);
    cfc_matrix = zeros(num_phase_freqs, num_amplitude_freqs);
    for i = 1:num_phase_freqs
        for j = 1:num_amplitude_freqs
            cfc_matrix(i, j) = pac_analysis(data, fs, phase_freqs(i,:), amplitude_freqs(j,:));
        end
    end
end

%% 8. Visualization Functions
function plot_psd(freqs, psd)
    figure;
    semilogy(freqs, psd);
    xlabel('Frequency (Hz)');
    ylabel('Power Spectral Density');
    title('Power Spectral Density of LFP');
end

function plot_coherence(freqs, coherency)
    figure;
    plot(freqs, coherency);
    xlabel('Frequency (Hz)');
    ylabel('Coherence');
    title('Coherence Analysis');
end

function plot_wavelet_transform(T, F, cfs)
    figure;
    imagesc(T, F, abs(cfs)); axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Wavelet Transform');
    colorbar;
end

%% Main Function
function main(file_path)
    [data, fs] = load_data(file_path);
    data_filtered = preprocess_data(data, fs, 1, 100, 50);
    [S, F, T] = time_frequency_analysis_stft(data_filtered, fs, 256);
    figure; imagesc(T, F, abs(S)); axis xy; title('STFT Magnitude'); xlabel('Time (s)'); ylabel('Frequency (Hz)');
    [cfs, freqs, T] = time_frequency_analysis_wavelet(data_filtered, fs);
    plot_wavelet_transform(T, freqs, cfs);
    [coherency, freqs_coherence] = coherence_analysis(data_filtered, data_filtered, fs);
    plot_coherence(freqs_coherence, coherency);
    mi = pac_analysis(data_filtered, fs, [4 8], [30 100]);
    disp(['PAC Modulation Index: ', num2str(mi)]);
    phase_freqs = [4, 8; 8, 12];
    amplitude_freqs = [30, 50; 50, 80];
    cfc_matrix = cfc_analysis(data_filtered, fs, phase_freqs, amplitude_freqs);
    disp('CFC Analysis Result:'); disp(cfc_matrix);
end

% Run the main function with an example file path
main('data/sample_lfp_data.mat');  % Adjust the path to your dataset