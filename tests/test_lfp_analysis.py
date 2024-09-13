# test_lfp_analysis.py

import unittest
import numpy as np
import neo
from neo.core import AnalogSignal
import quantities as pq
from lfp_analysis import (
    load_data, preprocess_data, time_frequency_analysis_stft,
    power_spectral_density, coherence_analysis, pac_analysis,
    cfc_analysis, plot_power_spectral_density, plot_coherence, plot_pac
)

class TestLFPAnalysis(unittest.TestCase):

    def setUp(self):
        """
        Setup a mock dataset for testing.
        """
        # Create a mock AnalogSignal using Neo
        sampling_rate = 1000 * pq.Hz
        duration = 10 * pq.s
        times = np.arange(0, duration.rescale(pq.s).magnitude, 1 / sampling_rate.rescale(pq.Hz).magnitude)
        signal = np.sin(2 * np.pi * 10 * times)  # Simulate a 10 Hz sine wave
        self.analog_signal = AnalogSignal(signal, sampling_rate=sampling_rate, units=pq.mV)

    def test_load_data(self):
        """
        Test if the data loading function works correctly with a mock dataset.
        """
        # Assuming the load_data function works with Neo data directly for simplicity in the test
        loaded_data = load_data('data/sample_lfp_data')  # Replace with mock data path
        self.assertIsNotNone(loaded_data)
        self.assertTrue(hasattr(loaded_data, 'get_traces'))
        self.assertGreater(loaded_data.get_traces().shape[0], 0)

    def test_preprocess_data(self):
        """
        Test the preprocessing function with a mock AnalogSignal.
        """
        preprocessed_data = preprocess_data(self.analog_signal, freq_min=1, freq_max=100, notch_freq=50)
        self.assertIsNotNone(preprocessed_data)
        self.assertTrue(hasattr(preprocessed_data, 'get_traces'))
        self.assertEqual(preprocessed_data.get_traces().shape, self.analog_signal.shape, "Shape mismatch after preprocessing")

    def test_time_frequency_analysis_stft(self):
        """
        Test the STFT-based time-frequency analysis function.
        """
        freqs, times, Zxx = time_frequency_analysis_stft(self.analog_signal)
        self.assertEqual(len(freqs), Zxx.shape[0], "Mismatch in frequency bins.")
        self.assertEqual(len(times), Zxx.shape[1], "Mismatch in time bins.")
        self.assertGreater(len(freqs), 0, "No frequency data returned from STFT.")
        self.assertGreater(np.max(np.abs(Zxx)), 0, "STFT result seems incorrect.")

    def test_power_spectral_density(self):
        """
        Test the Power Spectral Density (PSD) calculation function.
        """
        freqs, psd = power_spectral_density(self.analog_signal)
        self.assertEqual(len(freqs), len(psd), "Mismatch between frequency bins and PSD values.")
        self.assertGreater(len(freqs), 0, "No frequency data returned from PSD.")
        self.assertGreater(np.max(psd), 0, "PSD result seems incorrect.")

    def test_coherence_analysis(self):
        """
        Test coherence analysis between two mock signals.
        """
        coherency, freqs_coherence = coherence_analysis(self.analog_signal, self.analog_signal)
        self.assertEqual(len(coherency), len(freqs_coherence), "Mismatch between coherence and frequency bins.")
        self.assertGreater(len(coherency), 0, "No coherence data returned.")
        self.assertTrue(np.all((coherency >= 0) & (coherency <= 1)), "Coherence values should be between 0 and 1.")

    def test_pac_analysis(self):
        """
        Test the PAC analysis function.
        """
        pac_value = pac_analysis(self.analog_signal, (4, 8), (30, 100))
        self.assertIsInstance(pac_value, float, "PAC value should be a float.")
        self.assertGreaterEqual(pac_value, 0, "PAC values are generally non-negative.")

    def test_cfc_analysis(self):
        """
        Test the Cross-Frequency Coupling (CFC) analysis function.
        """
        cfc_result = cfc_analysis(self.analog_signal, (4, 8), (30, 100))  # Example frequencies
        self.assertIsNotNone(cfc_result, "CFC analysis failed.")
        self.assertTrue(cfc_result.size > 0, "CFC result should not be empty.")
        self.assertGreaterEqual(cfc_result.min(), 0, "CFC values should be non-negative.")

    def test_plot_functions(self):
        """
        Test if plot functions execute without errors. 
        These tests will not verify visual correctness but check for runtime errors.
        """
        freqs, psd = power_spectral_density(self.analog_signal)
        try:
            plot_power_spectral_density(freqs, psd)
        except Exception as e:
            self.fail(f"plot_power_spectral_density raised an exception: {e}")

        coherency, freqs_coherence = coherence_analysis(self.analog_signal, self.analog_signal)
        try:
            plot_coherence(freqs_coherence, coherency)
        except Exception as e:
            self.fail(f"plot_coherence raised an exception: {e}")
        
        pac_value = pac_analysis(self.analog_signal, (4, 8), (30, 100))
        try:
            plot_pac(pac_value)
        except Exception as e:
            self.fail(f"plot_pac raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
