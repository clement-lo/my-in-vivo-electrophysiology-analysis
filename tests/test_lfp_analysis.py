# test_lfp_analysis.py

import unittest
import numpy as np
import neo
from neo.core import AnalogSignal
import quantities as pq
from lfp_analysis import (
    load_data, preprocess_data, time_frequency_analysis, 
    coherence_analysis, pac_analysis
)

class TestLFPAnalysis(unittest.TestCase):

    def setUp(self):
        """
        Setup a mock dataset for testing.
        """
        # Create a mock AnalogSignal using Neo
        sampling_rate = 1000 * pq.Hz
        duration = 10 * pq.s
        times = np.arange(0, duration.rescale(pq.s).magnitude, 1/sampling_rate.rescale(pq.Hz).magnitude)
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

    def test_preprocess_data(self):
        """
        Test the preprocessing function with a mock AnalogSignal.
        """
        preprocessed_data = preprocess_data(self.analog_signal, freq_min=1, freq_max=100, notch_freq=50)
        self.assertIsNotNone(preprocessed_data)
        self.assertTrue(hasattr(preprocessed_data, 'get_traces'))

    def test_time_frequency_analysis(self):
        """
        Test the time-frequency analysis function.
        """
        freqs, psd = time_frequency_analysis(self.analog_signal)
        self.assertEqual(len(freqs), len(psd))
        self.assertGreater(len(freqs), 0)
        self.assertGreater(np.max(psd), 0)

    def test_coherence_analysis(self):
        """
        Test coherence analysis between two mock signals.
        """
        coherency, freqs_coherence = coherence_analysis(self.analog_signal, self.analog_signal)
        self.assertEqual(len(coherency), len(freqs_coherence))
        self.assertGreater(len(coherency), 0)
        self.assertTrue(np.all((coherency >= 0) & (coherency <= 1)))

    def test_pac_analysis(self):
        """
        Test the PAC analysis function.
        """
        pac_value = pac_analysis(self.analog_signal, (4, 8), (30, 100))
        self.assertIsInstance(pac_value, float)
        self.assertGreaterEqual(pac_value, 0)  # PAC values are generally non-negative

    def test_plot_functions(self):
        """
        Test if plot functions execute without errors. 
        These tests will not verify visual correctness.
        """
        freqs, psd = time_frequency_analysis(self.analog_signal)
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