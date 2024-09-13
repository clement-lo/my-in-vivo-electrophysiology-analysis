# test_spike_sorting_firing_rate_analysis.py

import unittest
import quantities as pq
import numpy as np
from spike_sorting_firing_rate_analysis import (
    load_data,
    preprocess_data,
    sort_spikes,
    postprocess_sorting,
    compute_quality_metrics,
    calculate_firing_rate,
    analyze_spike_train_correlation,
    perform_time_frequency_analysis,
)

# Example file path for testing purposes
TEST_FILE_PATH = 'data/sample_data'

class TestSpikeSortingFiringRateAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Set up test dependencies and data."""
        # Load the test data
        self.recording = load_data(TEST_FILE_PATH)
        self.recording_preprocessed = preprocess_data(self.recording)
        self.sorting = sort_spikes(self.recording_preprocessed)
        self.waveform_extractor = postprocess_sorting(self.sorting, self.recording_preprocessed)

    def test_load_data(self):
        """Test loading of electrophysiological data."""
        self.assertIsNotNone(self.recording)
        self.assertEqual(len(self.recording.get_channel_ids()), 4)  # Assuming 4 channels in test data

    def test_preprocess_data(self):
        """Test preprocessing of the data."""
        self.assertIsNotNone(self.recording_preprocessed)
        self.assertEqual(len(self.recording_preprocessed.get_channel_ids()), len(self.recording.get_channel_ids()))

    def test_sort_spikes(self):
        """Test spike sorting process."""
        self.assertIsNotNone(self.sorting)
        self.assertGreater(len(self.sorting.get_unit_ids()), 0)  # Ensure some units are detected

    def test_postprocess_sorting(self):
        """Test postprocessing of sorted spikes."""
        self.assertIsNotNone(self.waveform_extractor)
        self.assertGreater(self.waveform_extractor.get_waveforms(unit_id=self.sorting.unit_ids[0]).shape[0], 0)

    def test_compute_quality_metrics(self):
        """Test computation of quality metrics."""
        metrics = compute_quality_metrics(self.waveform_extractor)
        self.assertIsInstance(metrics, dict)
        self.assertIn('snr', metrics[self.sorting.unit_ids[0]])

    def test_calculate_firing_rate(self):
        """Test firing rate calculation."""
        firing_rates = calculate_firing_rate(self.sorting)
        self.assertIsInstance(firing_rates, dict)
        self.assertGreater(firing_rates[self.sorting.unit_ids[0]], 0)

    def test_analyze_spike_train_correlation(self):
        """Test spike train correlation analysis."""
        correlation_matrix = analyze_spike_train_correlation(self.sorting)
        self.assertIsInstance(correlation_matrix, np.ndarray)
        self.assertEqual(correlation_matrix.shape[0], len(self.sorting.unit_ids))

    def test_perform_time_frequency_analysis(self):
        """Test time-frequency analysis."""
        example_spike_train = self.sorting.get_unit_spike_train(self.sorting.unit_ids[0]) * pq.s
        freqs, psd = perform_time_frequency_analysis(example_spike_train)
        self.assertIsInstance(freqs, np.ndarray)
        self.assertIsInstance(psd, np.ndarray)
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(psd), 0)

if __name__ == '__main__':
    unittest.main()