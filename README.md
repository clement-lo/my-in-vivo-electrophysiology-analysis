# My In Vivo Electrophysiology Analysis Repository

## Overview

This repository provides a comprehensive toolkit for analyzing **in vivo electrophysiology** data, focusing on both **classic electrophysiology setups** (e.g., single-channel recordings) and **multi-electrode array (MEA) setups**. The repository integrates well-established libraries like **Neo**, **Elephant**, **SpikeInterface**, and **NeuroChaT**, offering robust support for a wide range of data formats and advanced analytical methods. It is designed to support researchers and analysts in neuroscience with modular, flexible, and extensible tools for data analysis, visualization, and interpretation.

## Table of Contents
[Repository Structure](#repository-structure)
[Prerequisites](#prerequisites)
[Installation](#installation)
[Data Formats and Handling](#data-formats-and-handling)
[Analysis Types](#analysis-types)
[Classic Electrophysiology Setups](#classic-electrophysiology-setups)
[Spike Sorting and Firing Rate Analysis](#spike-sorting-and-firing-rate-analysis)
[Local Field Potential (LFP) Analysis](#local-field-potential-lfp-analysis)
[Single-Unit and Multi-Unit Activity Analysis](#single-unit-and-multi-unit-activity-analysis)
[Multi-Electrode Array (MEA) Setups](#multi-electrode-array-mea-setups)
[Advanced Spike Sorting and Clustering](#advanced-spike-sorting-and-clustering)
[Functional Connectivity Analysis](#functional-connectivity-analysis)
[Effective Connectivity and Causal Inference](#effective-connectivity-and-causal-inference)
[Phase-Amplitude Coupling (PAC) and Cross-Frequency Coupling (CFC)](#phase-amplitude-coupling-pac-and-cross-frequency-coupling-cfc)
[Network Graph Analysis and Community Detection](#network-graph-analysis-and-community-detection)
[Burst Detection and Oscillatory Analysis](#burst-detection-and-oscillatory-analysis)
[Modularity and Extensibility](#modularity-and-extensibility)
[Integration with Established Libraries](#integration-with-established-libraries)
[Advanced Visualization and Interactive Notebooks](#advanced-visualization-and-interactive-notebooks)
[Unit Testing and Continuous Integration (CI)](#unit-testing-and-continuous-integration-ci)
[Example Datasets and Detailed Workflows](#example-datasets-and-detailed-workflows)
[License and Citation](#license-and-citation)
[References](#references)

## Repository Structure
```plaintext
├── README.md                                     # Main documentation file
├── requirements.txt                              # Python dependencies for the project
├── data/                                         # Directory for storing raw and preprocessed data
├── results/                                      # Output directory for results from analyses
├── scripts/                                      # Main directory containing all the analysis scripts
│   ├── single_limited_channel/                   # Directory for single-channel or limited-channel analyses
│   │   ├── python/                               # Python scripts for single/limited channel analyses
│   │   │   ├── spike_sorting_firing_rate_analysis.py
│   │   │   ├── lfp_analysis.py
│   │   │   ├── single_multi_unit_activity_analysis.py
│   │   ├── matlab/                               # MATLAB scripts for single/limited channel analyses
│   │   │   ├── spike_sorting_firing_rate_analysis.m
│   │   │   ├── lfp_analysis.m
│   │   │   ├── single_multi_unit_activity_analysis.m
│   │   ├── notebooks/                            # Jupyter Notebooks for single/limited channel analyses
│   │   │   ├── 01_Spike_Sorting_Firing_Rate_Analysis.ipynb
│   │   │   ├── 02_LFP_Analysis.ipynb
│   │   │   ├── 03_Single_Multi_Unit_Activity_Analysis.ipynb
│   ├── multi_electrode_array/                    # Directory for multi-electrode array analyses
│   │   ├── python/                               # Python scripts for MEA analyses
│   │   │   ├── advanced_spike_sorting_clustering.py
│   │   │   ├── functional_connectivity_analysis.py
│   │   │   ├── effective_connectivity_causal_inference.py
│   │   │   ├── pac_cfc_analysis.py
│   │   │   ├── network_graph_analysis.py
│   │   │   ├── burst_detection_oscillatory_analysis.py
│   │   ├── matlab/                               # MATLAB scripts for MEA analyses
│   │   │   ├── advanced_spike_sorting_clustering.m
│   │   │   ├── functional_connectivity_analysis.m
│   │   │   ├── effective_connectivity_causal_inference.m
│   │   │   ├── pac_cfc_analysis.m
│   │   │   ├── network_graph_analysis.m
│   │   │   ├── burst_detection_oscillatory_analysis.m
│   │   ├── notebooks/                            # Jupyter Notebooks for MEA analyses
│   │   │   ├── 01_Advanced_Spike_Sorting_Clustering.ipynb
│   │   │   ├── 02_Functional_Connectivity_Analysis.ipynb
│   │   │   ├── 03_Effective_Connectivity_Causal_Inference.ipynb
│   │   │   ├── 04_PAC_CFC_Analysis.ipynb
│   │   │   ├── 05_Network_Graph_Analysis.ipynb
│   │   │   ├── 06_Burst_Detection_Oscillatory_Analysis.ipynb
├── tests/                                        # Unit tests for the analysis scripts
│   ├── test_spike_sorting_firing_rate_analysis.py
│   ├── test_lfp_analysis.py
│   ├── test_single_multi_unit_activity_analysis.py
│   ├── test_advanced_spike_sorting_clustering.py
│   ├── test_functional_connectivity_analysis.py
│   ├── test_effective_connectivity_causal_inference.py
│   ├── test_pac_cfc_analysis.py
│   ├── test_network_graph_analysis.py
│   ├── test_burst_detection_oscillatory_analysis.py
├── examples/                                     # Example datasets and workflows
│   ├── example_data.csv
│   └── example_workflow.ipynb
├── CONTRIBUTING.md                               # Guidelines for contributing to the repository
└── LICENSE.md                                    # Licensing information
```
## Prerequisites
Ensure you have the following installed on your system:

- Python 3.10.14 or higher
- Jupyter Notebook
- MATLAB (optional, for some analyses)
- Git (for version control)
- Libraries: Neo, Elephant, SpikeInterface, Matplotlib, NumPy, SciPy

## Installation
To get started with this repository, follow the steps below:
Clone the Repository: Open Terminal (Mac) or Command Prompt (Windows), and run:

```bash
git clone https://github.com/clement-lo/my-in-vivo-electrophysiology-analysis.git
cd my-in-vivo-electrophysiology-analysis
Create a Virtual Environment:
```
```bash
python3 -m venv env
source env/bin/activate  # For Mac/Linux
.\env\Scripts\activate  # For Windows
Install Required Python Libraries:
```
```bash
pip install -r requirements.txt
```
Verify Installation: 
``` bash
Run python --version and pip list to confirm Python and required packages are installed correctly.
```

## Data Formats and Handling
This repository supports various data formats commonly used in in vivo electrophysiology research, including:
Neo Data Format: Supported by libraries like Neo and SpikeInterface for standardized data handling.
CSV and MATLAB Files: For users with custom data formats, we provide functions to load and preprocess these data types.
Ensure your data is properly formatted and organized in the data/ directory before running any analysis.

## Analysis Types
This repository is divided into **two** main types of analyses:

### 1. Classic Electrophysiology Setups
Classic electrophysiology setups focus on single-channel or limited-channel recordings. These setups are common in traditional in vivo studies, where recordings are made using electrodes positioned in specific brain areas to capture neuronal activity.

#### a. Spike Sorting and Firing Rate Analysis
- Objective: Detect and sort spikes from raw electrophysiological data, compute firing rates, analyze spike train dynamics, and visualize spike trains.
- Methods:
- - Spike Detection: Threshold-based and template matching approaches to detect spikes in continuous signals.
- - Feature Extraction: Extract key features like spike width, amplitude, and waveform shape for clustering.
- - Clustering Algorithms: Methods like K-means, Gaussian Mixture Models (GMM), and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) are used for sorting spikes into units.
- - Spike Train Analysis: Compute interspike intervals (ISI), peri-stimulus time histograms (PSTH), and burst detection.
- Tools: Python, MATLAB, SpikeInterface (SpikeInterface on GitHub), Neo (Neo on GitHub), Elephant (Elephant on GitHub).
- Outcome: Sorted spikes, firing rate histograms, raster plots, and autocorrelograms.
#### b. Local Field Potential (LFP) Analysis
- Objective: Analyze local field potentials to understand neural oscillations, rhythms, and interactions between brain regions.
- Methods:
- - Time-Frequency Analysis: Short-Time Fourier Transform (STFT), Wavelet Transforms to study power across different frequency bands (delta, theta, alpha, beta, gamma).
- - Coherence Analysis: Assess coherence between LFP signals recorded from different brain regions.
- - Phase-Amplitude Coupling (PAC): Investigate the coupling between the phase of low-frequency oscillations and the amplitude of high-frequency oscillations.
- - Cross-Frequency Coupling (CFC): Measure interactions between oscillatory activities at different frequencies.
- Tools: Python, MATLAB, Elephant, MNE-Python (MNE-Python on GitHub), PyWavelets (PyWavelets on GitHub).
- Outcome: Power spectral densities (PSD), spectrograms, coherence plots, phase-amplitude coupling indices.
#### c. Single-Unit and Multi-Unit Activity Analysis
- Objective: Analyze single-unit and multi-unit activity (SUA/MUA) to understand neuronal coding and information processing.
- Methods:
- - Single-Unit Isolation: Isolate and analyze activity from single neurons.
- - Multi-Unit Activity Analysis: Aggregate spikes from all recorded neurons to analyze general network activity.
- - Spike-Triggered Averaging: Align spikes to external stimuli to analyze sensory processing or motor control.
- Tools: Python, MATLAB, SpikeInterface, Neo.
- Outcome: Tuning curves, receptive fields, neuronal response profiles.

### 2. Multi-Electrode Array (MEA) Setups
Multi-electrode array (MEA) setups are designed for high-density recordings, allowing for more comprehensive analysis of network-level dynamics. This section provides a more advanced analysis to capture complex brain activity patterns.

#### a. Advanced Spike Sorting and Clustering
- Objective: Perform advanced spike sorting and clustering for MEA recordings to analyze network activity.
- Methods:
- - Template Matching and Subspace Projections: Use template matching and subspace projections to classify spikes from densely packed electrodes.
- - Dimensionality Reduction Techniques: PCA, t-SNE, and UMAP for visualizing high-dimensional spike features.
- - Spike Clustering Algorithms: HDBSCAN, Kilosort (Kilosort on GitHub), IronClust (IronClust on GitHub), and MountainSort (MountainSort on GitHub) for clustering spikes across electrodes.
- Tools: Python, MATLAB, SpikeInterface, Kilosort, MountainSort, IronClust.
- Outcome: Clustered spikes, spike train cross-correlograms, and sorted spike trains.
#### b. Functional Connectivity Analysis
- Objective: Analyze functional connectivity between neurons recorded from MEA setups to understand network communication and dynamics.
- Methods:
- - Cross-Correlation and Joint Peri-Stimulus Time Histogram (JPSTH): Measure correlation between spike trains to infer connectivity.
- - Granger Causality and Directed Transfer Function (DTF): Identify directional interactions between neurons.
- - Partial Directed Coherence (PDC) and Coherence Analysis: Study synchronization and communication between neural populations.
- Tools: Python, MATLAB, Elephant, MNE-Python.
- Outcome: Functional connectivity matrices, Granger causality graphs, and cross-correlograms.
#### c. Effective Connectivity and Causal Inference
- Objective: Investigate causal relationships and effective connectivity using directed graph models.
- Methods:
- - Dynamic Causal Modeling (DCM) and Transfer Entropy (TE): Infer the direction and strength of influence between neurons.
- - Generalized Linear Models (GLM) and Bayesian Networks: Statistical models for neural connectivity analysis.
- Tools: Python, MATLAB, Elephant, PyMC3 (PyMC3 on GitHub), statsmodels.
- Outcome: Directed connectivity graphs, transfer entropy values, causal interaction plots.
#### d. Phase-Amplitude Coupling (PAC) and Cross-Frequency Coupling (CFC)
- Objective: Measure phase-amplitude coupling (PAC) and cross-frequency coupling (CFC) to understand neural synchronization and communication.
- Methods:
- - Hilbert Transform: Extract phase and amplitude information for coupling analysis.
- - Modulation Index (MI) and Phase Locking Value (PLV): Quantify the strength of coupling.
- - Wavelet Coherence: Analyze coherence between signals across frequency bands.
- Tools: Python, MATLAB, NeuroChaT (NeuroChaT on GitHub), Elephant.
- Outcome: PAC plots, cross-frequency coupling matrices, coherence spectra.
#### e. Network Graph Analysis and Community Detection
- Objective: Model the brain as a network and analyze its structure, dynamics, and modular organization.
- Methods:
- - Graph Theory Metrics: Degree, betweenness, closeness centrality, clustering coefficient.
- - Community Detection Algorithms: Louvain, Leiden, and Infomap algorithms for detecting clusters or communities in brain networks.
- - Network Dynamics Analysis: Measure network robustness, small-world properties, and motif analysis.
- Tools: Python, NetworkX (NetworkX on GitHub), Neo, Brain Connectivity Toolbox (BCT).
- Outcome: Network graphs, community structures, connectivity heatmaps, centrality plots.
#### f. Burst Detection and Oscillatory Analysis
- Objective: Detect bursts of activity and analyze oscillatory phenomena in multi-electrode array recordings.
- Methods:
- - Burst Detection Algorithms: Poisson surprise, log-surprise, interval method.
- - Oscillatory Dynamics: Analyze oscillatory patterns using wavelet transforms, band-pass filtering, and Hilbert transform.
- Tools: Python, MATLAB, NeuroChaT, Elephant.
- Outcome: Burst raster plots, oscillatory power spectra, burst duration histograms.
## Modularity and Extensibility
The repository is designed to be modular, allowing for easy extension by adding new modules for specific analyses. Each type of analysis is encapsulated in its own script or function to enable reuse and extension. Contributing new analysis modules or extending existing ones can be done by following the structure and guidelines provided.

## Integration with Established Libraries
The repository integrates popular libraries such as:
- Neo: For standardized data handling and storage.
- Elephant: For advanced time-frequency analysis and spike train analysis.
- SpikeInterface: For efficient spike sorting and integration with multiple spike sorting algorithms.
- NeuroChaT: For in vivo spike train analysis and phase-locking analysis.
- Advanced Visualization and Interactive Notebooks
This repository provide Jupyter Notebooks for each analysis type, offering step-by-step guidance and interactive visualizations. Notebooks allow users to explore data interactively, modify parameters, and visualize results dynamically.

## Unit Testing and Continuous Integration (CI)
Unit tests are provided for each analysis script to ensure robustness and reliability. We recommend using GitHub Actions for Continuous Integration (CI) to run these tests automatically upon code updates.

## Example Datasets and Detailed Workflows
Example datasets and workflows are included to guide users through data preprocessing, analysis, and interpretation. These examples are located in the examples/ directory and provide end-to-end tutorials for various analyses.

## License and Citation
This repository is available under the GNU General Public License v3.0 (GPL-3.0) for academic use. For commercial use, please contact me for licensing information.

## References
- [Neo](https://github.com/NeuralEnsemble/python-neo) 
- [Elephant](https://github.com/NeuralEnsemble/elephant)
- [SpikeInterface](https://github.com/SpikeInterface/spikeinterface)
- [NeuroChaT](https://github.com/shanemomara/NeuroChaT)
- [Kilosort](https://github.com/MouseLand/Kilosort)
- [IronClust](https://github.com/flatironinstitute/ironclust)
- [MountainSort](https://github.com/flatironinstitute/mountainsort)
- [MNE-Python](https://github.com/mne-tools/mne-python)
- [PyWavelets](https://github.com/PyWavelets/pywt)
- [PyMC3](https://github.com/pymc-devs/pymc3)
- [NetworkX](https://github.com/networkx/networkx)