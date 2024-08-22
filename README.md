# In Vivo Electrophysiology Data Analysis

This repository contains Python code and Jupyter notebooks for processing and analysing in vivo electrophysiology datasets. It showcases a robust, automated pipeline for data preprocessing, feature extraction, and statistical analysis tailored to the study of neural activity.

## Process Overview

In vivo electrophysiology enables the investigation of electrical properties in living neuronal circuits. This project provides a comprehensive analysis workflow with a focus on reproducibility, modularity, and clarity, covering the following key steps:

- **Data Preprocessing:** Includes noise reduction (bandpass filtering, notch filtering), artifact removal (e.g., movement artifacts), and signal alignment.
- **Feature Extraction:** Automated extraction of relevant metrics like spike rates, inter-spike intervals (ISIs), local field potentials (LFPs), and power spectral densities (PSDs).
- **Data Analysis:** In-depth statistical analysis (e.g., t-tests, ANOVAs, correlation analysis) integrated with visualisation techniques such as raster plots, PSTHs, and time-frequency analyses.
- **Automated Reporting:** Generates customisable summary reports in LaTeX/Markdown format with key figures and metrics.

## Repository Structure

The repository is organised in:

- `data/`: Contains raw and preprocessed datasets. An example dataset is included for demonstration.
- `notebooks/`: Jupyter notebooks providing step-by-step interactive analysis. These include exploratory data analysis (EDA), signal processing, and statistical testing workflows.
- `src/`: Modular Python scripts for pipeline automation:
    - `preprocessing.py`: Handles data cleaning, filtering, and alignment.
    - `feature_extraction.py`: Extracts spikes, LFPs, and other relevant features.
    - `analysis.py`: Performs statistical analysis and generates visualisations.
    - `reporting.py`: Compiles analysis results into summary reports.
- `tests/`: Unit tests to validate the functionality of key modules, ensuring code reliability.
- `.github/`: GitHub Actions for continuous integration (CI), ensuring code quality and functionality.
- `environment.yml`: Conda environment file listing all dependencies for consistent setup.
- `requirements.txt`: Dependencies list for pip users.
- `.python-version`: Specifies the Python version for users leveraging `pyenv`.

## Getting Started

This project supports Conda or `pyenv` + `venv`. Follow the steps below based on your preferred method.

### Option 1: Installation Using Conda

1. Clone the repository:
    ```bash
    git clone https://github.com/clement-lo/my-electrophysiology-analysis.git
    cd my-electrophysiology-analysis
    ```

2. Set up the Conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate electrophysiology
    ```

3. Run the preprocessing and analysis pipeline:
    ```bash
    python src/preprocessing.py --input data/example_data.csv
    python src/analysis.py --input data/preprocessed_data.csv
    ```

### Option 2: Installation Using `pyenv` and `venv`

1. Clone the repository:
    ```bash
    git clone https://github.com/clement-lo/my-electrophysiology-analysis.git
    cd my-electrophysiology-analysis
    ```

2. Install the required Python version using `pyenv`:
    ```bash
    pyenv install 3.8.12
    pyenv local 3.8.12
    ```

3. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/Mac
    .venv\Scripts\activate     # On Windows
    ```

4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5. Run the preprocessing and analysis pipeline:
    ```bash
    python src/preprocessing.py --input data/example_data.csv
    python src/analysis.py --input data/preprocessed_data.csv
    ```

## Running the Analysis

1. Begin with exploratory data analysis using the Jupyter notebooks in `notebooks/`. This is useful for understanding the dataset, visualising key features, and identifying artifacts

   
2. Execute the preprocessing pipeline:
    ```bash
    python src/preprocessing.py --input data/example_data.csv
    ```
4. Run the analysis script:
    ```bash
    python src/analysis.py --input data/preprocessed_data.csv
    ```
5. Visualise data and generate summary:
    ```bash
    python src/visualise.py --input results/ --output report.md
    ```
