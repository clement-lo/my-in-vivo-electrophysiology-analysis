# In Vivo Electrophysiology Data Analysis

This repository contains Python code and Jupyter notebooks for processing and analyzing in vivo electrophysiology datasets. The project demonstrates automated pipelines for data preprocessing, feature extraction, and statistical analysis.

## Project Overview

In vivo electrophysiology is used to study the electrical properties of living neurons. This repository provides an end-to-end analysis workflow, including:

- **Data Preprocessing:** Noise reduction, filtering, and artifact removal.
- **Feature Extraction:** Calculating spike rates, local field potentials (LFPs), and other key metrics.
- **Data Analysis:** Statistical testing, visualization, and automated reporting.

## Repository Structure

- `data/`: Example datasets and raw data files.
- `notebooks/`: Jupyter notebooks for interactive data exploration.
- `src/`: Python scripts for data processing, feature extraction, and analysis.
- `tests/`: Unit tests to validate the code.
- `.github/`: GitHub Actions configuration for CI.
- `environment.yml`: Conda environment with all dependencies.
- `requirements.txt`: Dependencies list for pip users.
- `.python-version`: Python version file for `pyenv` users.
  
## Getting Started

You have two options for setting up the Python environment: using Conda or `pyenv` with `venv`. Choose the method that works best for you.

### Option 1: Installation Using Conda

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/my-electrophysiology-analysis.git
    cd my-electrophysiology-analysis
    ```

2. Set up the Conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate electrophysiology
    ```

3. Run the analysis:
    ```bash
    python src/analysis.py --input data/example_data.csv
    ```

### Option 2: Installation Using `pyenv` and `venv`

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/my-electrophysiology-analysis.git
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

5. Run the analysis:
    ```bash
    python src/analysis.py --input data/example_data.csv
    ```

## Running the Analysis

1. Start with exploratory data analysis using the Jupyter notebook in `notebooks/`.
2. Execute the processing pipeline:
    ```bash
    python src/analysis.py --input data/example_data.csv
    ```
