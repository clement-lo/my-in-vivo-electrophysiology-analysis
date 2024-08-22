# In Vivo Electrophysiology Data Analysis

This repository contains Python code and Jupyter notebooks for processing and analysing in vivo electrophysiology datasets. It showcases a robust, automated pipeline for data preprocessing, feature extraction, and statistical analysis tailored to the study of neural activity.

## Process Overview

In vivo electrophysiology enables the investigation of electrical properties in living neuronal circuits. This project provides a comprehensive analysis workflow with a focus on reproducibility, modularity, and clarity, covering the following key steps:

- **Data Preprocessing:** Includes noise reduction (bandpass filtering, notch filtering), artifact removal (e.g., movement artifacts), and signal alignment.
- **Feature Extraction:** Automated extraction of relevant metrics like spike rates, inter-spike intervals (ISIs), local field potentials (LFPs), and power spectral densities (PSDs).
- **Data Analysis:** Statistical analysis (e.g., t-tests, ANOVAs, correlation analysis) integrated with visualisation techniques such as raster plots, PSTHs, and time-frequency analyses.
- **Automated Reporting:** Generates customisable summary reports in LaTeX/Markdown format with key figures and metrics.

## Repository Structure

The repository is organised as follows:

- `data/`: Contains raw and preprocessed datasets. An example dataset is included for demonstration.
- `notebooks/`: Jupyter notebooks providing step-by-step interactive analysis. These include exploratory data analysis (EDA), signal processing, and statistical testing workflows.
- `src/`: Modular Python scripts for pipeline automation:
    - `preprocess.py`: Handles data cleaning, filtering, and alignment.
    - `analysis.py`: Performs statistical analysis and generates visualisations.
    - `visualise.py`: Creates detailed visual reports from processed data.
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
    python src/preprocess.py --input data/example_data.csv
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
    python src/preprocess.py --input data/example_data.csv
    python src/analysis.py --input data/preprocessed_data.csv
    ```

## Running the Analysis

### Option 1: Using Jupyter Notebooks

1. **Exploratory Data Analysis**: Begin by opening and running the Jupyter notebooks located in the `notebooks/` directory. These notebooks provide an interactive environment to explore the dataset, visualise key features, and identify artifacts.

   - To start, navigate to the repository directory and launch Jupyter:
     ```bash
     jupyter notebook
     ```
   - Open and run `01_Preprocessing.ipynb`, `02_signal_analysis.ipynb`, and `03_visualisation.ipynb` sequentially.

2. **Data Visualisation**: Use the notebooks to generate visualisations like raster plots and time-frequency analyses, which help in understanding the processed data.

### Option 2: Using Python Scripts

For a non-interactive, script-based approach:

1. **Data Preprocessing**:
    ```bash
    python src/preprocess.py --input data/example_data.csv
    ```

2. **Data Analysis**:
    ```bash
    python src/analysis.py --input data/preprocessed_data.csv
    ```

3. **Data Visualisation and Reporting**:
    ```bash
    python src/visualise.py --input results/ --output report.md
    ```

This approach is ideal for batch processing and integrating into automated pipelines.
