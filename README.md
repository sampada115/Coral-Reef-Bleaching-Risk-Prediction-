# Coral Reef Bleaching Risk Prediction

A Jupyter Notebook-based project for exploring, modeling, and predicting coral reef bleaching risk using environmental and oceanographic data. This repository contains notebooks that demonstrate data preparation, feature engineering, exploratory analysis, model training, evaluation, and visualization of bleaching risk predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Data](#data)
- [Notebooks](#notebooks)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modeling Approach](#modeling-approach)
- [Results & Evaluation](#results--evaluation)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
Coral bleaching is a major threat to reef ecosystems worldwide. This project demonstrates a workflow to predict bleaching risk from environmental variables (SST, anomalies, light/insolation proxies, chlorophyll, etc.) and other features. The goal is to provide an approachable, reproducible analysis and baseline models that can be extended.

## Features
- Data ingestion and cleaning pipeline demonstrated in a notebook
- Exploratory data analysis (EDA) with visualizations
- Feature engineering examples and dataset splitting
- Baseline machine learning models (e.g., logistic regression, random forest, XGBoost)
- Evaluation metrics and visualizations (ROC, precision/recall, confusion matrices)
- Notebook-friendly, reproducible steps for experimentation

## Data
This repository does not include raw data due to size and licensing constraints. Use the following guidelines:
- Expected inputs: CSVs, netCDF, or other tabular/time-series formats containing environmental variables and bleaching labels or observations.
- Organize your data into a `data/` directory with subfolders:
  - `data/raw/` — original downloaded files
  - `data/processed/` — cleaned and preprocessed files used by notebooks
- Example sources you might use:
  - NOAA Coral Reef Watch (SST, DHW)
  - Remote sensing products (MODIS chlorophyll)
  - In-situ reef observation datasets

Replace placeholders in the notebooks with your dataset paths.

## Notebooks
- `notebooks/01_data_preparation.ipynb` — load and clean raw data, create processed dataset
- `notebooks/02_exploratory_analysis.ipynb` — EDA and visualizations
- `notebooks/03_modeling.ipynb` — train baseline models and evaluate
- `notebooks/04_results_visualization.ipynb` — maps, time series, and model output visualizations

(If your repository contains different notebook filenames, update the list above accordingly.)

## Installation

1. Clone the repo:
   git clone https://github.com/sampada115/Coral-Reef-Bleaching-Risk-Prediction-.git
   cd Coral-Reef-Bleaching-Risk-Prediction-

2. Create a Python environment (recommended):
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate     # Windows

3. Install dependencies:
   pip install -r requirements.txt

Example minimal `requirements.txt`:
- jupyterlab
- notebook
- numpy
- pandas
- scikit-learn
- xgboost
- matplotlib
- seaborn
- geopandas (optional, for mapping)
- netCDF4 (optional)
- rasterio (optional)

You can also run the notebooks on Google Colab — change file paths or mount Google Drive as needed.

## Quick Start
1. Start Jupyter:
   jupyter lab
   or
   jupyter notebook

2. Open `notebooks/01_data_preparation.ipynb` and run cells in order. The notebooks are linear and intended to be executed sequentially.

3. If you have a large dataset, run preprocessing first and save processed files to `data/processed/` to speed up later runs.

## Modeling Approach
- Problem framing: binary classification (bleached vs non-bleached) or risk scoring/regression depending on labels available.
- Typical pipeline:
  - Data cleaning and temporal alignment
  - Feature extraction (SST statistics, degree heating weeks, seasonal features, location metadata)
  - Train/test split respecting temporal and/or spatial structure to avoid leakage
  - Baseline model(s): logistic regression, random forest, XGBoost
  - Calibration and threshold selection for risk categorization
- Evaluation metrics: ROC AUC, precision, recall, F1, confusion matrix, spatial validation if applicable.

## Results & Evaluation
Notebooks include example evaluation plots:
- ROC and PR curves
- Confusion matrices and classification reports
- Spatial visualizations of predicted risk (if geolocation data is available)

Interpretability notes:
- Use feature importance (e.g., SHAP) for insight into drivers of predicted bleaching risk.
- Validate models across regions and time periods to ensure generalization.

## Reproducibility
- Record package versions in `requirements.txt` or use `pip freeze > requirements.txt`.
- Use a fixed random seed in notebooks when training models for deterministic results.
- Save model artifacts and processed datasets to `artifacts/` or `models/` with timestamps.

## Acknowledgements & References
- NOAA Coral Reef Watch
- Relevant scientific literature on coral bleaching and environmental drivers
