# KNN Pipeline for CSTH Fault Detection

A production-quality K-Nearest Neighbors (KNN) pipeline for time series classification applied to the CSTH (Continuous Stirred Tank Heater) benchmark dataset for fault detection and diagnosis.

## Overview

This repository implements a comprehensive KNN-based approach for detecting faults in a simulated heating system. The pipeline includes advanced preprocessing, multiple distance metrics (Euclidean and DTW), and extensive evaluation capabilities.

### Dataset: CSTH Simulated Benchmark

- **Source**: [Zenodo Dataset (DOI: 10.5281/zenodo.10093059)](https://zenodo.org/records/10093059)
- **Task**: Binary time series classification for fault detection
- **Domain**: Process control simulation (continuous stirred tank heater)
- **Size**: 9,000 multivariate time series samples
  - Training: 6,300 samples (70%)
  - Validation: 900 samples (10%)
  - Test: 1,800 samples (20%)
- **Structure**: Each sample has shape `(T=200, F=3)`
  - 200 time steps
  - 3 process variables: cold water flow, tank level, temperature
- **Labels**:
  - `Y=0`: Normal operating conditions (50%)
  - `Y=1`: Faulty scenarios - instrumentation errors (50%)
- **Preprocessing**: Data normalized to [0, 1]

## Features

### Core Pipeline (`src/knn_pipeline.py`)

- **Preprocessing**:
  - StandardScaler for feature normalization
  - PCA-based time compression (independent PCA per timestep)
  - Configurable variance retention thresholds
  
- **Distance Metrics**:
  - Euclidean distance (vectorized, fast)
  - Dynamic Time Warping (DTW) with Sakoe-Chiba band constraint
  - Memory-efficient batch processing for large datasets

- **Classification**:
  - Weighted and unweighted KNN voting
  - Distance-based weighting (inverse distance)
  - Comprehensive cross-validation with GroupKFold

- **Evaluation**:
  - Accuracy, F1 (weighted & macro), precision, recall
  - Per-fold metrics for debugging
  - Confusion matrices and classification reports
  - Timing statistics

### CSTH Runner (`src/run_csth.py`)

Domain-specific runner for fault detection experiments:

- **Quick Test**: Fast validation on subset with cross-validation
- **Hyperparameter Search**: Grid search over k values (1-30)
- **Ablation Study**: Compare preprocessing configurations
- **Final Evaluation**: Train on combined train+val, test on holdout set

## Installation

### Using devenv (Recommended)

This project uses [devenv](https://devenv.sh/) for reproducible development environments:

```bash
# Install devenv (if not already installed)
# See: https://devenv.sh/getting-started/

# Enter the development environment
devenv shell

# All dependencies will be automatically installed
```

### Manual Installation

Alternatively, install dependencies manually:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy scipy pandas scikit-learn matplotlib \
            pyarrow python-pptx beautifulsoup4 lxml seaborn tqdm
```

## Usage

### Data Setup

1. Download the CSTH dataset from [Zenodo](https://zenodo.org/records/10093059)
2. Place the files in `data/raw/`:
   ```
   data/raw/
   ├── train.pt
   ├── val.pt
   └── test.pt
   ```

### Running Experiments

```bash
# Quick validation test (5-fold CV on validation set)
python src/run_csth.py --mode quick

# Hyperparameter search (grid search over k values)
python src/run_csth.py --mode search

# Ablation study (compare preprocessing configurations)
python src/run_csth.py --mode ablation

# Final test evaluation (requires best k from search)
python src/run_csth.py --mode final --best-k 25

# Run all experiments sequentially
python src/run_csth.py --mode all
```

### Example Output

```
FINAL TEST RESULTS - Fault Detection Performance
======================================================================
Accuracy:      0.9856
F1 (weighted): 0.9856
F1 (macro):    0.9856
Time:          12.34s

Classification Report (0=normal, 1=fault):
              precision    recall  f1-score   support

      normal      0.988     0.983     0.985       900
       fault      0.983     0.988     0.986       900

    accuracy                          0.986      1800
   macro avg      0.986     0.986     0.986      1800
weighted avg      0.986     0.986     0.986      1800

Fault Detection Metrics:
  Detection Rate (Recall):    0.9878
  False Alarm Rate:           0.0167
```

## Results

Results are saved to `results/` directory:

- `hyperparam_search_k.csv`: Grid search results for different k values
- `test_predictions_k{k}.csv`: Detailed predictions on test set
- `test_results_k{k}.json`: Summary metrics and configuration
- `ablation_study.csv`: Preprocessing comparison results

## Project Structure

```
.
├── data/raw/              # Dataset files (train.pt, val.pt, test.pt)
├── src/
│   ├── knn_pipeline.py    # Core KNN pipeline implementation
│   └── run_csth.py        # CSTH-specific runner and experiments
├── results/               # Experiment outputs
├── presentation_gen/      # Presentation generation utilities
├── devenv.nix            # Development environment configuration
└── README.md
```

## Key Components

### PipelineConfig

Configurable hyperparameters:
- Cross-validation: `n_splits`, group handling
- Preprocessing: `standardize`, `use_pca`, `pca_variance_threshold`
- KNN: `k`, `distance_metric`, `distance_weighting`
- DTW: `dtw_window`, `dtw_batch_size`

### TimeSeriesPreprocessor

Handles preprocessing pipeline:
- Feature standardization across all samples
- Per-timestep PCA compression
- Automatic variance-based component selection

### Distance Computation

- **Euclidean**: Fast vectorized computation on flattened sequences
- **DTW**: Custom implementation with Sakoe-Chiba band for efficiency

## Performance

Typical performance on CSTH dataset:
- **Accuracy**: ~98.5%
- **F1 Score**: ~98.5%
- **Detection Rate**: ~98.8%
- **False Alarm Rate**: ~1.7%
- **Training Time**: ~10-15s (full train+val on test)

## Citation

If you use this code or the CSTH dataset, please cite:

```bibtex
@article{yousef2025timeseries,
  title={Time Series Representation Learning via Cross-Domain Predictive and Contextual Contrasting: Application to Fault Detection},
  author={Yousef, Ibrahim and Shah, Sirish L. and Gopaluni, R. Bhushan},
  journal={Engineering Applications of Artificial Intelligence},
  year={2025},
  note={Available at SSRN: https://ssrn.com/abstract=5085741}
}
```

## License

- **Code**: MIT License (see repository)
- **Dataset**: CC BY-NC 4.0 (non-commercial use only)

## Requirements

- Python 3.12 (3.13 not recommended due to library compatibility)
- PyTorch (CPU version)
- NumPy, SciPy, scikit-learn
- Pandas, Matplotlib, Seaborn
- See `devenv.nix` for complete dependency list

## Contributing

This is a research prototype. For questions or issues, please refer to the original paper or create an issue in the repository.

## Acknowledgments

Dataset provided by Ibrahim Yousef, Sirish L. Shah, and R. Bhushan Gopaluni from the University of British Columbia. The CSTH simulation model is available at [Zenodo](https://zenodo.org/records/10093059).
