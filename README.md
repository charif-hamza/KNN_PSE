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

### Command-Line Options

```bash
python src/run_csth.py --help

Options:
  --data-dir PATH       Directory containing train.pt, val.pt, test.pt
                        (default: data/raw)
  --output-dir PATH     Output directory for results (default: results)
  --mode {quick,search,final,ablation,all}
                        Execution mode (default: quick)
  --best-k INT          Best k value for final evaluation (default: 10)
```

### Example Output

```
FINAL TEST RESULTS - Fault Detection Performance
======================================================================
Accuracy:      0.8361
F1 (weighted): 0.8345
F1 (macro):    0.8344
Time:          0.70s

Classification Report (0=normal, 1=fault):
              precision    recall  f1-score   support

      normal      0.917     0.738     0.818       897
       fault      0.782     0.934     0.851       903

    accuracy                          0.836      1800
   macro avg      0.849     0.836     0.834      1800
weighted avg      0.849     0.836     0.834      1800

Confusion Matrix:
                Predicted
              Normal  Fault
Actual Normal    662    235
       Fault      60    843

Fault Detection Metrics:
  Detection Rate (Recall):    0.9336
  False Alarm Rate:           0.2620
```

## Results

Results are saved to `results/` directory:

- `hyperparam_search_k.csv`: Grid search results for different k values
- `test_predictions_k{k}.csv`: Detailed predictions on test set with probabilities
- `test_results_k{k}.json`: Summary metrics and configuration
- `ablation_study.csv`: Preprocessing comparison results

## Project Structure

```
.
├── data/raw/              # Dataset files (train.pt, val.pt, test.pt)
├── src/
│   ├── __init__.py
│   ├── knn_pipeline.py    # Core KNN pipeline implementation
│   └── run_csth.py        # CSTH-specific runner and experiments
├── results/               # Experiment outputs
│   ├── hyperparam_search_k.csv
│   ├── test_predictions_k25.csv
│   └── test_results_k25.json
├── presentation_gen/      # Presentation generation utilities
├── devenv.nix            # Development environment configuration
├── devenv.lock
├── devenv.yaml
└── README.md
```

## Key Components

### PipelineConfig

Configurable hyperparameters:
- **Cross-validation**: `n_splits`, group handling
- **Preprocessing**: `standardize`, `use_pca`, `pca_variance_threshold`, `n_components`
- **KNN**: `k`, `distance_metric`, `distance_weighting`
- **DTW**: `dtw_window`, `dtw_batch_size`
- **Computation**: `n_jobs`, `verbose`

### TimeSeriesPreprocessor

Handles preprocessing pipeline:
- Feature standardization across all samples
- Per-timestep PCA compression
- Automatic variance-based component selection
- Consistent output shape handling

### Distance Computation

- **Euclidean**: Fast vectorized computation on flattened sequences
- **DTW**: Custom implementation with Sakoe-Chiba band for computational efficiency

## Performance

Achieved performance on CSTH dataset (k=25, Euclidean distance):

| Metric | Value |
|--------|-------|
| Accuracy | 83.61% |
| F1 Score (weighted) | 83.45% |
| F1 Score (macro) | 83.44% |
| Detection Rate | 93.36% |
| False Alarm Rate | 26.20% |
| Inference Time | 0.70s |

**Confusion Matrix** (n=1,800):
- True Negatives: 662
- False Positives: 235
- False Negatives: 60
- True Positives: 843

The model achieves high detection rate (93.4%) for faults but has moderate false alarm rate (26.2%), indicating a bias toward detecting faults to minimize missed detections—a reasonable trade-off for safety-critical applications.

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

- **Code**: Available for research and educational purposes
- **Dataset**: CC BY-NC 4.0 (non-commercial use only)

## Requirements

- Python 3.12 (avoid 3.13 due to library compatibility)
- PyTorch (CPU version sufficient)
- NumPy, SciPy, scikit-learn
- Pandas, Matplotlib, Seaborn
- See `devenv.nix` for complete dependency specification

### System Dependencies

The devenv configuration includes:
- Git
- zlib (libz.so.1)
- Standard C++ library (libstdc++.so.6, libgcc_s.so.1)
- GNU Fortran library (for numpy/scipy)

## Development

### Testing

Run the built-in tests on synthetic data:

```bash
python src/knn_pipeline.py
```

This will execute:
1. Euclidean KNN with grid search
2. DTW KNN with single k value
3. Ablation study on preprocessing options

### Customization

To use the pipeline on your own time series data:

```python
from knn_pipeline import evaluate_knn, PipelineConfig

# Your data: (n_samples, n_timesteps, n_features)
X_seq = ...  # Shape: (N, T, F)
y = ...      # Shape: (N,)
groups = ... # Shape: (N,) - for GroupKFold CV

config = PipelineConfig(
    k=10,
    distance_metric='euclidean',
    use_pca=True,
    standardize=True,
)

result = evaluate_knn(X_seq, y, groups, config)
print(result)
```

## Troubleshooting

**Issue**: `FileNotFoundError: Data file not found`
- Ensure dataset files are in `data/raw/` directory
- Check file names match exactly: `train.pt`, `val.pt`, `test.pt`

**Issue**: Memory errors with DTW
- Reduce `dtw_batch_size` in PipelineConfig
- Use Euclidean distance instead for faster computation

**Issue**: Poor performance
- Try different k values with `--mode search`
- Check class balance in your splits
- Verify data normalization

## Contributing

This is a research implementation. For questions or suggestions:
1. Check the original paper for methodology details
2. Review the code documentation in `src/knn_pipeline.py`
3. Open an issue with reproducible examples

## Acknowledgments

- **Dataset**: Ibrahim Yousef, Sirish L. Shah, and R. Bhushan Gopaluni (University of British Columbia)
- **CSTH Model**: Available at [Zenodo](https://zenodo.org/records/10093059)
- **Development Environment**: Built with [devenv](https://devenv.sh/)
