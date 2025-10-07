# KNN Pipeline for CSTH Fault Detection

This repository contains a production-ready K-Nearest Neighbors (KNN) pipeline for detecting instrumentation faults in the Continuous Stirred Tank Heater (CSTH) benchmark. The codebase focuses on reproducible experimentation, rich diagnostics, and automation around the end-to-end workflow—from dataset validation through model evaluation and report generation.

## Repository Tour

- `src/knn_pipeline.py` – Core time-series KNN implementation with preprocessing, distance computation, evaluation utilities, and helper functions for ablations/grid search.
- `src/run_csth.py` – Command line entry point that wires the pipeline to the CSTH dataset, handling data loading, experiment orchestration, and persistence of metrics/predictions.
- `notebooks/` – Exploratory material such as the distance metric comparison and a step-by-step tutorial.
- `results/` – Example experiment artifacts (CSV/JSON) produced by the CLI workflows.
- `presentation_gen/` – Script and generated slide deck summarizing the approach and findings.
- `data/` – Expected location for the CSTH dataset splits (`data/raw/train.pt`, etc.).

## Pipeline Highlights (`src/knn_pipeline.py`)

The pipeline is designed for multivariate time-series classification and exposes composable building blocks:

- **Preprocessing**
  - Optional global standardisation via `StandardScaler` applied consistently across time steps.
  - PCA-based time compression (`PCATimeCompressor`) that fits an independent PCA per time step while enforcing a common latent dimensionality. Variance retention targets or explicit component counts are configurable.
- **Distance Computation**
  - Vectorised Euclidean distances on flattened sequences for fast baselines.
  - Dynamic Time Warping (DTW) with an optional Sakoe–Chiba band; batching controls keep memory usage manageable during large evaluations.
- **Classification**
  - Weighted/unweighted voting based on precomputed distance matrices, supporting inverse-distance weighting for smoother decision boundaries.
- **Evaluation & Reporting**
  - Group-aware cross-validation (`GroupKFold`) with per-fold metrics, timing, and graceful handling of degenerate folds.
  - Rich summary objects (`EvalResult`) containing confusion matrices, scikit-learn classification reports, timing breakdowns, and metadata about the configuration.
  - Grid-search utilities (`grid_search_k`) for tuning `k` that reuse the evaluation pipeline.
- **Utilities & Demos**
  - Sliding-window generator (`make_windows`) for converting raw telemetry into model-ready tensors.
  - Synthetic data demo to sanity-check the pipeline without the CSTH dataset.

## CLI Workflows (`src/run_csth.py`)

The CLI wraps the core pipeline with CSTH-specific tooling:

- Robust dataset loading with schema validation, class balance reporting, and sanity checks for ranges/timesteps/features.
- Automatic group creation for cross-validation when explicit run identifiers are absent.
- Experiment modes:
  - `quick` – Fast diagnostic cross-validation on the validation split.
  - `search` – Grid search across `k` values with results written to `results/hyperparam_search_k.csv`.
  - `ablation` – Comparison of preprocessing configurations (scaling/PCA combinations) to quantify their impact.
  - `final` – Trains on train+val, evaluates on test, and persists predictions/metrics (see `results/test_predictions_k25.csv` and `results/test_results_k25.json` for examples).
  - `all` – Runs the quick check, hyperparameter search, and final evaluation sequentially.
- Outputs include JSON/CSV summaries plus pretty-printed confusion matrices and classification reports for quick inspection.

Invoke the CLI with:

```bash
python src/run_csth.py --mode quick
```

Run `python src/run_csth.py --help` for the full list of options, including `--data-dir`, `--output-dir`, and `--best-k`.

## Dataset: CSTH Simulated Benchmark

- **Source**: [Zenodo Dataset (DOI: 10.5281/zenodo.10093059)](https://zenodo.org/records/10093059)
- **Task**: Binary time-series classification (normal vs. fault)
- **Samples**: 9,000 sequences split into train (6,300), validation (900), and test (1,800)
- **Shape**: `(N, T=200, F=3)` with 200 time steps and three process variables (cold water flow, tank level, temperature)
- **Labels**: `0` for normal operation, `1` for instrumentation fault conditions (balanced dataset)
- **Normalization**: Values scaled to `[0, 1]`

Place the splits under `data/raw/` to align with the CLI defaults:

```
data/raw/
├── train.pt
├── val.pt
└── test.pt
```

## Running Experiments

```bash
# Quick validation experiment (5-fold GroupKFold over validation data)
python src/run_csth.py --mode quick

# Hyperparameter sweep over candidate k values
python src/run_csth.py --mode search --output-dir results

# Evaluate preprocessing variants (scaling/PCA)
python src/run_csth.py --mode ablation

# Final model evaluation on the held-out test split
python src/run_csth.py --mode final --best-k 25

# Execute quick, search, and final back-to-back
python src/run_csth.py --mode all --best-k 25
```

Example output from a final test run (Euclidean distance, `k=25`):

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
```

## Development Environment

### Using devenv (Recommended)

The project ships with a [devenv](https://devenv.sh/) configuration for reproducible local environments:

```bash
# Install devenv if needed (see https://devenv.sh/getting-started/)

# Enter the prepared shell with all dependencies
devenv shell
```

### Manual Setup

If you prefer a lightweight virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install torch numpy scipy pandas scikit-learn matplotlib \
            pyarrow python-pptx beautifulsoup4 lxml seaborn tqdm
```

### Testing

The repository includes a synthetic-data smoke test inside `src/knn_pipeline.py` (run the module directly) and the CLI commands above. Execute the full automated test suite with:

```bash
pytest
```

## Additional Assets

- **Notebooks** – Explore the methodology interactively via `notebooks/distance_metrics_comparison.ipynb` and `notebooks/tutorial.ipynb`.
- **Presentation Generator** – `presentation_gen/generate_presentation.py` builds a PowerPoint summary (`presentation_knn_csth_enhanced.pptx`).
- **Saved Artifacts** – CLI runs emit metrics/predictions into the `results/` directory; these files double as templates for integrating the pipeline into downstream reporting systems.
