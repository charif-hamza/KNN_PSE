# KNN Pipeline for CSTH Fault Detection

This repository contains a production-ready K-Nearest Neighbours (KNN) pipeline for detecting instrumentation faults in the Continuous Stirred Tank Heater (CSTH) benchmark. The project emphasises reproducibility, comprehensive diagnostics, and automation from dataset ingestion through model evaluation and reporting. The codebase is dual-licensed under MIT and Apache-2.0 so that academic and industrial users can adopt it with minimal friction.

## Quickstart

```bash
# 1. Create an isolated environment and install the project
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]

# 2. Fetch the CSTH dataset splits (downloads ~200 MB)
python scripts/download_data.py

# 3. Run a one-minute smoke test on the validation split
knn-csth --mode quick --data-dir data/raw --threshold-grid 0.3 0.5 0.7

# 4. Execute the automated quality gates
pytest
ruff check .
black --check .
mypy src
```

The `quick` mode performs a fast GroupKFold evaluation on the validation split, exercising preprocessing, distance computation, neighbour search, and reporting. More extensive experiments are available via the `knn-csth` console script that is installed alongside the package.

## Repository Tour

- `src/knn_pse/pipeline.py` – Core time-series KNN implementation with preprocessing, distance computation (Euclidean & DTW), evaluation utilities, run-manifest helpers, and grid-search routines.
- `src/knn_pse/cli.py` – Command-line interface that orchestrates CSTH experiments, handles dataset validation, and persists metrics/predictions.
- `scripts/download_data.py` – Downloads the CSTH splits from Zenodo with SHA-256 verification and writes a checksum manifest.
- `notebooks/` – Exploratory analysis notebooks (distance-metric comparisons, tutorial walkthroughs).
- `results/` – Example experiment artefacts (CSV/JSON) produced by the CLI workflows.
- `presentation_gen/` – Script and generated slide deck summarising the approach and findings.
- `data/` – Expected location for the CSTH dataset splits (`data/raw/train.pt`, etc.).

## Pipeline Highlights (`src/knn_pse/pipeline.py`)

The pipeline targets multivariate time-series classification and exposes composable building blocks:

- **Preprocessing**
  - Optional global standardisation via `StandardScaler` applied consistently across time steps.
  - PCA-based time compression (`PCATimeCompressor`) that fits an independent PCA per time step while enforcing a common latent dimensionality. Variance-retention targets or explicit component counts are configurable.
- **Distance Computation**
  - Vectorised Euclidean distances on flattened sequences with optional PyTorch acceleration (CPU/GPU) and lower-triangular batching that can spill to memory-mapped storage for large jobs.
  - Dynamic Time Warping with pluggable backends (`python`, `numba`, `fastdtw`, `dtaidistance`) and optional Sakoe–Chiba constraints.
  - Approximate nearest-neighbour search using Annoy or FAISS to accelerate large evaluation grids.
- **Classification**
  - Weighted/unweighted voting based on precomputed distances or ANN neighbour lookups with inverse-distance weighting for smoother decision boundaries.
- **Evaluation & Reporting**
  - Group-aware cross-validation (`GroupKFold`) with per-fold metrics, wall-clock timings, and graceful handling of degenerate folds.
  - Rich summary objects (`EvalResult`) including confusion matrices, scikit-learn classification reports, ROC-AUC/PR-AUC, per-class F1 scores, threshold sweeps, and metadata about the configuration.
  - Automatic run manifests capturing git commit, configuration, timings, and hashed data fingerprints for reproducibility.
- **Utilities & Demos**
  - Sliding-window generator (`make_windows`) for converting raw telemetry into model-ready tensors.
  - Synthetic data generator (`demo_synthetic`) for end-to-end smoke testing without the CSTH dataset.

## CLI Workflows (`src/knn_pse/cli.py`)

The CLI wraps the core pipeline with CSTH-specific tooling:

- Robust dataset loading with schema validation, class-balance reporting, and sanity checks for ranges/timesteps/features.
- Automatic group creation for cross-validation when explicit run identifiers are absent.
- Experiment modes:
  - `quick` – Fast diagnostic cross-validation on the validation split.
  - `search` – Grid search across `k` values.
  - `final` – Train on train+val, evaluate on test, persist predictions/metrics, and emit a run manifest.
  - `all` – Run quick, search, and final back-to-back.

Invoke the CLI with:

```bash
knn-csth --mode quick
```

Run `knn-csth --help` for the full list of options, including `--data-dir`, `--output-dir`, `--best-k`, `--distance`, and acceleration toggles.

## Dataset: CSTH Simulated Benchmark

- **Source**: [Zenodo Dataset (DOI: 10.5281/zenodo.10093059)](https://zenodo.org/records/10093059)
- **Task**: Binary time-series classification (normal vs. fault)
- **Samples**: 9,000 sequences split into train (6,300), validation (900), and test (1,800)
- **Shape**: `(N, T=200, F=3)` with 200 time steps and three process variables (cold water flow, tank level, temperature)
- **Labels**: `0` for normal operation, `1` for instrumentation fault conditions (balanced dataset)
- **Normalisation**: Values scaled to `[0, 1]`

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
knn-csth --mode quick

# Hyperparameter sweep over candidate k values
knn-csth --mode search --output-dir results

# Final model evaluation on the held-out test split
knn-csth --mode final --best-k 25

# Execute quick, search, and final back-to-back
knn-csth --mode all --best-k 25
```

Example output from a final test run (Euclidean distance, `k=25`):

```
FINAL TEST RESULTS - Fault Detection Performance
======================================================================
Accuracy:      0.8361
F1 (weighted): 0.8345
F1 (macro):    0.8344
Time:          0.70s
ROC-AUC:       0.9012
PR-AUC:        0.9085

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

pip install -e .[dev]
```

### Testing

The repository now includes a pytest suite under `tests/` that covers windowing, preprocessing, distance-matrix symmetry, and an end-to-end synthetic evaluation. Execute the automated test suite with:

```bash
pytest
```

Static analysis is available via `ruff`, `black`, and `mypy` (configured through pre-commit hooks).

## License

This project is available under the terms of both the [MIT](./LICENSE-MIT) and [Apache-2.0](./LICENSE-APACHE) licences. You may use, copy, modify, and distribute the software under either licence at your option.
