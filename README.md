# Bearing Predictive Maintenance

Reproducible research scaffold for an applied master's project on bearing condition monitoring and predictive maintenance.

## Dataset

This project uses Vinayak Tyagi's [Bearing Dataset on Kaggle](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset). The raw dataset is large (approximately 6.5 GB) and is **not redistributed by this repository**.

The repository's MIT license applies only to original code and documentation. Before using or redistributing the dataset, review its current terms, provenance, and access conditions on Kaggle; this project does not assert a dataset license.

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- A Kaggle account with access to the dataset
- Sufficient local storage for the download, extraction, and derived data

## Setup

```bash
make setup
```

This creates a local virtual environment and installs the locked project and development dependencies.

## Kaggle authentication

The project uses the official Kaggle CLI. Authenticate with one of Kaggle's supported methods before downloading. The interactive flow is:

```bash
uv run kaggle auth login
```

For non-interactive environments, Kaggle also supports `KAGGLE_API_TOKEN` or its legacy `~/.kaggle/kaggle.json`, `KAGGLE_USERNAME`, and `KAGGLE_KEY` credentials. Never commit credentials; relevant local files are ignored by Git.

## Download the data

```bash
make data
```

Equivalent direct command:

```bash
uv run bearing-download --destination data/raw
```

To ask Kaggle to replace an existing download:

```bash
uv run bearing-download --destination data/raw --force
```

The command downloads and extracts only `vinayak123tyagi/bearing-dataset`. All files under `data/raw/`, `data/interim/`, `data/processed/`, and `data/external/` remain local and ignored by Git.

## Development commands

```bash
make format      # Format code and apply safe lint fixes
make lint        # Check formatting and lint rules
make typecheck   # Run strict static type checking
make test        # Run tests with coverage
make check       # Run every CI check
make notebook    # Start JupyterLab
```

CI deliberately runs without Kaggle credentials and never downloads the dataset.

## Project layout

```text
.
├── .github/workflows/ci.yml       # Automated quality checks
├── data/                           # Local-only raw and derived datasets
├── notebooks/                      # Exploratory and communication notebooks
├── reports/                        # Written findings and generated figures
├── src/bearing_predictive_maintenance/
│   └── data.py                     # Safe Kaggle download command
├── tests/                          # Automated tests
├── Makefile                        # Common project commands
├── pyproject.toml                  # Package and tool configuration
└── uv.lock                         # Reproducible dependency lockfile
```

## Reproducibility policy

1. Keep raw source files immutable and outside Git.
2. Implement repeatable transformations in `src/bearing_predictive_maintenance/`.
3. Use notebooks as documented consumers of reusable code, not as the sole implementation.
4. Record random seeds, dataset partitions, parameters, and evaluation metrics for every experiment.
5. Do not publish raw or derived data until its licensing and privacy conditions have been verified.
