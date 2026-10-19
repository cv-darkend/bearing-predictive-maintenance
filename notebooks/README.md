# Notebooks

Use notebooks for documented exploration and communication, not as the only implementation of reusable logic. Move stable data loading, feature engineering, and evaluation code into `src/bearing_predictive_maintenance/` and import it here.

Notebooks are [marimo](https://marimo.io/) notebooks (`.py` files), not Jupyter. They are pure Python, reactive (re-running a cell re-runs everything that depends on it), and store no output — the committed source is the only artifact. Editing forbids defining the same variable name in more than one cell (marimo's MB002 rule); use cell-local names prefixed with `_` for anything that is not shared across cells.

Suggested numbering:

- `01_data_inventory.py`
- `02_signal_exploration.py`
- `03_data_preparation.py` (Phase III: cleaning counts, time axes, "Mixto" segmentation, leave-one-test-out folds)
- `04_feature_baseline.py`

## Workflow

```bash
make notebook    # Open the interactive marimo editor on notebooks/
make notebooks   # Export every notebook to build/notebooks/<name>.ipynb (with outputs) and .html
```

`make notebooks` requires `data/raw/` to be present (`make data`) and, for notebooks that read `data/interim/`, the derived tables from `make indicators`; `03_data_preparation.py` also reads `data/processed/ims_snapshot_manifest.csv` from `make manifest`. Exports are generated artifacts: `build/` is git-ignored and never committed. Reviewers and non-marimo readers should open the exported `.html` (self-contained, no server needed) or `.ipynb` under `build/notebooks/`.
