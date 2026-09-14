# Data directory

Downloaded data is local-only and is never committed to Git.

| Directory | Purpose |
| --- | --- |
| `raw/` | Immutable files downloaded from Kaggle. |
| `interim/` | Reversible intermediate transformations. |
| `processed/` | Analysis-ready datasets. |
| `external/` | Additional third-party data, if approved. |

Run `make data` from the repository root to populate `raw/`. Do not manually modify raw source files. Record every transformation in code and write derived outputs to `interim/` or `processed/`.
