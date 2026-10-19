# Feature: CRISP-DM Phase III — data preparation (NASA IMS)

Objective: cleaned, segmented, leakage-safe dataset index for Phase IV features and Phase V models.
Branch `feat/data-preparation` (stacked on `feat/eda-nasa-ims`) · delivery `ask-on-risk` (chain
strategy asked before PR creation) · runner `uv run pytest`, `make check`.

## Specs

- S1 — Scope is the thesis Phase III text (verbatim): "Fase III: A su vez se someten los datos a
  procesos de limpieza, normalización y control de calidad, con el fin de segmentar las señales en
  ventanas homogéneas; a través de las siguientes actividades: Limpieza y procesamiento [...]
  Normalización y estandarización [...] Segmentación de las señales". Goal stated by the user:
  "la idea seria sacar las Fases 2 y 3 Hoy."
- S2 — Cleaning: a per-channel snapshot manifest built from the interim tables (no raw re-read),
  with flags: post-shutdown (excluded: `spectral.flag_post_shutdown`, baseline_n=100,
  fraction 0.1 on max-channel RMS, the Phase II rule), saturated file (kept, flagged;
  `inventory.SATURATION_LEVEL`), test 3 after documented end (kept, flagged), read error (excluded).
- S3 — Time axes per snapshot: clock hours, operating hours (each gap counted at most one nominal
  600 s cadence step; ONE shared helper), and hours to the test's last valid capture on both axes.
  Per bearing: documented failure flag and failure mode. No label column (S5).
- S4 — Segmentation, user decision (verbatim): "De acuerdo con el Mixto." The unit of analysis is
  the full snapshot (20480 samples, for spectral features); time-domain features use its 10
  segments of 2048 samples. Signal-level cleaning: mean (DC) removal per channel.
- S5 — Target horizon, user decision (verbatim): "La idea creeria que es un analisis de
  sensibilidades, por ejemplo 12, 24, 36, 48, 72 pero esto solo lo doy como ejemplo." Phase III
  stays independent of H: it stores hours-to-end so any H is derived in Phase V. The grid is not
  fixed here (assumption: chosen in Phase V).
- S6 — Normalization/standardization without leakage: leave-one-test-out folds; feature
  standardization fitted on the training tests of each fold only.
- S7 — Notebook `notebooks/03_data_preparation.py` (marimo, Spanish narrative, English code) and
  report `reports/data_preparation_ims.md` (Spanish) with the cleaning counts, axes, segmentation
  and fold design; manifest written to `data/processed/` (git-ignored) via a CLI + `make` target.

## Tasks

- [x] T1 — S2, S3: `manifest.py` (+ tests): cleaning flags, time axes, `operating_hours` helper,
  CLI `bearing-manifest` → `data/processed/ims_snapshot_manifest.csv`. Route: delegated (Writer
  trigger (a) parallel with T2, disjoint files; also context backstop).
- [x] T2 — S4, S6: `segmentation.py` (+ tests): DC removal, full window + 10 segments of 2048;
  `folds.py` (+ tests): leave-one-test-out splits and train-only standardization. Route: delegated
  (parallel with T1).
- [x] T3 — S7: notebook 03 + report + Makefile/pyproject wiring + README. Route: after T1-T2.
- [x] T4 — Verify with `data-auditor` (counts, axes, leakage), `make check`.

## Log

- L1 (2026-10-18) — User: "Es el proyecto v1 la idea seria sacar las Fases 2 y 3 Hoy."
- L2 (2026-10-18) — User on H: "La idea creeria que es un analisis de sensibilidades, por ejemplo
  12, 24, 36, 48, 72 pero esto solo lo doy como ejemplo." Assistant noted the target variable is
  Phase V in the thesis; Phase III stores hours-to-end instead.
- L3 (2026-10-18) — User: "De acuerdo con el Mixto." (full snapshot as unit + 10 segments).
- L4 (2026-10-18) — Forecast ~700-900 authored lines; delivery strategy ask-on-risk, chain
  strategy question deferred to PR creation (no PR planned before Phase III closes).
- L5 (2026-10-18) — T1 1bb3873 (13 tests; real data 46480 rows, 46468 valid, post-shutdown
  files 0/2/1, saturated files 3/2/4, 1876 test-3 files after documented end, operating hours
  345.27/163.83/1053.38). T2 5aea5da (23 tests; sklearn not used: no py.typed under mypy strict;
  test 2 ch1 segment-RMS CV median 2.97%). Wiring 9e7aebc (`make manifest`, README). make check
  91 passed after clearing a stale .mypy_cache. T3 delegated (context backstop).
- L6 (2026-10-19) — T3 8d89a95. Native reviews approved+acknowledged: T1/T2/wiring slice
  87b2352..b1b7f91 (lineage review-bd3569fe8328e5a5), T3 slice (review-d01303da2268dd43), window
  fix slice 8d89a95..ea2a529 (review-099de... target 795f -> sliced lineage approved).
  data-auditor T4: all numbers reproduce from raw; 1 blocker (window-length evidence for
  "estabilidad estadística" and "capacidad computacional") fixed in ea2a529 and rechecked
  (L=2048 RMS CV 3.5/3.0%, kurtosis 7.6/10.4%; 4096 more stable; only 20480 resolves the ±3% BPFO
  window with 14.52 bins; Mixto ~3.87 ms/snapshot, ~180 s for 46468 rows, raw loading ~7.6 ms
  per file not included). Advisories (later work): manifest KeyError on unregistered test_id;
  untested no-valid-capture branch; summarize_segments statistic names duplicated; Fold.held_out
  typed object; report numbers copied by hand; saturation caveat deduplicated by file instead of
  channel; unused saturated_file column; H counts boundary (<= vs <) to fix in Phase V; positional
  mask in folds._as_mask. Phase III closed; next: Phase IV features, Phase V target grid.
