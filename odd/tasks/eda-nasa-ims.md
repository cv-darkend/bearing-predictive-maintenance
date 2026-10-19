# Feature: EDA of the NASA IMS bearing dataset

Locator: `odd/tasks/eda-nasa-ims.md` · Engram mirror: `odd/eda-nasa-ims/tasks`

## Objective

Deliver CRISP-DM Phase II (Data Understanding) for the NASA IMS run-to-failure dataset:
inventory, integrity/consistency verification, and exploratory analysis that grounds the
later decisions on target labeling (failure within a time window), windowing, and
leakage-safe splitting.

## Problem / why

The anteproyecto commits to "Dataset de vibraciones preparado y documentado, que incluya el
EDA". Modeling decisions (label horizon, split by bearing/test, feature families) must be
justified by evidence from this phase, not assumed.

## Scope (authorized 2026-09-17)

- Reusable, tested loading and inventory code in `src/bearing_predictive_maintenance/`.
- Snapshot-level time-domain indicator table written to `data/interim/` (git-ignored).
- Notebooks `notebooks/01_data_inventory.py` and `notebooks/02_signal_exploration.py` (marimo;
  marimo notebooks export cleanly via `make notebooks`).
- Written findings in `reports/eda_nasa_ims.md`.

Out of scope: final windowing, feature matrix, labels, and modeling (Phases III-V).

## Constraints

- `data/raw/` is immutable; derived outputs go to `data/interim/` or `data/processed/`.
- No new dependencies unless justified.
- Code, comments, and artifacts in English; report prose may follow the thesis language if requested.
- TDD: strict, enabled (session configuration). Runner: `uv run pytest`. Also `make check`
  (ruff format/check, mypy strict, pytest).

## Known facts (observed 2026-09-17)

| Test | Raw folder | Files | First file | Last file | Channels |
| --- | --- | --- | --- | --- | --- |
| 1 | `data/raw/1st_test/1st_test/` | 2156 | 2003.10.22.12.06.24 | 2003.11.25.23.39.56 | 8 |
| 2 | `data/raw/2nd_test/2nd_test/` | 984 | 2004.02.12.10.32.39 | 2004.02.19.06.22.39 | 4 |
| 3 | `data/raw/3rd_test/4th_test/txt/` | 6324 | 2004.03.04.09.27.46 | 2004.04.18.02.42.55 | 4 |

Each file: tab-separated, no header, CRLF, 20480 rows (1 s at 20 kHz).
Test 3 file count/end date appear to differ from NASA documentation (4448 files, ends 2004-04-04) — must be verified and documented.

## Tasks

- [x] T1 — IMS metadata + loader module (`ims.py`): test registry (paths, channel→bearing map, documented failures, sampling rate, shaft speed, bearing geometry), filename timestamp parsing, snapshot loader. Tests first.
- [x] T2 — Inventory and integrity checks: per-file table (test, timestamp, rows, cols, NaN/non-numeric count, gap to previous snapshot), cadence summary, documented-vs-observed comparison. Tests first.
- [x] T3 — Time-domain indicators per snapshot and channel (mean, std, RMS, peak, peak-to-peak, skewness, kurtosis, crest factor, impulse factor, shape factor, energy) over all tests → `data/interim/ims_snapshot_indicators.csv`, plus a CLI entry point. Tests first.
- [x] T4 — Notebook 01 data inventory: inventory, integrity, cadence gaps, test 3 discrepancy.
- [x] T5 — Notebook 02 signal exploration: raw waveforms healthy vs end-of-life, amplitude distributions, degradation trends per bearing, spectra/envelope early vs late with characteristic fault frequencies, channel correlation (test 1 x/y), onset of degradation.
- [x] T6 — Findings report `reports/eda_nasa_ims.md`: facts, anomalies, inclusion/exclusion criteria, implications for labeling, windowing, and splitting.
- [x] T7 — Migrate notebooks to marimo (user decision 2026-09-17): `notebooks/01_data_inventory.py` and `notebooks/02_signal_exploration.py` as the source of truth (no stored outputs); remove the `.ipynb` sources; add `marimo` dependency; `make notebooks` exports `.ipynb` (with outputs) and `.html` to a git-ignored location; report references updated. Reason: `.ipynb` with embedded base64 figures (1.8 MB) exceeded the native reviewer context (~1.9M tokens vs 1M limit).

### Phase II corrections (reopened 2026-10-18 after full review, see Log L2-L4)

Route: all inline (no Writer trigger: units are sequential and depend on T8's regenerated CSV).

- [x] T8 — Amplitude indicators on the mean-removed signal (RMS, peak, peak-to-peak, energy,
  crest/impulse/shape factor); keep `mean` as the DC level. Tests first: an offset sine gives
  `rms == std`; Pearson kurtosis of a sine is 1.5; skewness/kurtosis NaN on constant signal.
  Regenerate `data/interim/ims_snapshot_indicators.csv`.
- [x] T9 — Fault-frequency evidence: measure the shaft line and the outer-race peak in the same
  spectra to separate bearing slip from a sampling-rate/speed deviation (20480 samples vs
  "1 s at 20 kHz"); replace the fixed ±5 Hz read with a documented peak search; check 2×BSF
  for test 1 bearing 4. Tests first for the peak search.
- [x] T10 — Notebook evidence: integrity counts (non-finite, read errors, column count vs
  channel map) in notebook 01; temporal autocorrelation of indicators and within-snapshot
  segment stability in notebook 02; onset table in hours from the last valid capture.
- [x] T11 — Report corrections in `reports/eda_nasa_ims.md`: BPFO peak statement, test 1
  bearing 4 kurtosis episode, BSF growth (~2×), last-24 h pauses, test 3 raw spectrum, test 2
  growth timing, redundancy groups, leakage argument (temporal autocorrelation; bands and
  thresholds chosen on full series must be fitted on training folds), segment stability,
  geometry source (Qiu et al. 2006), 4448-files evidence, 5-min cadence runs, Kaggle source.
- [x] T12 — Edge-case hardening + tests: `detect_onset` searches after the baseline and
  validates parameters (F4); NaN handling in `detect_onset`/`flag_post_shutdown` (F3);
  `summarize_inventory` exposes read-error/non-finite counts (F1); unused params removed;
  inventory/registry assertions the review found missing.
- [~] T13 — Re-audit with `data-auditor` (claims) and `make check` DONE (0f7d37e, recheck B1-B3
  fixed); update the Phase II section of the Google Doc copy of "Proyecto V1" PENDING (blocked:
  no Google Doc copy in Drive yet, Google Docs connector not enabled).

- [x] T14 — Integrate the teammate's valid Phase II findings (origin/eda-santiago, ca030c1) into
  our format, not his notebooks (Log L8-L9): short intervals < 240 s (test 1: 9, test 3: 2); the
  13 s pair "posible duplicado" check; content-hash duplicate check over all files (inventory);
  explicit flat-signal criterion as a cross-check of post-shutdown flags; x/y axis correlation
  (Spearman) for RMS, kurtosis and crest factor (axis redundancy holds for RMS only); the a-priori
  2-8 kHz envelope band as a leakage-free option; fix rail wording (-5.000/+4.998, not ±4.999).
  Route: delegated (context backstop).

## Acceptance criteria

- `make check` passes.
- Notebooks execute top-to-bottom from a clean kernel with `data/raw/` present.
- Every numeric claim in the report is traceable to code/notebook output.

## Progress / evidence

- 2026-09-17: dataset downloaded via `make data` (1.56 GB zip, 6.2 GB extracted). Formats inspected.
- 2026-09-17: T1-T3 implemented (strict TDD). New modules: `src/bearing_predictive_maintenance/ims.py`
  (test registry, bearing geometry, `parse_snapshot_timestamp`, `list_snapshots`, `load_snapshot`,
  `characteristic_fault_frequencies`), `inventory.py` (`build_inventory`, `summarize_inventory`),
  `indicators.py` (`compute_time_domain_indicators`, `build_indicator_dataset`, CLI `main`/`build_parser`).
  New tests: `tests/test_ims.py`, `tests/test_inventory.py`, `tests/test_indicators.py` (25 tests total,
  synthetic fixtures only, no real data read in tests).
  - RED: `uv run pytest tests/test_ims.py` → `ImportError: cannot import name 'ims'` (module absent).
    Then `tests/test_inventory.py` → `ImportError: cannot import name 'inventory'`.
    Then `tests/test_indicators.py` → `ImportError: cannot import name 'indicators'`.
  - GREEN: `uv run pytest -q` → `25 passed` (94% coverage of the package; `ims.py` 100%,
    `inventory.py` 98%, `indicators.py` 96%).
  - `make check` → ruff format/check clean, `mypy` clean (added a `[[tool.mypy.overrides]]` for
    `pandas.*`/`scipy.*` with `ignore_missing_imports = true` — no stub packages are published/installed
    for these and adding `pandas-stubs`/`scipy-stubs` was out of scope as a new dependency), pytest
    `25 passed`.
  - Added CLI entry point `bearing-indicators` in `pyproject.toml` `[project.scripts]` and a `make
    indicators` target.
  - Real-data run: `uv run bearing-indicators --max-workers 16` → completed in 1m14s, wrote
    `data/interim/ims_inventory.csv` (9464 rows) and `data/interim/ims_snapshot_indicators.csv`
    (46480 rows = 9464 files × channels). Zero read errors, zero non-finite values, all files have
    exactly 20480 rows across all three tests.
  - Documented-vs-observed per test (from `summarize_inventory`):
    - Test 1: file count 2156/2156 match; start/end match documented; median gap 600s (5 files/finer
      cadence only over the first hour); **3 large real recording gaps** (weekends/pauses): the
      biggest is ~6.2 days (534,333 s) around 2003-10-29, plus ~4.0 days and ~3.8 days — 24 gaps
      exceed 1.5× the median.
    - Test 2: file count 984/984 match; start/end match documented exactly; median gap 600s;
      no gaps exceed 1.5× median (perfectly regular cadence).
    - Test 3: **confirms the discrepancy already noted above** — observed 6324 files vs documented
      4448, observed end 2004-04-18 02:42:55 vs documented 2004-04-04 19:01:57 (raw data extends
      ~13.5 days past the NASA readme's documented end); start matches; median gap 600s; 4 gaps
      exceed 1.5× median (max ~7.5 hours).
    - No duplicate timestamps in any test.
  - Surprising: the very last snapshot of test 2 and test 3 has near-zero amplitude (test 2 last-file
    RMS ≈0.001-0.002 vs a first-100-file median RMS of ≈0.06-0.10 across bearings) — the raw values are
    almost all `-0.002`/`0`/`0.002`, suggesting the rig may have already been shut down before that final
    recording. Test 1's degradation is visible in the indicators too: bearing 3 kurtosis rises from a
    ~3.4-3.6 baseline to ~7.3-8.9 at the last snapshot (its documented inner-race defect), while bearing
    4's kurtosis does not show a comparable rise despite its documented roller-element defect — worth
    flagging for T5/T6.
  - Note for T4-T6: `test_id` round-trips through CSV as an integer (pandas infers numeric dtype on
    read), so the report/notebooks must read it with `dtype={"test_id": str}` or otherwise coerce it
    back to a string to match the registry's string keys ("1"/"2"/"3").

- 2026-09-17 parent spot check: `make check` → 25 passed, 94% coverage. Tail inspection of max-channel
  RMS: test 2 has **2** dead trailing snapshots (06:12:39, 06:22:39; RMS ≈0.004/0.002) right after
  RMS ≈0.46-0.48; test 3 has **1** dead trailing snapshot (02:42:55, RMS ≈0.004) right after RMS
  ≈0.45→0.76. Test 1 has none (last RMS ≈0.59). Implication: exclude post-shutdown snapshots; test 3
  degradation peaks at the end of the extended recording (2004-04-18), so the documented 2004-04-04 end
  likely describes a truncated release, not the failure time.

- 2026-09-17: T4-T6 started. New module `src/bearing_predictive_maintenance/spectral.py` (strict TDD,
  tests first in `tests/test_spectral.py`, 12 tests): `amplitude_spectrum` (Hann-windowed, amplitude-
  corrected single-sided FFT), `envelope_spectrum` (Butterworth band-pass via `sosfiltfilt` + Hilbert
  envelope, then `amplitude_spectrum` of the demeaned envelope), `band_amplitude` (peak reader near a
  target frequency +/- tolerance), `detect_onset` (exploratory mean+n_sigma persistence-based onset
  finder, explicitly documented as not the final labeling rule), `flag_post_shutdown` (baseline-median-
  fraction based post-shutdown flag for max-channel RMS).
  - RED: `uv run pytest tests/test_spectral.py` → `ImportError: cannot import name 'spectral'`.
  - GREEN: `uv run pytest tests/test_spectral.py -q` → `12 passed`; full suite `uv run pytest -q` →
    `37 passed`, 95% package coverage (`spectral.py` 96%).
  - `make format` reformatted `band_amplitude`'s signature to one line; `make check` → ruff
    format/check clean, mypy clean, `37 passed`.

- 2026-09-17: T4-T6 completed. Built `notebooks/01_data_inventory.ipynb` (24 cells) and
  `notebooks/02_signal_exploration.ipynb` (35 cells) programmatically (nbformat), executed
  top-to-bottom with `uv run jupyter nbconvert --to notebook --execute --inplace ... --ExecutePreprocessor.timeout=1800`
  — both run cleanly (0 errors), notebook 01 in ~4s, notebook 02 in ~18s. 8 figures saved to
  `reports/figures/` (dpi 150): `01_cadence_gaps.png`, `02_raw_waveforms_early_late.png`,
  `02_amplitude_distributions.png`, `02_degradation_trends_test1.png`,
  `02_degradation_trends_test2_test3.png`, `02_indicator_correlation.png`,
  `02_spectra_envelope_early_late.png`, `02_envelope_amplitude_evolution.png`. All 8 opened
  and visually inspected (labels, legends, colors, overflow) — one collision found (heatmap
  title overlapping the colorbar "1.0" label) and fixed by shortening the title.
  - Key quantitative findings (see `reports/eda_nasa_ims.md` for full detail and citations):
    test 3 extends 13.32 days past its documented end; test1/test2/test3 have 0/2/1
    post-shutdown snapshots (RMS-based, `flag_post_shutdown`); test1 x/y RMS correlation per
    bearing is 0.98/0.98/0.99/0.93 (high redundancy); indicator correlation matrix splits into
    an amplitude cluster (rms/std/peak/peak_to_peak/energy, |r| 0.63-0.97) and an impulsivity
    cluster (kurtosis/crest_factor/impulse_factor, |r| 0.82-1.00), with skewness uncorrelated
    (|r| ≤ 0.22) to everything.
  - Envelope spectra (band chosen per-combo via a data-driven "where does late-early energy
    grow most" heuristic, not a literature reference): test 2 bearing 1 (BPFO) and test 3
    bearing 3 (BPFO) both show a clean, dominant envelope peak exactly at BPFO, while their
    raw spectra's dominant peaks are elsewhere (~700-900 Hz resonances) — textbook envelope
    use case. Test 1 bearing 3 (BPFI) shows elevated but less clean activity near BPFI.
  - **Negative/nuanced result for test 1 bearing 4 (BSF, roller element)**: no kurtosis rise
    (confirming T3) and no clean dominant envelope peak at BSF (dominant peak is ~65 Hz,
    unrelated) — but the envelope amplitude *specifically at BSF* does show a real, noisy
    ~3-4x rise in the last ~100h (baseline ≈0.002-0.004 → ≈0.008-0.012). Reported honestly as
    a weaker/more diffuse signature, not "no signal".
  - Onset (exploratory, `detect_onset`, sensitivity-tested over 3 parameter sets): highly
    parameter-sensitive, e.g. test1 bearing3 ranges 11.9-66.1h before end depending on
    (n_sigma, persistence); test1 bearing4 98.7-129.1h; test2 bearing1 60-65h; test3 bearing3
    40-42.5h.
  - Class imbalance candidates (last 24h/48h/72h of valid data, per failed bearing): ranges
    from 1.8%/7.7%/13.7% (test1, 2156 valid snapshots) to 14.8%/29.4%/44.1% (test2, only 982
    valid snapshots) to 2.3%/4.6%/6.2% (test3, 6323 valid snapshots) — varies a lot by test
    duration, flagged as an open question for horizon selection.
  - `make format` + `make check` on the whole repo (including notebooks, ruff lints `.ipynb`
    too): several `B905` (`zip()` without `strict=`) and `E501` (long lines) issues in the
    generated notebook source were fixed; final `make check` → ruff format/check clean, mypy
    clean, `37 passed`, 95% coverage.
  - Wrote `reports/eda_nasa_ims.md` (Spanish): resumen, fuente y estructura, inventario e
    integridad, hallazgos, criterios de inclusión/exclusión, implicaciones para preparación y
    modelado (etiqueta, ventanas, partición, desbalance, características), limitaciones y
    preguntas abiertas — every number cites its notebook/section, one explicit "por
    verificar" flag (envelope-band heuristic not cross-checked against literature).

- 2026-09-17 parent spot check (T4-T6): `make check` → 37 passed, 95% coverage. Recomputed class
  balance from `ims_inventory.csv`: matches report (test 1 H=24h = 39 snapshots because the last 13 h
  contain two acquisition pauses of ≈10.8 h and ≈7.1 h). Viewed figures
  `02_envelope_amplitude_evolution.png` and `02_degradation_trends_test2_test3.png`: fixed bearing
  colors, single y-axis, legends present. Envelope @BSF for test 1 bearing 4 rises from ≈700 h.
  Report edits by parent: clarified post-shutdown wording; added leave-one-test-out rationale and
  only-4-failed-bearings variance caveat; explained test 1 H=24h count; pooled imbalance example
  (test 3, H=24h ≈0.6%) and imbalance-robust metrics (PR-AUC).
- 2026-09-17 native review: RDD on; START returned consent envelope (risk medium, 14 files, 4347
  lines; `data/Anteproyecto Completo V2.docx` excluded from the candidate). Pending user decision.
  User granted twice (first START went stale because the parent edited this document after
  negotiation). Lineage `review-0162f59796ea9c59` created (lens: review-reliability); capture failed
  before review: reviewer prompt ~1.9M tokens > 1M limit, caused by notebook 02 embedded outputs.
  Lineage left unused (abandon requires maintainer authorization); a new START follows T7.

- 2026-09-17: T7 completed. Added `marimo>=0.24.2` as a runtime dependency (kept `jupyterlab`,
  still used by nothing else in this repo but not asked to be removed — left in place since
  removing an unrelated dependency was out of scope). Ran `uv run marimo convert` on both
  `.ipynb` files, then hand-cleaned the output: replaced `Path("../data/...")` with
  `Path(__file__).resolve().parent.parent`-based `REPO_ROOT` in both notebooks so paths resolve
  correctly regardless of cwd; changed every figure cell's last statement from `plt.show()` to
  the bare `fig`/`_fig` expression so marimo captures and exports it; removed one genuinely
  unused `BEARING_COLORS` constant in notebook 01 (never referenced there, F841); removed a
  dead `if True:` wrapper block in notebook 02; wrapped long Spanish markdown paragraphs to
  ≤96 cols (soft-wrapped, no content change) to satisfy `E501`. `marimo convert` had already
  resolved all repeated-name collisions itself (cell-local `_test_id`, `_combo`, `_ax`, etc.),
  so `uv run marimo check notebooks/*.py` reports zero issues (no MB002).
  - Ruff: one narrowly-scoped ignore added, `[tool.ruff.lint.per-file-ignores]`
    `"notebooks/*.py" = ["B018"]` — marimo cells display their output via a bare trailing
    expression (a DataFrame or a `Figure` as the last statement), which ruff's B018 reads as
    "useless expression"; this is intentional/idiomatic marimo, not a bug, and unavoidable.
    mypy config unchanged (`files = ["src", "tests"]`, notebooks not type-checked, as required).
  - Makefile: `notebook` now runs `uv run marimo edit notebooks`; added a `notebooks` target
    that loops over `notebooks/*.py` and runs `marimo export ipynb ... --include-outputs
    --sort=top-down` and `marimo export html ...` into `build/notebooks/`. `.gitignore` gained
    `build/` and `__marimo__/`.
  - Verification: `make notebooks` → both notebooks exported without errors (`01_data_inventory
    .ipynb` 192K, `.html` 284K; `02_signal_exploration.ipynb` 4.0M, `.html` 4.1M — the 02 export
    is large only because `--include-outputs` embeds 7 PNG figures as base64, same as before).
    Numeric equivalence spot check (old `.ipynb` outputs read before deletion, vs. new exported
    `.ipynb` outputs): test file counts 2156/984/6324 ✓; test 3 documented-vs-observed
    4448/6324, +13.320116 days ✓; gap medians 600.0 s all tests, gaps-over-1.5×-median 24/0/4 ✓;
    post-shutdown counts test1/test2/test3 = 0/2/1 (rows for test_id 2 and 3 only, RMS
    0.004018/0.002124/0.003966) ✓; test 1 x/y RMS correlation per bearing 0.980105/0.979993/
    0.991811/0.934093 ✓; chosen envelope bands (4300-5800, 3300-4800, 2300-3800, 3800-5300 Hz)
    ✓; onset table (all 12 rows, e.g. test1-bearing3 66.05h/11.87h/NaN) ✓; class-balance table
    (all 12 rows, e.g. test3 H=24h → 145/6323 = 0.022932) ✓ — every value matched exactly.
    Visually inspected the 7-panel envelope/spectrum figure and the test2/3 degradation-trends
    figure (both rendered from the exported `.ipynb`): correct bearing colors, legends, and the
    manually-wrapped multi-line `suptitle` render correctly with no overlap.
  - `make check` → ruff format/check clean, mypy clean (11 source files, `src`+`tests` only),
    `37 passed`. `git status --short` → no `.ipynb`, `build/`, or `__marimo__/` tracked or
    visible (both are gitignored); old `.ipynb` files removed.
    `notebooks/01_data_inventory.py` 12K, `notebooks/02_signal_exploration.py` 36K.
  - Updated `notebooks/README.md` (marimo workflow, `make notebook`/`make notebooks`), the root
    `README.md` development-commands list, and `reports/eda_nasa_ims.md`'s notebook-file
    reference (`.ipynb` → `.py`, mentions `make notebooks` exports); no report findings changed.
  - Surprising: none functionally — `marimo convert` already handled MB002 name-collision
    avoidance well by itself; the only real manual work was path robustness, matplotlib
    display convention (`fig` vs `plt.show()`), and line-length wrapping of prose.

## Next step

All tasks (T1-T7) complete and verified. Acceptance criteria met: `make check` passes, both
notebooks export cleanly via `make notebooks` with numerically-equivalent outputs to the prior
Jupyter versions, and every numeric claim in `reports/eda_nasa_ims.md` cites its
notebook/section. Open decisions for the team are listed in the report's "Limitaciones y
preguntas abiertas" section (label horizon, test 1 time axis, whether to keep the full test 3,
how to treat test 1 bearing 4 in labeling). Previous lineage `review-0162f59796ea9c59` was left
unused (context overflow from embedded notebook outputs, not abandoned).

Native review 2026-09-17: lineage `review-cea46f9f22b3eb63` (lens review-reliability, 18 files,
3492 lines, `.docx` excluded) → **approved**, acknowledged, authority burned. No blocking
findings. Delivery (commit/push) remains a separate user decision.

## Follow-ups (non-blocking review findings, not started)

- [ ] F1 (R3-001, warning) — `summarize_inventory` ignores unreadable rows when computing
  `rows_per_file_consistent` and exposes no read-error/non-finite counts; add them + test. The
  report's "no corrupt files" claim is still supported by the T3 run (0 read errors, 0 non-finite),
  but the notebook flag alone does not prove it.
- [ ] F2 (R3-002, warning) — `list_snapshots` aborts the whole run on one non-timestamp file name;
  decide skip-with-warning vs fail-fast, document and test.
- [ ] F3 (R3-003, warning) — `flag_post_shutdown` flags any low snapshot, not only a trailing run;
  NaN baseline silently flags nothing. Restrict to trailing run (or rename) + tests.
- [ ] F4 (R3-004) — validate `detect_onset` parameters (`persistence`, `baseline_n` > 0).
- [ ] F5 (R3-005) — test the `ProcessPoolExecutor` path matches the sequential path.
- [ ] F6 (R3-006) — minimum-length guard in `amplitude_spectrum`.
- [ ] F7 (R3-007) — derive notebook markdown numbers from computed values (f-strings/asserts).

Next phase (out of scope here): CRISP-DM Phase III (feature matrix, windowing, labels).

## Log

- L1 (2026-09-17) — Original request: CRISP-DM Phase II EDA of NASA IMS (see Scope).
- L2 (2026-10-18) — User: "Es el proyecto v1 la idea seria sacar las Fases 2 y 3 Hoy."
- L3 (2026-10-18) — User: "Vamos a trabajar sobre la copia, quiero que antes de que demos por
  termianda la Fase 2 le hagas un review completo para ver si esta bien."
  Review (3 read-only agents: claims, code, coverage vs "Proyecto V1"): make check green,
  inventory/gap/imbalance/onset numbers reproduce. Blockers: amplitude indicators not
  mean-removed (test 1 DC offset ≈ -0.115 g ≈ std; RMS trend +54% vs std +104% on ch7);
  outer-race envelope peak at 230.5-231.4 Hz vs BPFO 236.4 (ratio ≈ 20480/20000, unresolved);
  report errors (bearing 4 kurtosis episode, BSF ~2× not 3-4×, pauses in last 24 h not 13 h,
  test 3 raw spectrum <60 Hz); integrity counts not shown; leakage argument uses feature
  correlation; segment stability missing; no CWRU evidence in repo (open product decision).
- L4 (2026-10-18) — User: "Sí, déjalo así y arranca con las correcciones". Baseline committed
  on `feat/eda-nasa-ims`: 0a010f8 (code), 75f1a45 (notebooks/report), e9653c1 (data-auditor).
  Review assess base 3ca6647 committed-only → medium, review_due false (under_budget).
- L5 (2026-10-18) — T8 36a60fd (RED: offset sine rms 3.32 vs 1.41; GREEN 9 passed; CSV
  regenerated, centered stats unchanged, rms==std, t1 ch7 RMS +107%). T9+T12 fe86a59 (RED 5 +7
  failing; GREEN 24 spectral, 7 inventory, 10 ims). T9 finding: all observed fault lines 1-3%
  below kinematic (BPFO -2.1/-2.5%, BPFI -1.0%, BSF -1.6%), consistent with slip at 20 kHz;
  20.48 kHz would fit BPFO (+0.3%) but not BPFI (+1.4%); unresolved, kept 20 kHz +
  FAULT_REL_TOLERANCE 0.03. T10/T11: notebooks + report rewritten from executed outputs
  (bearing 4 kurtosis >8 in 190/381 snapshots 150-72 h before end, 0 in last 72 h; RMS
  autocorr lag1 0.95-0.995; segment CV RMS 2-5.5%; test 1 baseline RMS +20-23%; onset table
  changed: t1b3 34.2 h, t2 74.5/73.5/69.5 h, t3 133.2/41.2/41.2 h).
- L6 (2026-10-18) — data-auditor re-audit of 2b3df96: numbers reproduce; 3 blockers. B1 inner
  race weaker than claimed (envelope 1.4x, late kurtosis 146 -> 6 >8 after trimming 10 extreme
  samples). B2 slip reasoning wrong for BPFI (BPFO+BPFI = n*fr; slip raises BPFI). B3 +/-3%
  window was data-informed, not a priori; 7-19% of evolution points peak on the window edge.
  User granted native review (lens readability+reliability) for target sha256:374a2cdd...;
  deferred until the correction batch lands, since the candidate would change. One correction
  batch: B1-B3 + advisories (ADC saturation counts in inventory, n_saturated: 3/2/4 files,
  9/47/89 samples; operating hours 345.3/163.8/1053.4; descriptive stats; trimmed-kurtosis
  robustness; max-in-window 59.4 at 92.6 h; shape-factor wording; operating-hours positives
  146/291/435 for test 1; cadence wording in notebook 01).
- L7 (2026-10-18) — Recheck by data-auditor: B1-B3 fixed (residual wording: trends paragraph
  still lists bearing 3 kurtosis as "subida clara"; "+0.3%" holds for test 2 only (test 3
  -0.17%); "shaft lines too weak" has no notebook cell). Native review: full branch candidate
  failed lens_context_budget_exceeded; reviewed slice base 75f1a45 (tree 2c9a779) .. 0f7d37e,
  lineage review-c7023b62c51a70d4, lenses readability+reliability -> approved, acknowledged.
  Non-blocking advisories (later work): R2-001 operating-hours rule duplicated in nb01/nb02;
  R2-002 kurtosis 8 / 72 h / 150 h literals; R2-003/R3-002 hardcoded numbers in markdown (F7);
  R2-004 fault_peak docstring "2.2%"; R3-001 summarize_inventory KeyError on stale inventory
  without n_saturated; R3-003 SATURATION_LEVEL 4.99 vs ±4.999 boundary untested.
- L8 (2026-10-19) — User: "Hay que revisar lo que el obtuvo y si cuadra." data-auditor compared
  origin/eda-santiago with raw data and our reports: his tables reproduce; wrong conclusions (test 3
  lead time reversed, "siempre el primero", leaky provisional labels, RMS with DC, kurtosis max).
- L9 (2026-10-19) — User: "Incluso si alo no integrar como tal sus notebooks si no integrar lo util
  en los nuetsros para mantener un formato." Delivery (earlier): "stack a main pero usando merge
  foreward". T14 added; lands on feat/eda-nasa-ims, then merged forward into feat/data-preparation.
- L10 (2026-10-19) — T14 a751f86 (inventory content_sha256 + duplicate_content_count, 5 tests
  RED->GREEN; 0 content duplicates in 9464 files; short intervals 9/0/2; 13 s pair not identical;
  flat-signal check = same 3 post-shutdown files; Spearman x/y RMS 0.80-0.96 vs kurtosis/crest
  0.15-0.78; rails -5.000/+4.998). Merged forward into feat/data-preparation (8f20e58, 96 passed).
  Native review slice cb5fc91..8f20e58 approved+acknowledged; advisories: hardcoded phrases in nb01
  readings, duplicate-count logic reimplemented in nb01, KeyError on stale inventory without
  content_sha256, second read_bytes outside try/except. Full-branch candidate: budget exceeded.
