import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploración de señales — NASA IMS

    Cuaderno 02: formas de onda crudas, distribuciones de amplitud, tendencias de degradación por
    rodamiento, correlación entre ejes e indicadores, espectros y espectros de envolvente con las
    frecuencias características de falla, evolución temporal de la amplitud en la frecuencia
    documentada, una estimación exploratoria del inicio de la degradación, y las implicaciones para
    el etiquetado, las ventanas y la partición del conjunto de datos. Se apoya en
    `data/interim/ims_snapshot_indicators.csv`, en los datos crudos (`data/raw/`, solo lectura) y en
    `src/bearing_predictive_maintenance/spectral.py`.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from bearing_predictive_maintenance import ims
    from bearing_predictive_maintenance.spectral import (
        amplitude_spectrum,
        band_amplitude,
        detect_onset,
        envelope_spectrum,
        flag_post_shutdown,
    )

    # Fixed color per bearing (never cycled), shared with notebook 01.
    BEARING_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
    EARLY_COLOR = "#999999"  # neutral gray: always the earlier/healthier capture

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#cccccc",
            "axes.edgecolor": "#444444",
        }
    )

    REPO_ROOT = Path(__file__).resolve().parent.parent
    RAW_ROOT = REPO_ROOT / "data" / "raw"
    INTERIM_DIR = REPO_ROOT / "data" / "interim"
    FIGURES_DIR = REPO_ROOT / "reports" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return (
        BEARING_COLORS,
        EARLY_COLOR,
        FIGURES_DIR,
        INTERIM_DIR,
        RAW_ROOT,
        amplitude_spectrum,
        band_amplitude,
        detect_onset,
        envelope_spectrum,
        flag_post_shutdown,
        ims,
        np,
        pd,
        plt,
    )


@app.cell
def _(INTERIM_DIR, ims, pd):
    indicators = pd.read_csv(
        INTERIM_DIR / "ims_snapshot_indicators.csv",
        dtype={"test_id": str},
        parse_dates=["timestamp"],
    )
    FREQS = ims.characteristic_fault_frequencies()
    FREQ_LOOKUP = {"FTF": FREQS.ftf, "BPFO": FREQS.bpfo, "BPFI": FREQS.bpfi, "BSF": FREQS.bsf}

    FAILED_COMBOS = [
        {
            "test": "1",
            "bearing": 3,
            "axis": "x",
            "target_freq_name": "BPFI",
            "label": "Test 1 - Rodamiento 3 (pista interna)",
        },
        {
            "test": "1",
            "bearing": 4,
            "axis": "x",
            "target_freq_name": "BSF",
            "label": "Test 1 - Rodamiento 4 (elemento rodante)",
        },
        {
            "test": "2",
            "bearing": 1,
            "axis": "single",
            "target_freq_name": "BPFO",
            "label": "Test 2 - Rodamiento 1 (pista externa)",
        },
        {
            "test": "3",
            "bearing": 3,
            "axis": "single",
            "target_freq_name": "BPFO",
            "label": "Test 3 - Rodamiento 3 (pista externa)",
        },
    ]
    return FAILED_COMBOS, FREQS, FREQ_LOOKUP, indicators


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Última captura válida por test (excluyendo capturas posteriores al apagado)
    """)
    return


@app.cell
def _(flag_post_shutdown, ims, indicators, pd):
    max_rms = (
        indicators.groupby(["test_id", "file", "timestamp", "elapsed_hours"])["rms"]
        .max()
        .reset_index()
    )
    last_valid = {}
    first_snapshot = {}
    for _test_id in ims.TEST_REGISTRY:
        test_max_rms = (
            max_rms[max_rms["test_id"] == _test_id].sort_values("timestamp").reset_index(drop=True)
        )
        flags = flag_post_shutdown(
            test_max_rms["rms"], baseline_n=100, fraction_of_baseline_median=0.1
        )
        valid = test_max_rms[~flags]
        last_valid[_test_id] = valid.iloc[-1]
        first_snapshot[_test_id] = test_max_rms.iloc[0]
    pd.DataFrame(last_valid).T
    return first_snapshot, last_valid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En todo lo que sigue, **"temprano"** es el primer snapshot registrado del test y **"tardío"** es
    la última captura *válida* (excluyendo las capturas posteriores al apagado identificadas en el
    cuaderno 01).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Formas de onda crudas: temprano vs. tardío
    """)
    return


@app.cell
def _(
    BEARING_COLORS,
    EARLY_COLOR,
    FAILED_COMBOS,
    FIGURES_DIR,
    RAW_ROOT,
    first_snapshot,
    ims,
    last_valid,
    np,
    plt,
):
    def get_channel(test_id: str, bearing: int, axis: str) -> int:
        spec = ims.TEST_REGISTRY[test_id]
        for mapping in spec.channel_map:
            if mapping.bearing == bearing and mapping.axis == axis:
                return mapping.channel
        raise KeyError((test_id, bearing, axis))

    def load_channel_signal(test_id: str, filename: str, bearing: int, axis: str) -> np.ndarray:
        spec = ims.TEST_REGISTRY[test_id]
        path = RAW_ROOT / spec.raw_relative_path / filename
        array = ims.load_snapshot(path)
        channel = get_channel(test_id, bearing, axis)
        return np.asarray(array[:, channel - 1], dtype=np.float64)

    COMBO_SIGNALS = {}
    for _combo in FAILED_COMBOS:
        _test_id = _combo["test"]
        early_file = first_snapshot[_test_id]["file"]
        late_file = last_valid[_test_id]["file"]
        COMBO_SIGNALS[_combo["label"]] = {
            "early_file": early_file,
            "late_file": late_file,
            "early": load_channel_signal(_test_id, early_file, _combo["bearing"], _combo["axis"]),
            "late": load_channel_signal(_test_id, late_file, _combo["bearing"], _combo["axis"]),
        }
    N_SAMPLES_SHOWN = 2000
    t_short = np.arange(N_SAMPLES_SHOWN) / ims.SAMPLING_RATE_HZ
    _fig, _axes = plt.subplots(len(FAILED_COMBOS), 2, figsize=(11, 12))
    for _row, _combo in enumerate(FAILED_COMBOS):
        _signals = COMBO_SIGNALS[_combo["label"]]
        _color = BEARING_COLORS[_combo["bearing"]]
        _axes[_row, 0].plot(
            t_short, _signals["early"][:N_SAMPLES_SHOWN], color=EARLY_COLOR, linewidth=1.0
        )
        _axes[_row, 0].set_title(
            f"{_combo['label']}\ntemprano ({_signals['early_file']})", fontsize=8
        )
        _axes[_row, 1].plot(
            t_short, _signals["late"][:N_SAMPLES_SHOWN], color=_color, linewidth=1.0
        )
        _axes[_row, 1].set_title(
            f"{_combo['label']}\ntardío ({_signals['late_file']})", fontsize=8
        )  # 0.1 s at 20 kHz, kept short so the waveform stays legible
        for _ax in _axes[_row]:
            _ax.set_ylabel("amplitud (g)")
    _axes[-1, 0].set_xlabel("tiempo (s)")
    _axes[-1, 1].set_xlabel("tiempo (s)")
    _fig.suptitle(
        "Formas de onda crudas (0.1 s): temprano (gris) vs. tardío (color del rodamiento)"
    )
    _fig.tight_layout()
    _fig.savefig(FIGURES_DIR / "02_raw_waveforms_early_late.png", dpi=150)
    _fig
    return COMBO_SIGNALS, load_channel_signal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distribuciones de amplitud: temprano vs. tardío
    """)
    return


@app.cell
def _(
    BEARING_COLORS,
    COMBO_SIGNALS,
    EARLY_COLOR,
    FAILED_COMBOS,
    FIGURES_DIR,
    plt,
):
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 8))
    for _ax, _combo in zip(_axes.flat, FAILED_COMBOS, strict=True):
        _signals = COMBO_SIGNALS[_combo["label"]]
        _color = BEARING_COLORS[_combo["bearing"]]
        _ax.hist(
            _signals["early"], bins=80, color=EARLY_COLOR, alpha=0.6, density=True, label="temprano"
        )
        _ax.hist(_signals["late"], bins=80, color=_color, alpha=0.6, density=True, label="tardío")
        _ax.set_title(_combo["label"], fontsize=9)
        _ax.set_xlabel("amplitud (g)")
        _ax.set_ylabel("densidad")
        _ax.legend(fontsize=8)
    _fig.suptitle("Distribución de amplitud: temprano vs. tardío")
    _fig.tight_layout()
    _fig.savefig(FIGURES_DIR / "02_amplitude_distributions.png", dpi=150)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tendencias de degradación: RMS, curtosis y factor de cresta, por rodamiento

    El test 1 tiene huecos de tiempo grandes (pausas de operación, ver cuaderno 01). Usar
    directamente las horas de reloj transcurridas mezcla tiempo de funcionamiento real con tiempo de
    inactividad del banco, lo que puede distorsionar la forma aparente de una tendencia de
    degradación (tramos planos que en realidad son huecos, no estabilidad del rodamiento). Por eso,
    para el test 1 se muestran **ambos ejes** — horas de reloj y el índice secuencial de snapshot
    (que refleja solo el orden de operación, ignorando cuánto duró cada pausa) — para decidir con
    evidencia cuál es más adecuado para ventanas y modelado. Los tests 2 y 3 no tienen huecos
    relevantes (ver cuaderno 01), así que se muestran solo con horas de reloj.
    """)
    return


@app.cell
def _(indicators, pd):
    METRICS = ["rms", "kurtosis", "crest_factor"]
    METRIC_LABELS = {"rms": "RMS", "kurtosis": "Curtosis", "crest_factor": "Factor de cresta"}

    def bearing_trend(test_id: str, metric: str) -> pd.DataFrame:
        test_data = indicators[indicators["test_id"] == test_id]
        grouped = (
            test_data.groupby(["file", "elapsed_hours", "bearing"])[metric].mean().reset_index()
        )
        return grouped.sort_values("elapsed_hours")

    test1_files = (
        indicators[indicators["test_id"] == "1"][["file", "elapsed_hours"]]
        .drop_duplicates()
        .sort_values("elapsed_hours")
        .reset_index(drop=True)
    )
    file_to_index = dict(zip(test1_files["file"], test1_files.index, strict=True))
    return METRICS, METRIC_LABELS, bearing_trend, file_to_index


@app.cell
def _(
    BEARING_COLORS,
    FIGURES_DIR,
    METRICS,
    METRIC_LABELS,
    bearing_trend,
    file_to_index,
    plt,
):
    _fig, _axes = plt.subplots(3, 2, figsize=(12, 10))
    for _row, _metric in enumerate(METRICS):
        _trend = bearing_trend("1", _metric)
        _trend["snapshot_index"] = _trend["file"].map(file_to_index)
        for _bearing in range(1, 5):
            _bt = _trend[_trend["bearing"] == _bearing]
            _axes[_row, 0].plot(
                _bt["elapsed_hours"],
                _bt[_metric],
                color=BEARING_COLORS[_bearing],
                label=f"rodamiento {_bearing}",
                linewidth=1.4,
            )
            _axes[_row, 1].plot(
                _bt["snapshot_index"],
                _bt[_metric],
                color=BEARING_COLORS[_bearing],
                label=f"rodamiento {_bearing}",
                linewidth=1.4,
            )
        _axes[_row, 0].set_ylabel(METRIC_LABELS[_metric])
    _axes[0, 0].set_title("horas de reloj transcurridas")
    _axes[0, 1].set_title("índice secuencial de snapshot")
    _axes[0, 0].legend(fontsize=7, loc="upper left")
    _axes[-1, 0].set_xlabel("horas de reloj")
    _axes[-1, 1].set_xlabel("índice de snapshot")
    _fig.suptitle("Test 1: tendencias de degradación por rodamiento (dos ejes temporales)")
    _fig.tight_layout()
    _fig.savefig(FIGURES_DIR / "02_degradation_trends_test1.png", dpi=150)
    _fig
    return


@app.cell
def _(BEARING_COLORS, FIGURES_DIR, METRICS, METRIC_LABELS, bearing_trend, plt):
    _fig, _axes = plt.subplots(3, 2, figsize=(12, 10))
    for _row, _metric in enumerate(METRICS):
        for col, _test_id in enumerate(["2", "3"]):
            _trend = bearing_trend(_test_id, _metric)
            for _bearing in range(1, 5):
                _bt = _trend[_trend["bearing"] == _bearing]
                _axes[_row, col].plot(
                    _bt["elapsed_hours"],
                    _bt[_metric],
                    color=BEARING_COLORS[_bearing],
                    label=f"rodamiento {_bearing}",
                    linewidth=1.4,
                )
            if _row == 0:
                _axes[_row, col].set_title(f"Test {_test_id}")
        _axes[_row, 0].set_ylabel(METRIC_LABELS[_metric])
    _axes[0, 0].legend(fontsize=7, loc="upper left")
    _axes[-1, 0].set_xlabel("horas de reloj")
    _axes[-1, 1].set_xlabel("horas de reloj")
    _fig.suptitle("Test 2 y test 3: tendencias de degradación por rodamiento")
    _fig.tight_layout()
    _fig.savefig(FIGURES_DIR / "02_degradation_trends_test2_test3.png", dpi=150)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Decisión y justificación:** para el test 1, la forma de la tendencia (dónde empieza a subir la
    curtosis/RMS del rodamiento 3) es visualmente muy similar en ambos ejes, pero el eje de horas de
    reloj **comprime** el tramo final (donde están las pausas largas) y puede sugerir una subida más
    abrupta de lo que realmente es en tiempo de operación. Para cuantificar *cuánto tiempo de
    operación* precede a la falla (relevante para definir ventanas de predicción), se recomienda
    usar el **índice secuencial de snapshot** (o una versión de "horas de operación" que excluya los
    huecos), no las horas de reloj crudas. Las horas de reloj siguen siendo relevantes si se
    quisiera correlacionar la degradación con el calendario (p. ej. turnos de trabajo), pero no es
    el caso aquí.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlación entre ejes x/y (test 1) y entre indicadores
    """)
    return


@app.cell
def _(indicators, np, pd):
    test1 = indicators[indicators["test_id"] == "1"]
    xy_corr_rows = []
    for _bearing in range(1, 5):
        x_series = (
            test1[(test1["bearing"] == _bearing) & (test1["axis"] == "x")]
            .sort_values("elapsed_hours")["rms"]
            .to_numpy()
        )
        y_series = (
            test1[(test1["bearing"] == _bearing) & (test1["axis"] == "y")]
            .sort_values("elapsed_hours")["rms"]
            .to_numpy()
        )
        r = float(np.corrcoef(x_series, y_series)[0, 1])
        xy_corr_rows.append({"rodamiento": _bearing, "correlacion_rms_x_y": r})
    pd.DataFrame(xy_corr_rows)
    return (test1,)


@app.cell
def _(FIGURES_DIR, plt, test1):
    metric_cols = [
        "rms",
        "std",
        "peak",
        "peak_to_peak",
        "skewness",
        "kurtosis",
        "crest_factor",
        "impulse_factor",
        "shape_factor",
        "energy",
    ]
    corr_matrix = test1[metric_cols].corr()
    _fig, _ax = plt.subplots(figsize=(7, 6))
    im = _ax.imshow(corr_matrix.abs(), cmap="Blues", vmin=0, vmax=1)
    _ax.set_xticks(range(len(metric_cols)))
    _ax.set_xticklabels(metric_cols, rotation=45, ha="right")
    _ax.set_yticks(range(len(metric_cols)))
    _ax.set_yticklabels(metric_cols)
    for i in range(len(metric_cols)):
        for j in range(len(metric_cols)):
            value = corr_matrix.values[i, j]
            text_color = "white" if abs(value) > 0.6 else "black"
            _ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)
    _ax.set_title("Correlación entre indicadores (test 1)")
    _fig.colorbar(im, ax=_ax, label="|correlación de Pearson|")
    _fig.tight_layout()
    _fig.savefig(FIGURES_DIR / "02_indicator_correlation.png", dpi=150)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El color del mapa codifica |r| (correlación en valor absoluto, escala de azul secuencial); el
    texto en cada celda muestra el valor de r con su signo. Una correlación alta (en valor absoluto)
    entre dos indicadores señala redundancia: aportan información similar, por lo que incluir ambos
    como variables de un modelo no necesariamente añade poder predictivo y puede introducir
    colinealidad. Esto se usa más abajo para priorizar qué familias de características son más
    prometedoras de cara al modelado (Fase III).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Espectro y espectro de envolvente: temprano vs. tardío
    """)
    return


@app.cell
def _(COMBO_SIGNALS, FAILED_COMBOS, amplitude_spectrum, ims, np, pd):
    def suggest_envelope_band(
        early_signal: np.ndarray,
        late_signal: np.ndarray,
        fs: float,
        low_bound: float = 300.0,
        high_bound: float = 8000.0,
        bin_width: float = 500.0,
    ) -> tuple[float, float]:
        """Pick a resonance band by finding where late-vs-early spectral energy grows most.

        Simple, data-driven rationale (not a fixed literature value): scan the raw
        spectrum in fixed-width bins between ``low_bound`` and ``high_bound``, and pick
        the bin with the largest (late - early) amplitude increase. The returned band is
        that bin widened by one bin on each side, to keep the band-pass filter from
        being pathologically narrow.
        """
        _, amps_early = amplitude_spectrum(early_signal, fs)
        freqs_late, amps_late = amplitude_spectrum(late_signal, fs)
        diff = amps_late - amps_early
        starts = np.arange(low_bound, high_bound, bin_width)
        increases = [
            diff[(freqs_late >= start) & (freqs_late < start + bin_width)].sum() for start in starts
        ]
        best_start = float(starts[int(np.argmax(increases))])
        band_low = max(low_bound, best_start - bin_width)
        band_high = min(high_bound, best_start + 2 * bin_width)
        return (band_low, band_high)

    CHOSEN_BANDS = {
        combo["label"]: suggest_envelope_band(
            COMBO_SIGNALS[combo["label"]]["early"],
            COMBO_SIGNALS[combo["label"]]["late"],
            ims.SAMPLING_RATE_HZ,
        )
        for combo in FAILED_COMBOS
    }
    pd.DataFrame([{"combo": k, "banda_hz": v} for k, v in CHOSEN_BANDS.items()])
    return (CHOSEN_BANDS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Justificación de la banda de envolvente:** en vez de fijar una banda de resonancia por
    referencia externa, se calcula dónde crece más la energía espectral cruda entre la captura
    temprana y la tardía (bins de 500 Hz entre 300 y 8000 Hz), y se usa esa región (ensanchada) como
    banda de filtrado pasa-banda antes de la envolvente de Hilbert. Es una heurística simple y
    explícita, no una referencia de literatura sin verificar.
    """)
    return


@app.cell
def _(
    BEARING_COLORS,
    CHOSEN_BANDS,
    COMBO_SIGNALS,
    EARLY_COLOR,
    FAILED_COMBOS,
    FIGURES_DIR,
    FREQS,
    FREQ_LOOKUP,
    amplitude_spectrum,
    envelope_spectrum,
    ims,
    plt,
):
    _fig, _axes = plt.subplots(len(FAILED_COMBOS), 2, figsize=(13, 15))
    for _row, _combo in enumerate(FAILED_COMBOS):
        _signals = COMBO_SIGNALS[_combo["label"]]
        _color = BEARING_COLORS[_combo["bearing"]]
        _band = CHOSEN_BANDS[_combo["label"]]
        freqs_e, amps_e = amplitude_spectrum(_signals["early"], ims.SAMPLING_RATE_HZ)
        freqs_l, amps_l = amplitude_spectrum(_signals["late"], ims.SAMPLING_RATE_HZ)
        ax_raw = _axes[_row, 0]
        ax_raw.plot(freqs_e, amps_e, color=EARLY_COLOR, linewidth=1.0, label="temprano")
        ax_raw.plot(freqs_l, amps_l, color=_color, linewidth=1.0, label="tardío")
        ax_raw.set_xlim(0, 1000)
        ax_raw.set_title(f"{_combo['label']}\nespectro crudo", fontsize=8)
        env_freqs_e, env_amps_e = envelope_spectrum(
            _signals["early"], ims.SAMPLING_RATE_HZ, band=_band
        )
        env_freqs_l, env_amps_l = envelope_spectrum(
            _signals["late"], ims.SAMPLING_RATE_HZ, band=_band
        )
        ax_env = _axes[_row, 1]
        ax_env.plot(env_freqs_e, env_amps_e, color=EARLY_COLOR, linewidth=1.0, label="temprano")
        ax_env.plot(env_freqs_l, env_amps_l, color=_color, linewidth=1.0, label="tardío")
        ax_env.set_xlim(0, 500)
        ax_env.set_title(
            f"{_combo['label']}\nenvolvente (banda {_band[0]:.0f}-{_band[1]:.0f} Hz)", fontsize=8
        )
        for _ax in (ax_raw, ax_env):
            for name, freq in FREQ_LOOKUP.items():
                is_target = name == _combo["target_freq_name"]
                _ax.axvline(
                    freq,
                    color="#c0392b" if is_target else "#aaaaaa",
                    linestyle="--" if is_target else ":",
                    linewidth=1.3 if is_target else 0.8,
                )
                if is_target:
                    _ax.annotate(
                        name,
                        xy=(freq, _ax.get_ylim()[1]),
                        xytext=(2, -2),
                        textcoords="offset points",
                        fontsize=7,
                        color="#c0392b",
                        va="top",
                    )
            _ax.axvline(2 * FREQS.bsf, color="#aaaaaa", linestyle=":", linewidth=0.8)
            _ax.set_xlabel("frecuencia (Hz)")
            _ax.set_ylabel("amplitud")
        _axes[_row, 0].legend(fontsize=7, loc="upper right")
    _fig.suptitle(
        "Espectro y envolvente: temprano vs. tardío\n"
        "línea roja discontinua = frecuencia característica documentada, "
        "líneas grises punteadas = otras (FTF/BPFO/BPFI/BSF/2xBSF)"
    )
    _fig.tight_layout()
    _fig.savefig(FIGURES_DIR / "02_spectra_envelope_early_late.png", dpi=150)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lectura de la figura anterior:**

    - **Test 2, rodamiento 1 (BPFO)** y **test 3, rodamiento 3 (BPFO)**: el espectro crudo tardío
    tiene picos dominantes que **no** coinciden con BPFO (parecen resonancias estructurales cerca de
    700-900 Hz), pero el espectro de envolvente muestra un pico limpio y dominante exactamente en
    BPFO — el caso de manual de libro de texto de por qué se necesita la envolvente: el defecto está
    modulando una resonancia, no generando energía directa y visible en el espectro crudo.
    - **Test 1, rodamiento 3 (BPFI)**: el espectro de envolvente tardío muestra actividad elevada en
    general (incluyendo cerca de BPFI), pero el pico en BPFI no es tan limpio ni dominante como en
    los casos de pista externa; hay varios picos de magnitud comparable en otras frecuencias.
    - **Test 1, rodamiento 4 (BSF, elemento rodante) — hallazgo honesto:** el pico dominante del
    espectro de envolvente tardío está cerca de 65 Hz, **no** en BSF (~140 Hz) ni en 2×BSF. Es
    decir, tampoco en el dominio de la frecuencia hay un pico limpio y aislado en la frecuencia
    documentada del defecto. Sin embargo, la sección siguiente (evolución temporal) muestra que la
    amplitud específicamente en BSF **sí** tiene una tendencia ascendente real a lo largo del test
    (ver más abajo) — solo que no se manifiesta como un tono limpio y dominante como en los otros
    tres casos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evolución temporal de la amplitud en la frecuencia característica documentada
    """)
    return


@app.cell
def _(
    CHOSEN_BANDS,
    FAILED_COMBOS,
    FREQ_LOOKUP,
    RAW_ROOT,
    band_amplitude,
    envelope_spectrum,
    ims,
    indicators,
    last_valid,
    load_channel_signal,
    pd,
):
    SUBSAMPLE_STEP = {"1": 5, "2": 3, "3": 15}
    elapsed_lookup = {
        test_id: dict(
            zip(
                indicators[indicators["test_id"] == test_id]["file"],
                indicators[indicators["test_id"] == test_id]["elapsed_hours"],
                strict=True,
            )
        )
        for test_id in ims.TEST_REGISTRY
    }
    evolution_frames = []
    for _combo in FAILED_COMBOS:
        _test_id = _combo["test"]
        step = SUBSAMPLE_STEP[_test_id]
        _band = CHOSEN_BANDS[_combo["label"]]
        target = FREQ_LOOKUP[_combo["target_freq_name"]]
        paths = ims.list_snapshots(RAW_ROOT, _test_id)
        last_valid_file = last_valid[_test_id]["file"]
        last_valid_pos = next((i for i, p in enumerate(paths) if p.name == last_valid_file))
        subsampled = paths[: last_valid_pos + 1 : step]
        rows = []
        for path in subsampled:
            signal_ = load_channel_signal(_test_id, path.name, _combo["bearing"], _combo["axis"])
            freqs_, amps_ = envelope_spectrum(signal_, ims.SAMPLING_RATE_HZ, band=_band)
            amp_at_target = band_amplitude(freqs_, amps_, target=target, tolerance=5.0)
            rows.append(
                {
                    "combo": _combo["label"],
                    "test": _test_id,
                    "file": path.name,
                    "elapsed_hours": elapsed_lookup[_test_id][path.name],
                    "envelope_amp": amp_at_target,
                }
            )
        evolution_frames.append(pd.DataFrame(rows))
    evolution = pd.concat(evolution_frames, ignore_index=True)
    print(
        f"Submuestreo: test1 cada {SUBSAMPLE_STEP['1']}, "
        f"test2 cada {SUBSAMPLE_STEP['2']}, "
        f"test3 cada {SUBSAMPLE_STEP['3']} archivos"
    )
    evolution.groupby("combo").size().rename("n_puntos_evaluados")
    return (evolution,)


@app.cell
def _(BEARING_COLORS, FAILED_COMBOS, FIGURES_DIR, evolution, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 8))
    for _ax, _combo in zip(_axes.flat, FAILED_COMBOS, strict=True):
        _data = evolution[evolution["combo"] == _combo["label"]].sort_values("elapsed_hours")
        _ax.plot(
            _data["elapsed_hours"],
            _data["envelope_amp"],
            color=BEARING_COLORS[_combo["bearing"]],
            marker="o",
            markersize=2,
            linewidth=1.0,
        )
        _ax.set_title(f"{_combo['label']} (falla documentada)", fontsize=9)
        _ax.set_xlabel("horas transcurridas")
        _ax.set_ylabel(f"amplitud envolvente @ {_combo['target_freq_name']}")
    _fig.suptitle(
        "Evolución de la amplitud de envolvente en la frecuencia característica documentada"
    )
    _fig.tight_layout()
    _fig.savefig(FIGURES_DIR / "02_envelope_amplitude_evolution.png", dpi=150)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Hallazgo clave (rodamiento 4, test 1):** aunque el espectro de envolvente no muestra un pico
    limpio y dominante en BSF (panel anterior), la amplitud **específicamente** en BSF sí crece a lo
    largo del test: de una línea base ≈0.002-0.004 a valores ≈0.008-0.012 en las últimas ~100 horas,
    un aumento real de ≈3-4x, aunque ruidoso (sin la subida limpia y monótona que muestran los otros
    tres rodamientos). Esto contrasta con la curtosis (dominio del tiempo), que no mostraba ninguna
    subida para este rodamiento (T3): **el defecto de elemento rodante sí deja una huella medible,
    pero solo en el dominio de la frecuencia y de forma más débil y difusa que los defectos de
    pista** — una distinción importante para no descartar este caso como "sin señal" al diseñar
    características para el modelado.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inicio exploratorio de degradación (onset), por rodamiento fallado
    """)
    return


@app.cell
def _(FAILED_COMBOS, detect_onset, evolution, pd):
    ONSET_CONFIGS = [
        {"n_sigma": 2.5, "persistence": 3},
        {"n_sigma": 3.0, "persistence": 5},
        {"n_sigma": 4.0, "persistence": 10},
    ]
    onset_rows = []
    for _combo in FAILED_COMBOS:
        label = _combo["label"]
        combo_evolution = evolution[evolution["combo"] == label]
        _data = combo_evolution.sort_values("elapsed_hours").reset_index(drop=True)
        series = _data["envelope_amp"]
        baseline_n = max(5, len(series) // 5)
        _last_elapsed = _data["elapsed_hours"].iloc[-1]
        for cfg in ONSET_CONFIGS:
            onset_idx = detect_onset(
                series,
                baseline_n=baseline_n,
                n_sigma=cfg["n_sigma"],
                persistence=cfg["persistence"],
            )
            if onset_idx is not None:
                onset_elapsed = _data["elapsed_hours"].iloc[onset_idx]
                hours_before_end = _last_elapsed - onset_elapsed
            else:
                onset_elapsed = None
                hours_before_end = None
            onset_rows.append(
                {
                    "combo": label,
                    "n_sigma": cfg["n_sigma"],
                    "persistence": cfg["persistence"],
                    "baseline_n": baseline_n,
                    "onset_elapsed_hours": onset_elapsed,
                    "horas_antes_del_final_valido": hours_before_end,
                }
            )
    onset_table = pd.DataFrame(onset_rows)
    onset_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Este detector es **exploratorio**: un umbral simple media+n_sigma sobre la propia serie, con una
    persistencia mínima de puntos consecutivos por encima del umbral. No es la regla de etiquetado
    final del modelado — solo da una primera estimación, honesta y reproducible, de "cuánto tiempo
    antes del final documentado empieza a verse la degradación", y su sensibilidad a los parámetros
    (arriba) muestra cuánto varía esa estimación según qué tan estricto sea el criterio.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cierre: implicaciones para la variable objetivo, ventanas, partición y desbalance

    ### Variable objetivo y ventanas

    La evidencia de esta exploración apunta a una variable objetivo del tipo *"falla dentro de una
    ventana de H horas"*, calculada hacia atrás desde el último snapshot válido de cada rodamiento
    con falla documentada (excluyendo capturas post-apagado). El horizonte H es una decisión de
    negocio/modelado, no solo estadística: la sección de onset exploratorio de arriba da un rango
    plausible de cuándo la degradación se vuelve detectable, pero varía con los parámetros del
    detector, así que **no debe tomarse como una frontera exacta**.

    ### Partición sin fuga

    La partición debe hacerse **por rodamiento/test completo**, nunca mezclando snapshots del mismo
    rodamiento fallado entre entrenamiento y validación/prueba: los snapshots consecutivos de un
    mismo rodamiento están altamente correlacionados (ver la sección de correlación de indicadores),
    por lo que una partición aleatoria a nivel de snapshot filtraría información de la degradación
    entre conjuntos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Desbalance de clases estimado (candidatos, no decisiones)
    """)
    return


@app.cell
def _(FAILED_COMBOS, indicators, last_valid, pd):
    HORIZONS_HOURS = [24, 48, 72]
    imbalance_rows = []
    for _combo in FAILED_COMBOS:
        _test_id = _combo["test"]
        test_files = indicators[indicators["test_id"] == _test_id][
            ["file", "elapsed_hours"]
        ].drop_duplicates()
        _last_elapsed = last_valid[_test_id]["elapsed_hours"]
        valid_files = test_files[test_files["elapsed_hours"] <= _last_elapsed]
        n_total = len(valid_files)
        for horizon in HORIZONS_HOURS:
            n_positive = int((valid_files["elapsed_hours"] >= _last_elapsed - horizon).sum())
            imbalance_rows.append(
                {
                    "combo": _combo["label"],
                    "horizonte_h": horizon,
                    "n_total_snapshots": n_total,
                    "n_positivos": n_positive,
                    "proporcion_positivos": n_positive / n_total,
                }
            )
    imbalance_table = pd.DataFrame(imbalance_rows)
    imbalance_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Características prometedoras

    De la matriz de correlación entre indicadores (test 1): varios estadísticos de amplitud (RMS,
    pico, pico-a-pico, energía, desviación estándar) están fuertemente correlacionados entre sí
    (redundantes), mientras que curtosis y factor de cresta capturan una dimensión distinta
    (impulsividad) — ambas familias, más las amplitudes en las frecuencias características
    (envolvente), son las candidatas más prometedoras para la Fase III, evitando incluir todas las
    variantes redundantes de amplitud.

    ### Preguntas abiertas para el equipo

    1. ¿Qué horizonte H (24/48/72 h u otro) se adopta como variable objetivo, dado el desbalance de
    clases estimado arriba?
    2. ¿Se usa el índice secuencial de snapshot o una versión de "horas de operación" sin huecos
    como eje temporal de referencia para las ventanas del test 1?
    3. ¿Se conserva el test 3 completo (6324 archivos) para el modelado, dado que su fin documentado
    no coincide con el fin observado?
    4. El rodamiento 4 del test 1 (defecto de elemento rodante) no mostró una subida clara de
    curtosis; su comportamiento en frecuencia (ver más arriba) debe discutirse explícitamente con el
    equipo antes de asumir que es un caso positivo "típico" para el modelo.
    """)
    return


if __name__ == "__main__":
    app.run()
