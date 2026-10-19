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
    from bearing_predictive_maintenance.indicators import compute_time_domain_indicators
    from bearing_predictive_maintenance.spectral import (
        amplitude_spectrum,
        detect_onset,
        envelope_spectrum,
        fault_peak,
        flag_post_shutdown,
    )

    # Fixed color per bearing (never cycled), shared with notebook 01.
    BEARING_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
    EARLY_COLOR = "#999999"  # neutral gray: always the earlier/healthier capture

    pd.set_option("display.max_rows", 100)
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
        compute_time_domain_indicators,
        detect_onset,
        envelope_spectrum,
        fault_peak,
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
    # Observed fault lines sit a few percent off the kinematic frequencies (bearing slip,
    # shaft-speed deviation, sampling-rate ambiguity), so peaks are searched within a
    # relative window instead of a fixed width in Hz. Chosen after seeing that the IMS
    # outer-race peak fell outside a +/-5 Hz window: a data-informed parameter, not tuned
    # per bearing, that must be fixed or fitted on training folds only in Phase III.
    FAULT_REL_TOLERANCE = 0.03

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
    return FAILED_COMBOS, FAULT_REL_TOLERANCE, FREQS, FREQ_LOOKUP, indicators


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
    ## Estadísticos descriptivos por test y rodamiento

    Capturas válidas de cada rodamiento (en el test 1, ambos ejes juntos): mediana y rango
    intercuartílico de RMS y curtosis.
    """)
    return


@app.cell
def _(indicators, last_valid, pd):
    _rows = []
    for (_test_id, _bearing), _group in indicators.groupby(["test_id", "bearing"]):
        _valid = _group[_group["elapsed_hours"] <= last_valid[_test_id]["elapsed_hours"]]
        _row = {"test": _test_id, "rodamiento": int(_bearing), "capturas": _valid["file"].nunique()}
        for _metric in ("rms", "kurtosis"):
            _q = _valid[_metric].quantile([0.25, 0.5, 0.75])
            _row[f"{_metric}_p25"] = float(_q.loc[0.25])
            _row[f"{_metric}_mediana"] = float(_q.loc[0.5])
            _row[f"{_metric}_p75"] = float(_q.loc[0.75])
        _rows.append(_row)
    descriptive_stats = pd.DataFrame(_rows)
    descriptive_stats
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
    evidencia cuál es más adecuado para ventanas y modelado. El test 2 no tiene huecos y el test 3
    solo tiene 4 huecos menores (el mayor ≈7.5 h, ver cuaderno 01), así que se muestran solo con
    horas de reloj. Las curvas promedian los ejes x/y de cada rodamiento por captura.
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
    abrupta de lo que realmente es en tiempo de operación. Ninguno de los dos ejes equivale a
    tiempo de operación: las horas de reloj incluyen las pausas, y el índice de snapshot cuenta
    igual las capturas tomadas cada 5 y cada 10 minutos (ráfagas documentadas en el cuaderno 01).
    Para cuantificar *cuánto tiempo de operación* precede a la falla se recomienda construir **horas
    de operación** acumulando cada hueco entre capturas solo hasta la cadencia nominal (600 s), de
    modo que las pausas no sumen tiempo. La elección final del eje es una decisión de la Fase III.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Curtosis por rodamiento al final del test 1

    Máximo de curtosis entre los ejes x/y por captura, contado por tramos de horas antes de la
    última captura válida. La línea base es la mediana del primer 20% del test.
    """)
    return


@app.cell
def _(indicators, last_valid, pd):
    _test1 = indicators[indicators["test_id"] == "1"]
    _end = last_valid["1"]["elapsed_hours"]
    _rows = []
    for _bearing in range(1, 5):
        _per_file = _test1[_test1["bearing"] == _bearing].groupby("elapsed_hours")["kurtosis"].max()
        _before_end = _end - _per_file.index.to_numpy()
        _mid = (_before_end > 72) & (_before_end <= 150)
        _last = _before_end <= 72
        _rows.append(
            {
                "rodamiento": _bearing,
                "curtosis_base_mediana": float(_per_file[_before_end >= 0.8 * _end].median()),
                "capturas_150_a_72h": int(_mid.sum()),
                "curtosis_mayor_8_150_a_72h": int(((_per_file > 8) & _mid).sum()),
                "capturas_ultimas_72h": int(_last.sum()),
                "curtosis_mayor_8_ultimas_72h": int(((_per_file > 8) & _last).sum()),
                "curtosis_max": float(_per_file.max()),
                "horas_antes_del_final_en_max": float(_end - _per_file.idxmax()),
                "curtosis_max_150_a_72h": float(_per_file[_mid].max()),
                "horas_antes_del_final_en_max_150_a_72h": float(_end - _per_file[_mid].idxmax()),
            }
        )
    kurtosis_episodes = pd.DataFrame(_rows)
    kurtosis_episodes
    return


@app.cell
def _(compute_time_domain_indicators, indicators, last_valid, load_channel_signal, np, pd):
    # Robustness: are the kurtosis episodes carried by a few extreme samples? Recompute
    # kurtosis after dropping the N_TRIMMED largest |x - mean| samples of each channel.
    N_TRIMMED = 10
    _end = last_valid["1"]["elapsed_hours"]
    _files = (
        indicators[(indicators["test_id"] == "1") & (indicators["elapsed_hours"] <= _end)]
        .drop_duplicates("file")
        .set_index("file")["elapsed_hours"]
    )
    _files = _files[_end - _files <= 150]
    _rows = []
    for _bearing in (3, 4):
        for _file, _elapsed in _files.items():
            _kurt = []
            for _axis in ("x", "y"):
                _signal = load_channel_signal("1", _file, _bearing, _axis)
                _order = np.argsort(np.abs(_signal - _signal.mean()))
                _trimmed = _signal[_order[: len(_signal) - N_TRIMMED]].reshape(-1, 1)
                _kurt.append(compute_time_domain_indicators(_trimmed).loc[1, "kurtosis"])
            _rows.append(
                {"rodamiento": _bearing, "horas_antes": _end - _elapsed, "curtosis": max(_kurt)}
            )
    _trim = pd.DataFrame(_rows)
    _trim["tramo"] = np.where(_trim["horas_antes"] > 72, "150_a_72h", "ultimas_72h")
    trimmed_kurtosis = (
        _trim.assign(mayor_8=_trim["curtosis"] > 8)
        .groupby(["rodamiento", "tramo"])
        .agg(capturas=("curtosis", "size"), curtosis_mayor_8=("mayor_8", "sum"))
        .reset_index()
    )
    trimmed_kurtosis
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **El rodamiento 4 tiene un episodio fuerte de curtosis, pero no al final.** Entre 150 y 72 h
    antes de la última captura válida, 190 de sus 381 capturas superan una curtosis de 8 (línea base
    ≈4.3), con un máximo de 59.4 a ≈93 h del final dentro de ese tramo; en las últimas 72 h ninguna
    captura supera 8. El rodamiento 3 hace lo contrario: 146 de sus 296 capturas de las últimas
    72 h superan 8 (máximo 74.6 a ≈64 h del final). Los rodamientos 1 y 2 nunca superan 8. En la
    figura de tendencias el episodio del rodamiento 4 se ve atenuado porque se promedian los ejes.

    **Robustez (tabla siguiente).** Al quitar las 10 muestras más extremas de cada captura (de
    20480), el episodio del rodamiento 4 se mantiene en 108 de las 190 capturas, mientras que la
    subida final del rodamiento 3 cae de 146 a 6 capturas: la curtosis tardía del rodamiento 3 la
    producen unos pocos impactos aislados. Los máximos de 74.6 de ambos rodamientos son de ese tipo.
    Implicación: un criterio de etiquetado basado en curtosis se dispararía ~150 h antes del final
    para el rodamiento 4 y luego dejaría de hacerlo, y para el rodamiento 3 dependería de pocas
    muestras.
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
    ## Autocorrelación temporal de los indicadores

    Correlación de cada serie de indicadores consigo misma desplazada k capturas, para los
    rodamientos con falla documentada (solo capturas válidas). Con la cadencia habitual de 10 min,
    6 capturas ≈ 1 h, 36 ≈ 6 h y 144 ≈ 24 h; en el test 1 las pausas hacen que el desfase en horas
    no sea uniforme.
    """)
    return


@app.cell
def _(FAILED_COMBOS, indicators, last_valid, pd):
    def combo_indicators(combo: dict) -> pd.DataFrame:
        """Valid snapshots of one failed bearing/axis, in time order."""
        _test_id = combo["test"]
        _mask = (
            (indicators["test_id"] == _test_id)
            & (indicators["bearing"] == combo["bearing"])
            & (indicators["axis"] == combo["axis"])
            & (indicators["elapsed_hours"] <= last_valid[_test_id]["elapsed_hours"])
        )
        return indicators[_mask].sort_values("timestamp").reset_index(drop=True)

    AUTOCORR_LAGS = {"1": 1, "6": 6, "36": 36, "144": 144}
    _rows = []
    for _combo in FAILED_COMBOS:
        _data = combo_indicators(_combo)
        for _metric in ("rms", "kurtosis"):
            _row = {"combo": _combo["label"], "indicador": _metric}
            for _name, _lag in AUTOCORR_LAGS.items():
                _row[f"autocorr_lag_{_name}"] = float(_data[_metric].autocorr(lag=_lag))
            _rows.append(_row)
    autocorr_table = pd.DataFrame(_rows)
    autocorr_table
    return (combo_indicators,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lectura:** el RMS conserva una autocorrelación de 0.95-0.995 entre capturas consecutivas y de
    0.45-0.85 a 144 capturas (≈24 h); la curtosis, más ruidosa, de 0.52-0.69 entre capturas
    consecutivas. Capturas vecinas de un mismo rodamiento son casi copias entre sí: esta es la
    evidencia de que una partición aleatoria por captura filtraría información entre
    entrenamiento y prueba.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estabilidad de los segmentos

    **Dentro de cada captura:** cada captura de 20480 muestras se divide en 10 segmentos de 2048
    muestras (≈0.1 s) y se mide la variación de RMS y curtosis entre segmentos (coeficiente de
    variación), en las 20 primeras y las 20 últimas capturas válidas de cada rodamiento fallado.
    **En la línea base:** se compara la mediana de RMS y curtosis entre la primera y la segunda
    mitad del primer 20% de capturas válidas.
    """)
    return


@app.cell
def _(
    FAILED_COMBOS,
    combo_indicators,
    compute_time_domain_indicators,
    load_channel_signal,
    np,
    pd,
):
    N_SEGMENTS = 10
    N_STAGE_SNAPSHOTS = 20
    _rows = []
    for _combo in FAILED_COMBOS:
        _files = combo_indicators(_combo)["file"].tolist()
        for _stage, _subset in (
            ("temprano", _files[:N_STAGE_SNAPSHOTS]),
            ("tardío", _files[-N_STAGE_SNAPSHOTS:]),
        ):
            _cv = {"rms": [], "kurtosis": []}
            for _file in _subset:
                _signal = load_channel_signal(
                    _combo["test"], _file, _combo["bearing"], _combo["axis"]
                )
                _usable = len(_signal) // N_SEGMENTS * N_SEGMENTS
                _segments = _signal[:_usable].reshape(N_SEGMENTS, -1).T
                _seg = compute_time_domain_indicators(_segments)
                for _metric in _cv:
                    _cv[_metric].append(_seg[_metric].std() / _seg[_metric].mean())
            _rows.append(
                {
                    "combo": _combo["label"],
                    "etapa": _stage,
                    "cv_rms_mediana": float(np.median(_cv["rms"])),
                    "cv_curtosis_mediana": float(np.median(_cv["kurtosis"])),
                }
            )
    segment_stability = pd.DataFrame(_rows)
    segment_stability
    return


@app.cell
def _(FAILED_COMBOS, combo_indicators, pd):
    _rows = []
    for _combo in FAILED_COMBOS:
        _data = combo_indicators(_combo)
        _baseline = _data.iloc[: max(10, len(_data) // 5)]
        _half = len(_baseline) // 2
        for _metric in ("rms", "kurtosis"):
            _first = _baseline[_metric].iloc[:_half].median()
            _second = _baseline[_metric].iloc[_half:].median()
            _rows.append(
                {
                    "combo": _combo["label"],
                    "indicador": _metric,
                    "capturas_linea_base": len(_baseline),
                    "mediana_primera_mitad": float(_first),
                    "mediana_segunda_mitad": float(_second),
                    "cambio_relativo": float(_second / _first - 1),
                    "cv_linea_base": float(_baseline[_metric].std() / _baseline[_metric].mean()),
                }
            )
    baseline_stability = pd.DataFrame(_rows)
    baseline_stability
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Dentro de cada captura**, el RMS varía entre 2% y 5.5% entre segmentos de 2048 muestras
    (mediana), tanto al inicio como al final: una ventana de ≈0.1 s ya da un RMS estable. La
    curtosis varía 7-10% al inicio, pero en la etapa tardía llega a 32% (test 1, rodamiento 3) y 15%
    (test 3, rodamiento 3), porque los impactos del defecto no aparecen en todos los segmentos: en
    ventanas cortas la curtosis tardía será ruidosa.

    **En la línea base**, los tests 2 y 3 son estables (la mediana de RMS y curtosis cambia menos de
    1.2% entre la primera y la segunda mitad del primer 20% de capturas válidas). El test 1 no lo
    es: el RMS de los rodamientos 3 y 4 sube 20-23% dentro de ese mismo tramo, y la curtosis del
    rodamiento 4 sube 22%. Un umbral calibrado solo con el inicio del test 1 se dispararía antes de
    tiempo; la línea base de este test debe tratarse como un periodo de asentamiento o deriva, no
    como un estado sano estable.
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


@app.cell
def _(
    CHOSEN_BANDS,
    COMBO_SIGNALS,
    FAILED_COMBOS,
    FAULT_REL_TOLERANCE,
    FREQ_LOOKUP,
    amplitude_spectrum,
    envelope_spectrum,
    fault_peak,
    ims,
    np,
    pd,
):
    def top_peaks(freqs: np.ndarray, amps: np.ndarray, low: float, high: float, n: int) -> str:
        """Frequencies of the n largest bins in [low, high], at least 5 Hz apart."""
        _mask = (freqs >= low) & (freqs <= high)
        _chosen: list[float] = []
        for _index in np.argsort(amps[_mask])[::-1]:
            _freq = float(freqs[_mask][_index])
            if all(abs(_freq - _other) >= 5.0 for _other in _chosen):
                _chosen.append(_freq)
            if len(_chosen) == n:
                break
        return ", ".join(f"{_freq:.1f}" for _freq in _chosen)

    _rows = []
    for _combo in FAILED_COMBOS:
        _late = COMBO_SIGNALS[_combo["label"]]["late"]
        _env_freqs, _env_amps = envelope_spectrum(
            _late, ims.SAMPLING_RATE_HZ, band=CHOSEN_BANDS[_combo["label"]]
        )
        _raw_freqs, _raw_amps = amplitude_spectrum(_late, ims.SAMPLING_RATE_HZ)
        _name = _combo["target_freq_name"]
        for _multiple in (1, 2):
            _target = _multiple * FREQ_LOOKUP[_name]
            _peak_freq, _peak_amp = fault_peak(
                _env_freqs, _env_amps, target=_target, rel_tolerance=FAULT_REL_TOLERANCE
            )
            _rows.append(
                {
                    "combo": _combo["label"],
                    "frecuencia": f"{_multiple}x{_name}",
                    "teorica_hz": _target,
                    "pico_observado_hz": _peak_freq,
                    "desviacion_pct": 100 * (_peak_freq / _target - 1),
                    "amplitud_pico": _peak_amp,
                    "picos_envolvente_dominantes_hz": top_peaks(_env_freqs, _env_amps, 5, 500, 3),
                    "picos_crudos_dominantes_hz": top_peaks(_raw_freqs, _raw_amps, 5, 1000, 4),
                }
            )
    observed_fault_peaks = pd.DataFrame(_rows)
    observed_fault_peaks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lectura de la figura anterior:**

    La tabla siguiente mide cada pico (búsqueda dentro de ±3% de la frecuencia teórica) y lista los
    picos dominantes de la captura tardía.

    - **Test 2, rodamiento 1 (BPFO)** y **test 3, rodamiento 3 (BPFO)**: el espectro de envolvente
    tardío tiene un pico limpio y dominante en 231.4 Hz (test 2) y 230.5 Hz (test 3), con su
    armónico en 461.9 Hz. Está 2.1-2.5% **por debajo** del BPFO calculado (236.4 Hz), no exactamente
    en él. En el espectro crudo, en cambio, dominan otras frecuencias: 693 y 925 Hz en el test 2, y
    contenido de baja frecuencia (13.7 y 57.6 Hz) seguido de 924 y 692 Hz en el test 3. El defecto
    se revela con claridad solo al demodular, el comportamiento esperado de un defecto que modula
    una resonancia.
    - **Test 1, rodamiento 3 (BPFI)**: hay un pico a 293.9 Hz (1.0% bajo BPFI), pero no domina: los
    mayores picos de la envolvente están en 131.8 y 260.7 Hz.
    - **Test 1, rodamiento 4 (BSF, elemento rodante)**: el pico dominante de la envolvente está en
    65.4 Hz, cercano a 2× la frecuencia nominal del eje (66.7 Hz), no en BSF (139.9 Hz) ni en 2×BSF.
    Dentro de la ventana de BSF el pico más alto (137.7 Hz) es pequeño. La sección de evolución
    temporal muestra, aun así, una tendencia ascendente en esa frecuencia.

    **Desviación de las frecuencias características.** Todas las líneas observadas caen 1-3% por
    debajo de las frecuencias cinemáticas. El deslizamiento de la jaula no basta para explicarlo:
    baja BPFO pero sube BPFI, porque BPFO + BPFI = 16 × frecuencia del eje. Con 20 kHz, las líneas
    de pista externa (−2.1% y −2.5%) son compatibles con deslizamiento, pero la de pista interna
    (−1.0%) requiere además un eje ≈1% más lento que los 2000 RPM nominales (con 1980 RPM el BPFI
    teórico es 293.96 Hz, frente a 293.9 Hz observado). Con 20.48 kHz, que es la otra lectura
    posible del `readme` (20480 muestras descritas como 1 s a 20 kHz), la línea de pista interna
    (+1.4%) sí sería compatible con deslizamiento, pero las de pista externa quedarían sin él
    (+0.3%). Las líneas del eje son demasiado débiles para medir su velocidad, así que los datos no
    resuelven la ambigüedad: se conserva el valor documentado, 20 kHz. Los picos se buscan en una
    ventana relativa de ±3%; esa tolerancia se eligió después de observar que el pico de pista
    externa quedaba fuera de la ventana fija de ±5 Hz, así que es un parámetro informado por los
    datos.
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
    FAULT_REL_TOLERANCE,
    FREQ_LOOKUP,
    RAW_ROOT,
    envelope_spectrum,
    fault_peak,
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
            _peak_freq, amp_at_target = fault_peak(
                freqs_, amps_, target=target, rel_tolerance=FAULT_REL_TOLERANCE
            )
            rows.append(
                {
                    "combo": _combo["label"],
                    "test": _test_id,
                    "file": path.name,
                    "elapsed_hours": elapsed_lookup[_test_id][path.name],
                    "envelope_amp": amp_at_target,
                    "peak_freq_hz": _peak_freq,
                    "at_window_edge": abs(_peak_freq - target)
                    >= FAULT_REL_TOLERANCE * target - (freqs_[1] - freqs_[0]),
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


@app.cell
def _(FAILED_COMBOS, evolution, last_valid, pd):
    _rows = []
    for _combo in FAILED_COMBOS:
        _data = evolution[evolution["combo"] == _combo["label"]].sort_values("elapsed_hours")
        _end = last_valid[_combo["test"]]["elapsed_hours"]
        _baseline = _data["envelope_amp"].iloc[: max(5, len(_data) // 5)]
        _late = _data.loc[_data["elapsed_hours"] >= _end - 100, "envelope_amp"]
        _rows.append(
            {
                "combo": _combo["label"],
                "linea_base_media": float(_baseline.mean()),
                "ultimas_100h_mediana": float(_late.median()),
                "ultimas_100h_p90": float(_late.quantile(0.9)),
                "razon_mediana": float(_late.median() / _baseline.mean()),
                "razon_p90": float(_late.quantile(0.9) / _baseline.mean()),
                # Share of points whose maximum sits on the search-window edge bin: there the
                # value is the window boundary, not a spectral line.
                "fraccion_pico_en_borde": float(_data["at_window_edge"].mean()),
            }
        )
    envelope_growth = pd.DataFrame(_rows)
    envelope_growth
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Crecimiento de la amplitud de envolvente (tabla anterior).** Se compara la media del primer
    20% de la serie con la mediana y el percentil 90 de las últimas 100 h válidas. Solo los
    rodamientos de pista externa muestran un crecimiento fuerte y concentrado al final (test 2:
    5.7× en mediana y 19× en p90; test 3: 2.6× en mediana pero 51× en p90, porque la subida ocurre
    en las últimas horas). Los dos rodamientos del test 1 crecen poco: el de pista interna 1.4× en
    mediana (2.4× en p90) y el de elemento rodante 1.9× (2.9× en p90). La pista interna no es, por
    tanto, un caso de huella clara en la envolvente. Junto con el episodio de curtosis del
    rodamiento 4 entre 150 y 72 h antes del final, esto indica que el defecto de elemento rodante
    sí deja huella en ambos dominios, aunque débil y no concentrada al final.

    **Picos en el borde de la ventana.** En el 7-19% de los puntos de cada serie el máximo cae en el
    bin del borde de la ventana de ±3% (`fraccion_pico_en_borde`): en esos puntos la "amplitud en la
    frecuencia de falla" no es una línea espectral sino el borde de la búsqueda, lo que añade ruido
    sobre todo a los rodamientos del test 1 y al test 3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inicio exploratorio de degradación (onset), por rodamiento fallado
    """)
    return


@app.cell
def _(FAILED_COMBOS, detect_onset, evolution, last_valid, pd):
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
        _last_elapsed = last_valid[_combo["test"]]["elapsed_hours"]
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
    antes de la última captura válida empieza a verse la degradación", y su sensibilidad a los
    parámetros (arriba) muestra cuánto varía esa estimación según qué tan estricto sea el criterio.
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

    La partición debe hacerse **por test completo**, nunca mezclando capturas del mismo rodamiento
    entre entrenamiento y validación/prueba: la autocorrelación temporal de los indicadores es muy
    alta (sección "Autocorrelación temporal"), así que una partición aleatoria por captura
    filtraría información de la degradación entre conjuntos. Los cuatro rodamientos de un test
    comparten eje, carga y fecha, por lo que la opción más conservadora es dejar un test fuera en
    cada partición (*leave-one-test-out*). Esa validación tiene dos límites: solo hay 4 rodamientos
    con falla en 3 corridas independientes, y los modos de falla no se repiten entre tests (el test
    1 tiene pista interna y elemento rodante; los tests 2 y 3, pista externa).

    Además, varios parámetros de esta exploración se eligieron mirando la serie completa: las
    bandas de envolvente se derivan de la captura de fin de vida, y los umbrales de onset y de
    apagado usan la propia serie. Si se reutilizan en la Fase III son **fuga de la etiqueta**: deben
    fijarse a priori o ajustarse solo con los datos de entrenamiento de cada partición.
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
    _NOMINAL_CADENCE_S = 600.0
    imbalance_rows = []
    for _combo in FAILED_COMBOS:
        _test_id = _combo["test"]
        test_files = (
            indicators[indicators["test_id"] == _test_id][["file", "timestamp", "elapsed_hours"]]
            .drop_duplicates()
            .sort_values("timestamp")
        )
        _last_elapsed = last_valid[_test_id]["elapsed_hours"]
        valid_files = test_files[test_files["elapsed_hours"] <= _last_elapsed]
        n_total = len(valid_files)
        # Operating hours: each gap counts at most one nominal cadence step.
        _gaps = valid_files["timestamp"].diff().dt.total_seconds().fillna(0.0)
        _operating = _gaps.clip(upper=_NOMINAL_CADENCE_S).cumsum() / 3600
        for horizon in HORIZONS_HOURS:
            n_positive = int((valid_files["elapsed_hours"] >= _last_elapsed - horizon).sum())
            _n_operating = int((_operating >= _operating.iloc[-1] - horizon).sum())
            imbalance_rows.append(
                {
                    "combo": _combo["label"],
                    "horizonte_h": horizon,
                    "n_total_snapshots": n_total,
                    "n_positivos": n_positive,
                    "proporcion_positivos": n_positive / n_total,
                    "n_positivos_horas_operacion": _n_operating,
                }
            )
    imbalance_table = pd.DataFrame(imbalance_rows)
    imbalance_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Características prometedoras

    De la matriz de correlación entre indicadores (test 1, todos los rodamientos y capturas): RMS,
    desviación estándar y energía son redundantes entre sí (|r| 0.90-1.00). Pico y pico-a-pico
    (|r| 0.99 entre sí) se parecen más a la familia de impulsividad (|r| 0.76-0.89 con curtosis,
    factor de cresta y factor de impulso) que al RMS (|r| 0.69-0.74). Curtosis, factor de cresta y
    factor de impulso forman un grupo (|r| 0.81-1.00); el factor de forma queda en un punto
    intermedio (|r| 0.57-0.64 con la familia de impulsividad y 0.33-0.36 con RMS y energía) y la
    asimetría no se correlaciona con ningún otro
    (|r| ≤ 0.16). Estas correlaciones mezclan rodamientos y momentos con una tendencia compartida,
    lo que las infla; no miden la redundancia entre rodamientos del mismo eje. Para la Fase III son
    prometedoras una variable de energía (RMS), una de impulsividad (curtosis o factor de cresta,
    preferiblemente con una versión robusta a muestras aisladas) y las amplitudes de envolvente en
    las frecuencias características, que solo mostraron una huella clara en pista externa, sin
    incluir todas las variantes
    redundantes.

    ### Preguntas abiertas para el equipo

    1. ¿Qué horizonte H (24/48/72 h u otro) se adopta como variable objetivo, dado el desbalance de
    clases estimado arriba?
    2. ¿Se usa el índice secuencial de snapshot o una versión de "horas de operación" sin huecos
    como eje temporal de referencia para las ventanas del test 1?
    3. ¿Se conserva el test 3 completo (6324 archivos) para el modelado, dado que su fin documentado
    no coincide con el fin observado?
    4. El rodamiento 4 del test 1 (defecto de elemento rodante) tiene un episodio de curtosis entre
    150 y 72 h antes del final que luego desaparece, y una huella débil en BSF: debe decidirse
    explícitamente cómo tratarlo en el etiquetado antes de asumir que es un caso positivo "típico".
    5. Los rodamientos sin falla documentada se etiquetarían como negativos en todo el test, pero
    el `readme` indica que todas las fallas ocurrieron después de superar la vida de diseño, así que
    su estado al final del test es desconocido. ¿Se aceptan como negativos o se excluyen sus
    últimas capturas?
    """)
    return


if __name__ == "__main__":
    app.run()
