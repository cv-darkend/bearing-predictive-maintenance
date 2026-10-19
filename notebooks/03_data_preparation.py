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
    # Preparación de datos — NASA IMS

    Cuaderno 03: Fase III (Data Preparation, CRISP-DM). El documento del proyecto define su
    alcance así: "Fase III: A su vez se someten los datos a procesos de limpieza, normalización y
    control de calidad, con el fin de segmentar las señales en ventanas homogéneas; a través de las
    siguientes actividades: Limpieza y procesamiento [...] Normalización y estandarización [...]
    Segmentación de las señales".

    El cuaderno documenta los criterios de limpieza y sus conteos, los dos ejes temporales de cada
    captura, el diseño de segmentación, y la normalización sin fuga de información entre tests. Se
    apoya en `data/processed/ims_snapshot_manifest.csv` (generado por `make manifest` a partir de
    las tablas de `data/interim/`), en `data/interim/ims_snapshot_indicators.csv`, en capturas
    crudas puntuales (`data/raw/`, solo lectura) y en los módulos `manifest.py`, `segmentation.py` y
    `folds.py` de `src/bearing_predictive_maintenance/`.

    Dos decisiones del equipo enmarcan esta fase:

    - **Segmentación "Mixto"** ("De acuerdo con el Mixto."): la unidad de análisis es la captura
      completa (20480 muestras), que es también la ventana para las características espectrales;
      las características de tiempo se calculan sobre sus 10 segmentos de 2048 muestras y se
      resumen de nuevo a una fila por captura.
    - **Horizonte de predicción H como análisis de sensibilidad** ("La idea creeria que es un
      analisis de sensibilidades, por ejemplo 12, 24, 36, 48, 72 pero esto solo lo doy como
      ejemplo."). La variable objetivo pertenece a la Fase V: aquí no se crea ninguna etiqueta. La
      Fase III guarda las horas hasta el final de cada test, de modo que cualquier H se deriva
      después sin rehacer la preparación.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from bearing_predictive_maintenance import ims
    from bearing_predictive_maintenance.folds import leave_one_test_out, standardize_by_fold
    from bearing_predictive_maintenance.inventory import SATURATION_LEVEL
    from bearing_predictive_maintenance.manifest import (
        MANIFEST_COLUMNS,
        NOMINAL_CADENCE_S,
        POST_SHUTDOWN_BASELINE_N,
        POST_SHUTDOWN_FRACTION_OF_BASELINE_MEDIAN,
    )
    from bearing_predictive_maintenance.segmentation import (
        N_SEGMENTS,
        SEGMENT_LENGTH,
        SUMMARY_STATISTICS,
        remove_dc,
        segment_indicators,
        summarize_segments,
    )

    # Fixed color per bearing (never cycled), shared with notebooks 01 and 02.
    BEARING_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
    # Spanish display names for the documented failure modes of ims.TEST_REGISTRY.
    FAILURE_MODE_ES = {
        "inner race defect": "pista interna",
        "roller element defect": "elemento rodante",
        "outer race failure": "pista externa",
    }

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
    PROCESSED_DIR = REPO_ROOT / "data" / "processed"
    FIGURES_DIR = REPO_ROOT / "reports" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return (
        BEARING_COLORS,
        FAILURE_MODE_ES,
        FIGURES_DIR,
        INTERIM_DIR,
        MANIFEST_COLUMNS,
        NOMINAL_CADENCE_S,
        N_SEGMENTS,
        POST_SHUTDOWN_BASELINE_N,
        POST_SHUTDOWN_FRACTION_OF_BASELINE_MEDIAN,
        PROCESSED_DIR,
        RAW_ROOT,
        SATURATION_LEVEL,
        SEGMENT_LENGTH,
        SUMMARY_STATISTICS,
        ims,
        leave_one_test_out,
        np,
        pd,
        plt,
        remove_dc,
        segment_indicators,
        standardize_by_fold,
        summarize_segments,
    )


@app.cell
def _(INTERIM_DIR, PROCESSED_DIR, pd):
    manifest = pd.read_csv(
        PROCESSED_DIR / "ims_snapshot_manifest.csv",
        dtype={"test_id": str, "file": str},
        keep_default_na=False,
        parse_dates=["timestamp"],
    )
    indicators = pd.read_csv(
        INTERIM_DIR / "ims_snapshot_indicators.csv",
        dtype={"test_id": str, "file": str},
        parse_dates=["timestamp"],
    )
    # One row per (test, file): the cleaning flags and time axes are per file.
    manifest_files = manifest.drop_duplicates(["test_id", "file"]).reset_index(drop=True)
    valid_rows = manifest[manifest["valid"]]
    manifest.head()
    return indicators, manifest, manifest_files, valid_rows


@app.cell(hide_code=True)
def _(
    POST_SHUTDOWN_BASELINE_N,
    POST_SHUTDOWN_FRACTION_OF_BASELINE_MEDIAN,
    SATURATION_LEVEL,
    mo,
):
    mo.md(f"""
    ## Limpieza y control de calidad

    El manifiesto tiene una fila por (test, archivo, canal) y se construye solo a partir de las
    tablas intermedias (no se releen los archivos crudos). Criterios aplicados a cada archivo:

    1. **Captura posterior al apagado — excluida.** Regla de la Fase II (cuaderno 01): un archivo
       es posterior al apagado cuando el RMS máximo entre sus canales es menor que
       {POST_SHUTDOWN_FRACTION_OF_BASELINE_MEDIAN} veces la mediana de los primeros
       {POST_SHUTDOWN_BASELINE_N} archivos de su test (`spectral.flag_post_shutdown`).
    2. **Error de lectura — excluido.** El inventario registra un error al leer el archivo.
    3. **Archivo con saturación — se conserva, marcado.** Alguna muestra alcanza
       ±{SATURATION_LEVEL} (`inventory.SATURATION_LEVEL`). Son las capturas finales, donde la
       degradación es mayor; excluirlas eliminaría justamente la evidencia de falla.
    4. **Captura posterior al fin documentado — se conserva, marcada.** El test 3 continúa más allá
       de la fecha de fin del `readme` de NASA (discrepancia de la Fase II); se mantiene y la
       decisión final queda para la Fase V.

    Una captura es **válida** cuando no es posterior al apagado ni tiene error de lectura.
    """)
    return


@app.cell
def _(manifest, manifest_files, pd):
    _rows = []
    for _test_id, _files in manifest_files.groupby("test_id"):
        _channel_rows = manifest[manifest["test_id"] == _test_id]
        _rows.append(
            {
                "test": _test_id,
                "archivos": len(_files),
                "archivos_validos": int(_files["valid"].sum()),
                "posteriores_apagado_excluidos": int(_files["post_shutdown"].sum()),
                "error_lectura_excluidos": int(_files["read_error"].sum()),
                "saturados_conservados": int(_files["saturated_file"].sum()),
                "tras_fin_documentado_conservados": int(_files["after_documented_end"].sum()),
                "filas_manifiesto": len(_channel_rows),
                "filas_validas": int(_channel_rows["valid"].sum()),
            }
        )
    cleaning_summary = pd.DataFrame(_rows).set_index("test")
    cleaning_summary.loc["total"] = cleaning_summary.sum()
    cleaning_summary
    return (cleaning_summary,)


@app.cell(hide_code=True)
def _(cleaning_summary, manifest_files, mo):
    _total = cleaning_summary.loc["total"]
    _saturated = manifest_files[manifest_files["saturated_file"]]
    _saturated_hours = _saturated["hours_to_end_clock"].max()
    mo.md(f"""
    El manifiesto tiene **{_total["filas_manifiesto"]} filas** (canal × archivo), de las cuales
    **{_total["filas_validas"]} son válidas**. Se excluyen {_total["posteriores_apagado_excluidos"]}
    archivos posteriores al apagado y {_total["error_lectura_excluidos"]} con error de lectura; se
    conservan marcados {_total["saturados_conservados"]} archivos con saturación y
    {_total["tras_fin_documentado_conservados"]} archivos posteriores al fin documentado (todos del
    test 3). De los archivos con saturación, {int(_saturated["valid"].sum())} de
    {len(_saturated)} son válidos, y todos están a lo sumo
    {_saturated_hours:.2f} h de reloj antes de la última captura válida de su test: son capturas
    del final de la vida útil.
    """)
    return


@app.cell(hide_code=True)
def _(NOMINAL_CADENCE_S, mo):
    mo.md(f"""
    ## Ejes temporales

    Cada captura tiene dos ejes:

    - **Horas de reloj** (`clock_hours`): tiempo transcurrido desde la primera captura del test.
    - **Horas de operación** (`operating_hours`): cada hueco entre capturas consecutivas cuenta como
      máximo un paso de la cadencia nominal ({NOMINAL_CADENCE_S:.0f} s), de modo que las pausas del
      banco no suman tiempo de funcionamiento. Es una única definición compartida
      (`manifest.operating_hours`).

    Sobre ambos ejes se guardan las horas hasta la última captura válida del test
    (`hours_to_end_clock`, `hours_to_end_operating`), sin ninguna columna de etiqueta.
    """)
    return


@app.cell
def _(manifest_files, pd):
    _rows = []
    for _test_id, _files in manifest_files.groupby("test_id"):
        _valid = _files[_files["valid"]]
        _clock = float(_valid["clock_hours"].max())
        _operating = float(_valid["operating_hours"].max())
        _rows.append(
            {
                "test": _test_id,
                "horas_reloj_ultima_valida": _clock,
                "horas_operacion_ultima_valida": _operating,
                "horas_pausa": _clock - _operating,
                "fraccion_pausa": (_clock - _operating) / _clock,
                "horas_reloj_todos_los_archivos": float(_files["clock_hours"].max()),
                "horas_operacion_todos_los_archivos": float(_files["operating_hours"].max()),
            }
        )
    time_axes_summary = pd.DataFrame(_rows).set_index("test").round(2)
    time_axes_summary
    return (time_axes_summary,)


@app.cell
def _(FIGURES_DIR, manifest_files, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4))
    for _ax, (_test_id, _files) in zip(_axes, manifest_files.groupby("test_id"), strict=True):
        _valid = _files[_files["valid"]]
        _ax.plot(_valid["clock_hours"], _valid["operating_hours"], color="#2a78d6")
        _limit = float(_valid["clock_hours"].max())
        _ax.plot([0, _limit], [0, _limit], color="#999999", linestyle="--", label="sin pausas")
        _ax.set_title(f"Test {_test_id}")
        _ax.set_xlabel("Horas de reloj")
        _ax.set_ylabel("Horas de operación")
        _ax.legend(loc="upper left")
    _fig.suptitle("Horas de operación frente a horas de reloj (capturas válidas)")
    _fig.tight_layout()
    _fig.savefig(FIGURES_DIR / "03_time_axes.png", dpi=150)
    _fig
    return


@app.cell(hide_code=True)
def _(mo, time_axes_summary):
    _t1 = time_axes_summary.loc["1"]
    _t2 = time_axes_summary.loc["2"]
    _t3 = time_axes_summary.loc["3"]
    mo.md(f"""
    El test 1 concentra las pausas: de {_t1["horas_reloj_ultima_valida"]} h de reloj hasta su
    última captura válida, solo {_t1["horas_operacion_ultima_valida"]} h son de operación
    ({_t1["fraccion_pausa"]:.0%} del tiempo de reloj son pausas). El test 2 tiene
    {_t2["horas_pausa"]} h de pausa y el test 3 {_t3["horas_pausa"]} h
    ({_t3["fraccion_pausa"]:.0%}). La elección del eje para definir H es una decisión abierta de la
    Fase V.

    ### Ilustración: capturas dentro de H en la grilla de sensibilidad

    La tabla siguiente **no es una etiqueta**: ilustra, con la grilla que el equipo dio como
    ejemplo (12, 24, 36, 48, 72 h), cuántas capturas válidas de cada rodamiento con falla
    documentada quedan a H horas o menos de la última captura válida, en cada eje. Sirve para
    dimensionar el análisis de sensibilidad de la Fase V.
    """)
    return


@app.cell
def _(FAILURE_MODE_ES, manifest, pd):
    EXAMPLE_HORIZONS_H = (12, 24, 36, 48, 72)
    # In test 1 each bearing has two channels (x/y); count files, not channel rows.
    _failed_files = manifest[manifest["valid"] & manifest["documented_failure"]].drop_duplicates(
        ["test_id", "bearing", "file"]
    )
    _rows = []
    for (_test_id, _bearing), _files in _failed_files.groupby(["test_id", "bearing"]):
        _row = {
            "test": _test_id,
            "rodamiento": int(_bearing),
            "modo_falla": FAILURE_MODE_ES[_files["failure_mode"].iloc[0]],
            "capturas_validas": len(_files),
        }
        for _horizon in EXAMPLE_HORIZONS_H:
            _row[f"reloj_H{_horizon}"] = int((_files["hours_to_end_clock"] <= _horizon).sum())
            _row[f"operacion_H{_horizon}"] = int(
                (_files["hours_to_end_operating"] <= _horizon).sum()
            )
        _rows.append(_row)
    horizon_illustration = pd.DataFrame(_rows)
    horizon_illustration
    return EXAMPLE_HORIZONS_H, horizon_illustration


@app.cell(hide_code=True)
def _(EXAMPLE_HORIZONS_H, horizon_illustration, mo):
    _differences = []
    for _row in horizon_illustration.drop_duplicates("test").itertuples():
        _changed = [
            f"H = {_h} h: {getattr(_row, f'reloj_H{_h}')} frente a "
            f"{getattr(_row, f'operacion_H{_h}')}"
            for _h in EXAMPLE_HORIZONS_H
            if getattr(_row, f"reloj_H{_h}") != getattr(_row, f"operacion_H{_h}")
        ]
        _differences.append(
            f"- Test {_row.test}: " + ("; ".join(_changed) if _changed else "sin diferencias")
        )
    _lines = "\n".join(_differences)
    mo.md(f"""
    El eje cambia los conteos donde hay pausas dentro de las últimas H horas (capturas en el eje
    de reloj frente al eje de operación, por rodamiento fallado):

    {_lines}

    Como el manifiesto guarda ambas distancias al final, la Fase V puede recorrer cualquier grilla
    de H en cualquiera de los dos ejes.
    """)
    return


@app.cell
def _(N_SEGMENTS, SEGMENT_LENGTH, ims):
    # Spectral resolution of the two windows of the "Mixto" design, against the
    # +/-3% relative search window used for fault lines in Phase II (notebook 02).
    FAULT_REL_TOLERANCE = 0.03
    _fs = ims.SAMPLING_RATE_HZ
    _bpfo = ims.characteristic_fault_frequencies().bpfo
    resolution = {
        "fs_hz": _fs,
        "snapshot_samples": ims.EXPECTED_ROWS_PER_SNAPSHOT,
        "segment_samples": SEGMENT_LENGTH,
        "n_segments": N_SEGMENTS,
        "df_snapshot_hz": _fs / ims.EXPECTED_ROWS_PER_SNAPSHOT,
        "df_segment_hz": _fs / SEGMENT_LENGTH,
        "bpfo_hz": _bpfo,
        "window_width_hz": 2 * FAULT_REL_TOLERANCE * _bpfo,
    }
    resolution["bins_in_window_snapshot"] = (
        resolution["window_width_hz"] / resolution["df_snapshot_hz"]
    )
    resolution["bins_in_window_segment"] = (
        resolution["window_width_hz"] / resolution["df_segment_hz"]
    )
    resolution
    return FAULT_REL_TOLERANCE, resolution


@app.cell(hide_code=True)
def _(FAULT_REL_TOLERANCE, mo, resolution):
    _r = resolution
    mo.md(f"""
    ## Segmentación de las señales (diseño "Mixto")

    **Limpieza a nivel de señal:** a cada canal se le resta su media (nivel DC,
    `segmentation.remove_dc`).

    **Por qué la captura completa es la ventana espectral.** Con fs = {_r["fs_hz"]:.0f} Hz, la
    captura completa de {_r["snapshot_samples"]} muestras da una resolución de
    {_r["df_snapshot_hz"]:.3f} Hz, mientras que un segmento de {_r["segment_samples"]} muestras
    da {_r["df_segment_hz"]:.2f} Hz. La ventana de búsqueda de ±{FAULT_REL_TOLERANCE:.0%}
    alrededor de BPFO ({_r["bpfo_hz"]:.2f} Hz) mide {_r["window_width_hz"]:.2f} Hz: contiene
    {_r["bins_in_window_snapshot"]:.1f} líneas espectrales con la captura completa y solo
    {_r["bins_in_window_segment"]:.2f} con un segmento, insuficiente para ubicar el pico de falla.

    **Por qué los segmentos para las características de tiempo.** Los {_r["n_segments"]} segmentos
    contiguos y sin solapamiento de {_r["segment_samples"]} muestras permiten resumir cada
    indicador con estadísticos robustos (mediana, percentil 90, máximo). La mediana es robusta a
    unas pocas muestras extremas (la Fase II mostró que la curtosis final del test 1 depende de
    ellas), y el percentil 90 y el máximo conservan la actividad impulsiva concentrada en pocos
    segmentos.

    Demostración: última captura válida de cada test, canal del rodamiento con falla documentada
    (en el test 1, eje x).
    """)
    return


@app.cell
def _(FAILURE_MODE_ES, RAW_ROOT, ims, manifest, pd, remove_dc, segment_indicators):
    _last_rows = (
        manifest[manifest["valid"] & manifest["documented_failure"]]
        .loc[lambda frame: frame["axis"].isin(["x", "single"])]
        .sort_values("timestamp")
        .groupby(["test_id", "bearing"])
        .tail(1)
        .sort_values(["test_id", "bearing"])
    )
    _tables = []
    demo_snapshots = []
    for _row in _last_rows.itertuples():
        _spec = ims.TEST_REGISTRY[_row.test_id]
        _array = ims.load_snapshot(RAW_ROOT / _spec.raw_relative_path / _row.file)
        _signal = remove_dc(_array[:, int(_row.channel) - 1])
        _mode = FAILURE_MODE_ES[_row.failure_mode]
        _label = f"Test {_row.test_id} - Rodamiento {int(_row.bearing)} ({_mode})"
        demo_snapshots.append(
            {
                "test_id": _row.test_id,
                "file": _row.file,
                "channel": int(_row.channel),
                "bearing": int(_row.bearing),
                "label": _label,
            }
        )
        _table = segment_indicators(_signal)[["rms", "kurtosis", "crest_factor"]]
        _table.insert(0, "caso", _label)
        _tables.append(_table.reset_index())
    segment_table = pd.concat(_tables, ignore_index=True)
    segment_table
    return demo_snapshots, segment_table


@app.cell
def _(SUMMARY_STATISTICS, demo_snapshots, indicators, pd, segment_table, summarize_segments):
    _rows = []
    for _snapshot in demo_snapshots:
        _table = segment_table[segment_table["caso"] == _snapshot["label"]].drop(
            columns=["caso", "segment"]
        )
        _full = indicators[
            (indicators["test_id"] == _snapshot["test_id"])
            & (indicators["file"] == _snapshot["file"])
            & (indicators["channel"] == _snapshot["channel"])
        ].iloc[0]
        _row = {"caso": _snapshot["label"], "archivo": _snapshot["file"]}
        for _statistic in SUMMARY_STATISTICS:
            _row.update(summarize_segments(_table, _statistic).to_dict())
        _row["rms_captura_completa"] = float(_full["rms"])
        _row["kurtosis_captura_completa"] = float(_full["kurtosis"])
        _rows.append(_row)
    segment_summary = pd.DataFrame(_rows)
    segment_summary[
        [
            "caso",
            "archivo",
            "rms_median",
            "rms_max",
            "rms_captura_completa",
            "kurtosis_median",
            "kurtosis_p90",
            "kurtosis_max",
            "kurtosis_captura_completa",
        ]
    ].round(3)
    return (segment_summary,)


@app.cell
def _(BEARING_COLORS, FIGURES_DIR, demo_snapshots, plt, segment_table):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    for _snapshot in demo_snapshots:
        _table = segment_table[segment_table["caso"] == _snapshot["label"]]
        _color = BEARING_COLORS[_snapshot["bearing"]]
        _style = "-" if _snapshot["test_id"] != "1" else "--"
        for _ax, _metric in zip(_axes, ("rms", "kurtosis"), strict=True):
            _ax.plot(
                _table["segment"],
                _table[_metric],
                marker="o",
                color=_color,
                linestyle=_style,
                label=_snapshot["label"],
            )
    _axes[0].set_title("RMS por segmento")
    _axes[1].set_title("Curtosis por segmento")
    for _ax in _axes:
        _ax.set_xlabel("Segmento (2048 muestras)")
    _axes[0].set_ylabel("RMS (g)")
    _axes[1].set_ylabel("Curtosis")
    _axes[1].legend(fontsize=8)
    _fig.suptitle("Indicadores por segmento en la última captura válida de cada rodamiento fallado")
    _fig.tight_layout()
    _fig.savefig(FIGURES_DIR / "03_segment_indicators.png", dpi=150)
    _fig
    return


@app.cell(hide_code=True)
def _(mo, segment_summary):
    _rms_ratio = segment_summary["rms_max"] / segment_summary["rms_median"]
    _ratio = segment_summary["kurtosis_max"] / segment_summary["kurtosis_median"]
    _widest = segment_summary.loc[_ratio.idxmax()]
    mo.md(f"""
    En los cuatro casos el RMS varía poco entre segmentos (el máximo por segmento es entre
    {_rms_ratio.min():.2f} y {_rms_ratio.max():.2f} veces la mediana), mientras que la curtosis
    varía más: el caso con mayor dispersión es {_widest["caso"]}, donde la curtosis máxima por
    segmento es {_ratio.max():.2f} veces la mediana ({_widest["kurtosis_max"]:.2f} frente a
    {_widest["kurtosis_median"]:.2f}; la curtosis de la captura completa es
    {_widest["kurtosis_captura_completa"]:.2f}). Por eso se guardan los tres resúmenes y la
    selección se deja a la Fase IV.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Normalización y estandarización sin fuga

    Las capturas de un mismo test están fuertemente autocorrelacionadas: una partición aleatoria
    por filas pondría capturas casi idénticas en entrenamiento y prueba. Los tres tests son las
    unidades independientes, así que la validación usa **leave-one-test-out**
    (`folds.leave_one_test_out`): cada partición deja un test completo fuera y entrena con los
    otros dos.

    **Regla:** la estandarización de características (media y desviación estándar poblacional)
    se ajusta **solo con las filas de los tests de entrenamiento** de cada partición
    (`folds.standardize_by_fold`) y luego se aplica a todas las filas. El test que queda fuera
    nunca influye en los parámetros. Lo mismo aplica a cualquier parámetro aprendido de los datos
    (por ejemplo, umbrales o la tolerancia de búsqueda espectral) en las fases siguientes.
    """)
    return


@app.cell
def _(FAILURE_MODE_ES, leave_one_test_out, pd, valid_rows):
    def failure_modes(frame: pd.DataFrame) -> str:
        modes = sorted({FAILURE_MODE_ES[mode] for mode in frame["failure_mode"] if mode != ""})
        return ", ".join(modes)

    validation_folds = leave_one_test_out(valid_rows)
    _rows = []
    for _fold in validation_folds:
        _train = valid_rows.loc[_fold.train_index]
        _test = valid_rows.loc[_fold.test_index]
        _rows.append(
            {
                "test_fuera": _fold.held_out,
                "tests_entrenamiento": ", ".join(sorted(_train["test_id"].unique())),
                "filas_entrenamiento": len(_train),
                "filas_prueba": len(_test),
                "modos_falla_entrenamiento": failure_modes(_train),
                "modos_falla_prueba": failure_modes(_test),
            }
        )
    fold_summary = pd.DataFrame(_rows)
    fold_summary
    return fold_summary, validation_folds


@app.cell
def _(indicators, pd, standardize_by_fold, valid_rows, validation_folds):
    FEATURES_DEMO = ["rms", "kurtosis"]
    _features = valid_rows[["test_id", "file", "channel"]].merge(
        indicators[["test_id", "file", "channel", *FEATURES_DEMO]],
        on=["test_id", "file", "channel"],
        how="left",
        validate="one_to_one",
    )
    _features.index = valid_rows.index
    _rows = []
    for _fold in validation_folds:
        _train_mask = valid_rows.index.isin(_fold.train_index)
        _scaled = standardize_by_fold(_features[FEATURES_DEMO], _train_mask)
        for _feature in FEATURES_DEMO:
            _train = _scaled.loc[_train_mask, _feature]
            _test = _scaled.loc[~_train_mask, _feature]
            _rows.append(
                {
                    "test_fuera": _fold.held_out,
                    "caracteristica": _feature,
                    "media_entrenamiento": float(_train.mean()),
                    "std_entrenamiento": float(_train.std(ddof=0)),
                    "media_prueba": float(_test.mean()),
                    "std_prueba": float(_test.std(ddof=0)),
                }
            )
    standardization_demo = pd.DataFrame(_rows)
    standardization_demo.round(3)
    return FEATURES_DEMO, standardization_demo


@app.cell(hide_code=True)
def _(FEATURES_DEMO, fold_summary, mo, np, standardization_demo):
    _train_mean = float(np.abs(standardization_demo["media_entrenamiento"]).max())
    _train_std_dev = float(np.abs(standardization_demo["std_entrenamiento"] - 1).max())
    _test_mean = standardization_demo["media_prueba"].abs()
    _worst = standardization_demo.loc[_test_mean.idxmax()]
    _unseen = []
    for _fold in fold_summary.itertuples():
        _train_modes = set(_fold.modos_falla_entrenamiento.split(", "))
        _missing = [m for m in _fold.modos_falla_prueba.split(", ") if m not in _train_modes]
        if _missing:
            _unseen.append(f"test {_fold.test_fuera} fuera: {', '.join(_missing)}")
    mo.md(f"""
    Con {", ".join(FEATURES_DEMO)} de la tabla de indicadores, en cada partición las filas de
    entrenamiento quedan con media ≈ 0 (|media| máxima {_train_mean:.1e}) y desviación estándar
    ≈ 1 (desvío máximo respecto de 1: {_train_std_dev:.1e}), mientras que el test fuera no: el caso
    más alejado es `{_worst["caracteristica"]}` con el test {_worst["test_fuera"]} fuera, con media
    {_worst["media_prueba"]:.2f} y desviación estándar {_worst["std_prueba"]:.2f}. Esa diferencia
    es la que el modelo debe enfrentar con un test nunca visto; estandarizar con todos los datos la
    ocultaría.

    La tabla de particiones muestra además que los modos de falla no se reparten igual entre tests.
    Particiones cuyo test fuera tiene modos de falla ausentes del entrenamiento:
    {"; ".join(_unseen) if _unseen else "ninguna"}. En esas particiones el modelo se evalúa sobre
    un tipo de falla que nunca vio, lo que debe considerarse al interpretar sus métricas.
    """)
    return


@app.cell(hide_code=True)
def _(MANIFEST_COLUMNS, cleaning_summary, fold_summary, mo, time_axes_summary):
    _paused = time_axes_summary[time_axes_summary["horas_pausa"] > 0].index
    _columns = ", ".join(f"`{column}`" for column in MANIFEST_COLUMNS)
    mo.md(f"""
    ## Cierre: entregables para la Fase IV y decisiones abiertas

    La Fase IV (ingeniería de características) recibe:

    - **El manifiesto** `data/processed/ims_snapshot_manifest.csv`
      ({cleaning_summary.loc["total", "filas_manifiesto"]} filas,
      {cleaning_summary.loc["total", "filas_validas"]} válidas), con las columnas {_columns}.
      Las características se generan solo para las filas con `valid` verdadero.
    - **Las funciones de segmentación** de `segmentation.py`: `remove_dc`, `split_segments`,
      `segment_indicators` y `summarize_segments` (mediana, p90, máximo); las características
      espectrales se calculan sobre la captura completa.
    - **Las particiones** de `folds.py`: `leave_one_test_out` ({len(fold_summary)} particiones) y
      `standardize_by_fold` / `fit_standardizer`, ajustadas solo con los tests de entrenamiento.

    Decisiones que quedan abiertas para la Fase V:

    1. La grilla de horizontes H del análisis de sensibilidad (12, 24, 36, 48, 72 h es un ejemplo
       del equipo, no una grilla fijada).
    2. El eje temporal para definir H (horas de reloj u horas de operación); los ejes difieren en
       los tests {" y ".join(_paused)}, sobre todo en el test 1.
    3. El etiquetado del rodamiento 4 del test 1 (elemento rodante), cuya huella es débil y no
       monótona, y el de los rodamientos sin falla documentada, cuyo estado al final del test es
       desconocido.
    4. Conservar las capturas del test 3 posteriores a su fin documentado
       ({cleaning_summary.loc["3", "tras_fin_documentado_conservados"]} archivos), hoy conservadas
       y marcadas.
    """)
    return


if __name__ == "__main__":
    app.run()
