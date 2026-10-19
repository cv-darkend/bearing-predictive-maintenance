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
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from bearing_predictive_maintenance import ims, spectral
    from bearing_predictive_maintenance.folds import leave_one_test_out, standardize_by_fold
    from bearing_predictive_maintenance.indicators import TIME_DOMAIN_INDICATOR_NAMES
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
        split_segments,
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
        TIME_DOMAIN_INDICATOR_NAMES,
        ims,
        leave_one_test_out,
        np,
        pd,
        plt,
        remove_dc,
        segment_indicators,
        spectral,
        split_segments,
        standardize_by_fold,
        summarize_segments,
        time,
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
    # +/-3% relative search window used for fault lines in Phase II. The literal
    # mirrors FAULT_REL_TOLERANCE of notebooks/02_signal_exploration.py: src/ has
    # no shared constant for it.
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
def _(
    FAILURE_MODE_ES,
    RAW_ROOT,
    SATURATION_LEVEL,
    ims,
    manifest,
    np,
    pd,
    remove_dc,
    segment_indicators,
):
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
        _raw_channel = _array[:, int(_row.channel) - 1]
        _signal = remove_dc(_raw_channel)
        _mode = FAILURE_MODE_ES[_row.failure_mode]
        _label = f"Test {_row.test_id} - Rodamiento {int(_row.bearing)} ({_mode})"
        demo_snapshots.append(
            {
                "test_id": _row.test_id,
                "file": _row.file,
                "channel": int(_row.channel),
                "bearing": int(_row.bearing),
                "label": _label,
                # Samples clipped at the acquisition limit (inventory.SATURATION_LEVEL).
                "saturated_samples": int(np.sum(np.abs(_raw_channel) >= SATURATION_LEVEL)),
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
        _row = {
            "caso": _snapshot["label"],
            "archivo": _snapshot["file"],
            "muestras_saturadas": _snapshot["saturated_samples"],
        }
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
            "muestras_saturadas",
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
def _(SATURATION_LEVEL, mo, segment_summary):
    _rms_ratio = segment_summary["rms_max"] / segment_summary["rms_median"]
    _ratio = segment_summary["kurtosis_max"] / segment_summary["kurtosis_median"]
    _widest = segment_summary.loc[_ratio.idxmax()]
    _files = segment_summary.drop_duplicates("archivo")
    _saturated = "; ".join(
        f"{_row.caso}: {_row.muestras_saturadas}" for _row in _files.itertuples()
    )
    _n_saturated_files = int((_files["muestras_saturadas"] > 0).sum())
    mo.md(f"""
    Los cuatro casos corresponden a {segment_summary["archivo"].nunique()} archivos distintos (los
    rodamientos 3 y 4 del test 1 comparten archivo). En los cuatro casos el RMS varía poco entre
    segmentos (el máximo por segmento es entre {_rms_ratio.min():.2f} y {_rms_ratio.max():.2f}
    veces la mediana), mientras que la curtosis varía más: el caso con mayor dispersión es
    {_widest["caso"]}, donde la curtosis máxima por segmento es {_ratio.max():.2f} veces la
    mediana ({_widest["kurtosis_max"]:.2f} frente a {_widest["kurtosis_median"]:.2f}; la curtosis
    de la captura completa es {_widest["kurtosis_captura_completa"]:.2f}). Por eso se guardan los
    tres resúmenes y la selección se deja a la Fase IV.

    **Advertencia:** {_n_saturated_files} de estos {len(_files)} archivos finales tienen muestras
    saturadas (±{SATURATION_LEVEL}) en el canal mostrado (columna `muestras_saturadas`, por
    archivo: {_saturated}). En esos casos los picos, el factor de cresta y la curtosis están
    recortados, de modo que la tabla subestima la impulsividad real.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Elección de la longitud de ventana

    El documento del proyecto pide que las ventanas sean "de longitud homogénea, definidas con
    base en criterios de estabilidad estadística, contenido espectral y capacidad computacional".
    La sección anterior justifica el contenido espectral; esta sección compara las longitudes
    L = 1024, 2048, 4096, 10240 y 20480 muestras frente a los tres criterios, sobre una muestra
    de todo el conjunto y no solo sobre las capturas finales.

    **Muestra.** Para cada rodamiento con falla documentada (canal del eje x en el test 1, canal
    único en los tests 2 y 3; capturas válidas del manifiesto) se toman tres grupos: las 20
    primeras capturas válidas (`inicio`), 20 capturas espaciadas uniformemente a lo largo del test
    (`a_lo_largo`) y las 20 últimas capturas válidas (`final`). En el test 1 los rodamientos 3 y 4
    comparten archivos y se miden en sus propios canales.

    **Criterios, para cada L:**

    - *Estabilidad estadística:* coeficiente de variación (desviación estándar poblacional sobre
      la media) del RMS y de la curtosis entre los segmentos contiguos y sin solapamiento de cada
      captura (`segmentation.segment_indicators`); se reportan su mediana y su percentil 90 por
      grupo. Con L = 20480 hay un solo segmento y el coeficiente no está definido (n/a).
    - *Contenido espectral:* resolución fs/L frente al ancho de la ventana de búsqueda de ±3%
      alrededor de BPFO.
    - *Capacidad computacional:* tiempo de pared por captura (`time.perf_counter`, mediana) de
      `segment_indicators` sobre los segmentos de longitud L más un espectro de envolvente
      (`spectral.envelope_spectrum`, banda fija 2300-3800 Hz, la elegida en la Fase II para el
      test 2) por cada ventana de longitud L, y el número de filas y de valores resultantes por
      captura y para las filas válidas del manifiesto. La banda fija solo sirve para medir costo;
      no es una elección de características.

    Antecedente de la Fase II (`reports/eda_nasa_ims.md`, "Estabilidad de los segmentos"): con
    segmentos de 2048 muestras, el coeficiente de variación mediano del RMS entre segmentos es de
    2% a 5.5%, y el de la curtosis llega a 32% al final del test 1 (rodamiento 3).
    """)
    return


@app.cell
def _(manifest, np, pd):
    WINDOW_LENGTHS = (1024, 2048, 4096, 10240, 20480)
    SAMPLE_PER_GROUP = 20
    # Fixed band (Phase II choice for test 2), used only to time one envelope spectrum.
    TIMING_ENVELOPE_BAND_HZ = (2300.0, 3800.0)
    _failed = (
        manifest[manifest["valid"] & manifest["documented_failure"]]
        .loc[lambda frame: frame["axis"].isin(["x", "single"])]
        .sort_values(["test_id", "bearing", "timestamp"])
    )
    _parts = []
    for _key, _rows in _failed.groupby(["test_id", "bearing"]):
        _rows = _rows.reset_index(drop=True)
        _even = np.unique(np.linspace(0, len(_rows) - 1, SAMPLE_PER_GROUP).round().astype(int))
        _parts.append(_rows.head(SAMPLE_PER_GROUP).assign(grupo="inicio"))
        _parts.append(_rows.iloc[_even].assign(grupo="a_lo_largo"))
        _parts.append(_rows.tail(SAMPLE_PER_GROUP).assign(grupo="final"))
    window_sample = pd.concat(_parts, ignore_index=True)
    window_sample.groupby(["test_id", "bearing", "grupo"]).size().unstack()
    return TIMING_ENVELOPE_BAND_HZ, WINDOW_LENGTHS, window_sample


@app.cell
def _(
    RAW_ROOT,
    TIMING_ENVELOPE_BAND_HZ,
    WINDOW_LENGTHS,
    ims,
    pd,
    remove_dc,
    segment_indicators,
    spectral,
    split_segments,
    time,
    window_sample,
):
    def coefficient_of_variation(values: pd.Series) -> float:
        # Undefined with a single segment (L = 20480).
        if len(values) < 2:
            return float("nan")
        return float(values.std(ddof=0) / values.mean())

    _fs = ims.SAMPLING_RATE_HZ
    _arrays = {}
    _rows = []
    for _row in window_sample.itertuples():
        _key = (_row.test_id, _row.file)
        if _key not in _arrays:
            _spec = ims.TEST_REGISTRY[_row.test_id]
            _arrays[_key] = ims.load_snapshot(RAW_ROOT / _spec.raw_relative_path / _row.file)
        _signal = remove_dc(_arrays[_key][:, int(_row.channel) - 1])
        for _length in WINDOW_LENGTHS:
            _start = time.perf_counter()
            _table = segment_indicators(_signal, _length)
            for _window in split_segments(_signal, _length):
                spectral.envelope_spectrum(_window, _fs, TIMING_ENVELOPE_BAND_HZ)
            _elapsed = time.perf_counter() - _start
            _rows.append(
                {
                    "test_id": _row.test_id,
                    "bearing": int(_row.bearing),
                    "grupo": _row.grupo,
                    "saturated_file": bool(_row.saturated_file),
                    "L": _length,
                    "cv_rms": coefficient_of_variation(_table["rms"]),
                    "cv_kurtosis": coefficient_of_variation(_table["kurtosis"]),
                    "seconds": _elapsed,
                }
            )
    # Cost of the chosen "Mixto" design on the same snapshots: time-domain indicators
    # on 2048-sample segments plus one envelope spectrum of the full snapshot.
    _mixed_seconds = []
    for _row in window_sample.itertuples():
        _signal = remove_dc(_arrays[(_row.test_id, _row.file)][:, int(_row.channel) - 1])
        _start = time.perf_counter()
        segment_indicators(_signal)
        spectral.envelope_spectrum(_signal, _fs, TIMING_ENVELOPE_BAND_HZ)
        _mixed_seconds.append(time.perf_counter() - _start)
    window_measurements = pd.DataFrame(_rows)
    mixed_seconds = pd.Series(_mixed_seconds, name="seconds")
    n_sample_files = len(_arrays)
    window_measurements.head()
    return mixed_seconds, n_sample_files, window_measurements


@app.cell
def _(
    FAULT_REL_TOLERANCE,
    TIME_DOMAIN_INDICATOR_NAMES,
    WINDOW_LENGTHS,
    ims,
    pd,
    resolution,
    valid_rows,
    window_measurements,
):
    _n_valid = len(valid_rows)
    _rows = []
    for _length in WINDOW_LENGTHS:
        _subset = window_measurements[window_measurements["L"] == _length]
        _row = {"L": _length}
        for _group in ("inicio", "a_lo_largo", "final"):
            _values = _subset[_subset["grupo"] == _group]
            for _metric, _name in (("cv_rms", "cv_rms"), ("cv_kurtosis", "cv_curtosis")):
                _row[f"{_name}_mediana_{_group}"] = _values[_metric].median()
                _row[f"{_name}_p90_{_group}"] = _values[_metric].quantile(0.9)
        _n_windows = ims.EXPECTED_ROWS_PER_SNAPSHOT // _length
        _df = ims.SAMPLING_RATE_HZ / _length
        _seconds = float(_subset["seconds"].median())
        _row.update(
            {
                "resolucion_hz": _df,
                "lineas_en_ventana_bpfo": resolution["window_width_hz"] / _df,
                "ms_por_captura": 1e3 * _seconds,
                "ventanas_por_captura": _n_windows,
                "filas_conjunto_si_ventana_es_fila": _n_windows * _n_valid,
                "valores_tiempo_por_captura": _n_windows * len(TIME_DOMAIN_INDICATOR_NAMES),
                "lineas_envolvente_por_captura": _n_windows * (_length // 2 + 1),
                "segundos_conjunto": _seconds * _n_valid,
            }
        )
        _rows.append(_row)
    window_comparison = pd.DataFrame(_rows).set_index("L")
    assert FAULT_REL_TOLERANCE == 0.03  # the window width in `resolution` uses it
    window_comparison.T.round(4)
    return (window_comparison,)


@app.cell(hide_code=True)
def _(
    TIME_DOMAIN_INDICATOR_NAMES,
    mixed_seconds,
    mo,
    n_sample_files,
    pd,
    resolution,
    valid_rows,
    window_comparison,
    window_measurements,
    window_sample,
):
    def pct(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:.1%}"

    _c = window_comparison
    _lines = []
    for _length, _r in _c.iterrows():
        _lines.append(
            f"| {_length} | {pct(_r['cv_rms_mediana_inicio'])} / {pct(_r['cv_rms_mediana_final'])} "
            f"| {pct(_r['cv_rms_p90_final'])} "
            f"| {pct(_r['cv_curtosis_mediana_inicio'])} / {pct(_r['cv_curtosis_mediana_final'])} "
            f"| {pct(_r['cv_curtosis_p90_final'])} "
            f"| {_r['resolucion_hz']:.3f} | {_r['lineas_en_ventana_bpfo']:.2f} "
            f"| {_r['ms_por_captura']:.2f} | {int(_r['ventanas_por_captura'])} "
            f"| {int(_r['filas_conjunto_si_ventana_es_fila'])} | {_r['segundos_conjunto']:.0f} |"
        )
    _header = [
        "| L | CV RMS mediana (inicio / final) | CV RMS p90 final "
        "| CV curtosis mediana (inicio / final) | CV curtosis p90 final | fs/L (Hz) "
        "| Líneas en ±3% BPFO | ms por captura | Ventanas | Filas del conjunto "
        "| s del conjunto |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    _table = "\n".join(_header + _lines)
    _n_valid = len(valid_rows)
    _mixed_ms = 1e3 * float(mixed_seconds.median())
    _saturated = int(window_sample["saturated_file"].sum())
    _saturated_by_group = ", ".join(
        f"{_count} en `{_group}`"
        for _group, _count in window_sample[window_sample["saturated_file"]]
        .groupby("grupo")
        .size()
        .items()
    )
    _l2048 = _c.loc[2048]
    _l4096 = _c.loc[4096]
    _l10240 = _c.loc[10240]
    _per_bearing = (
        window_measurements[
            (window_measurements["L"] == 2048) & (window_measurements["grupo"] == "final")
        ]
        .groupby(["test_id", "bearing"])["cv_kurtosis"]
        .median()
    )
    _late_max = _per_bearing.idxmax()
    _multi = _c[_c["ventanas_por_captura"] > 1]
    # Guards for the qualitative statements written in the narrative below.
    assert (_multi["cv_curtosis_mediana_final"] > _multi["cv_curtosis_mediana_inicio"]).all()
    assert max(_l2048["cv_rms_mediana_inicio"], _l2048["cv_rms_mediana_final"]) < 0.05
    assert (_c.loc[[4096, 10240], "cv_rms_mediana_final"] < _l2048["cv_rms_mediana_final"]).all()
    assert list(_c.index[_c["lineas_en_ventana_bpfo"] > 10]) == [20480]
    _max_minutes = float(_c["segundos_conjunto"].max()) / 60
    mo.md(f"""
    Muestra medida: {len(window_sample)} pares (captura, canal) de {n_sample_files} archivos
    distintos; {_saturated} de esos pares son archivos con saturación
    ({_saturated_by_group}; el grupo `a_lo_largo` incluye ambos extremos del test, también la
    última captura válida).
    Tiempos medidos en esta máquina, en un solo hilo; son órdenes de magnitud, no un banco de
    pruebas. Columnas: coeficiente de variación (CV) mediano `inicio` / `final` y percentil 90
    `final`, resolución, líneas espectrales dentro de la ventana de ±3% de BPFO
    ({resolution["window_width_hz"]:.2f} Hz), tiempo mediano por captura, ventanas por captura,
    filas del conjunto si cada ventana fuera una fila ({_n_valid} filas válidas × ventanas) y
    tiempo estimado para todo el conjunto.

    {_table}

    **Estabilidad estadística.** El CV entre segmentos baja al crecer L, porque cada estimación
    usa más muestras: con L = 2048 el CV mediano del RMS es {pct(_l2048["cv_rms_mediana_inicio"])}
    al inicio y {pct(_l2048["cv_rms_mediana_final"])} al final (coherente con el 2-5.5% de la
    Fase II), y el de la curtosis {pct(_l2048["cv_curtosis_mediana_inicio"])} y
    {pct(_l2048["cv_curtosis_mediana_final"])}, con un percentil 90 final de
    {pct(_l2048["cv_curtosis_p90_final"])}; el mayor CV mediano final de la curtosis es el del
    test {_late_max[0]}, rodamiento {_late_max[1]} ({pct(_per_bearing.max())}). Con L = 4096 los
    valores bajan a {pct(_l4096["cv_rms_mediana_final"])} (RMS) y
    {pct(_l4096["cv_curtosis_mediana_final"])} (curtosis) al final, y con L = 10240 a
    {pct(_l10240["cv_rms_mediana_final"])} y {pct(_l10240["cv_curtosis_mediana_final"])}. Por sí
    solo, este criterio favorece ventanas más largas. Pero parte de la variación al final no es
    ruido de estimación sino la actividad impulsiva del defecto, que no aparece en todos los
    segmentos: el CV de la curtosis sube del inicio al final en todas las L con más de un
    segmento. Las ventanas largas la promedian, y con 2 segmentos (L = 10240) ya no se pueden
    calcular resúmenes robustos como el percentil 90.

    **Contenido espectral.** Solo la captura completa (L = 20480) da más de 10 líneas dentro de
    la ventana de ±3% de BPFO; L = 10240 da {_l10240["lineas_en_ventana_bpfo"]:.1f} y L = 2048
    da {_l2048["lineas_en_ventana_bpfo"]:.2f}.

    **Capacidad computacional.** Ninguna longitud es restrictiva: el costo por captura es de
    milisegundos, y el conjunto completo ({_n_valid} filas válidas) se procesa en a lo sumo
    {_max_minutes:.0f} min con cualquier L. El diseño "Mixto" (indicadores en 10 segmentos de
    2048 más un espectro de envolvente de la captura completa) cuesta {_mixed_ms:.2f} ms por
    captura, unos {_mixed_ms * _n_valid / 1e3:.0f} s para el conjunto, y produce una fila por
    captura con {len(TIME_DOMAIN_INDICATOR_NAMES)} indicadores × 3 resúmenes de tiempo. Las
    ventanas cortas multiplican las filas y los valores de tiempo (L = 1024 daría
    {int(_c.loc[1024, "filas_conjunto_si_ventana_es_fila"])} filas), sin ganancia espectral.

    **Conclusión.** El diseño "Mixto" queda respaldado, con un matiz honesto: L = 2048 no es la
    longitud más estable (L = 4096 o 10240 dan CV menores), sino el compromiso que mantiene el CV
    del RMS por debajo de 5% en la mediana y conserva 10 segmentos para los resúmenes mediana, p90
    y máximo que capturan la impulsividad tardía. L = 4096 (5 segmentos) es una alternativa
    razonable para la Fase IV si se prefiere estabilidad sobre detalle. Para el espectro, los
    datos favorecen claramente la captura completa. El costo no discrimina entre opciones.
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
    (por ejemplo, umbrales o la tolerancia de búsqueda espectral) en las fases siguientes. En
    particular, las bandas de envolvente de la Fase II se eligieron mirando capturas del final de
    la vida útil, y la tolerancia de ±3% alrededor de las frecuencias de falla se tomó de esa misma
    exploración: ambas deben fijarse a priori (antes de ver el test de prueba) o reajustarse dentro
    de cada partición solo con los tests de entrenamiento.

    **Columnas del manifiesto que no son características.** Las columnas siguientes sirven solo
    para filtrar, particionar y etiquetar; la Fase IV y la Fase V **no** deben usarlas como
    entradas del modelo, porque filtran información del final de la vida útil o de la identidad
    del test o del rodamiento:

    - `hours_to_end_clock`, `hours_to_end_operating`: son la distancia al final, es decir, la
      variable de la que se deriva la etiqueta.
    - `clock_hours`, `operating_hours`: sumadas con las horas hasta el final dan una constante por
      test, y los tests tienen duraciones distintas, de modo que revelan cuánto falta.
    - `saturated_file`: solo es verdadera en archivos de la última hora (o menos) de cada test,
      cuya saturación está en los canales que fallan.
    - `after_documented_end`, `documented_failure`, `failure_mode`: describen el desenlace del
      test o del rodamiento.
    - `valid`, `post_shutdown`: marcas de limpieza para filtrar.
    - `test_id`, `bearing`, `channel`, `timestamp`: identidad y fecha (el rodamiento 3 falla en 2
      de los 3 tests, de modo que `bearing` correlaciona con la falla).
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
