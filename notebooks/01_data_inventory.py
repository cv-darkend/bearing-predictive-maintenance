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
    # Inventario y verificación de integridad — NASA IMS

    Este cuaderno cubre la Fase II (Data Understanding) para el conjunto de datos de vibraciones de
    rodamientos NASA IMS: inventario de archivos crudos por test, verificación de integridad (conteo
    de archivos, fechas, canales, filas por archivo), cadencia de muestreo (huecos temporales), la
    discrepancia observada en el test 3 respecto a la documentación de NASA, y la detección de
    capturas posteriores al apagado del banco de pruebas. Los datos usados provienen de
    `data/interim/ims_inventory.csv` y `data/interim/ims_snapshot_indicators.csv`, generados por
    `src/bearing_predictive_maintenance/indicators.py` (`make indicators`).
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    from bearing_predictive_maintenance import ims
    from bearing_predictive_maintenance.inventory import summarize_inventory
    from bearing_predictive_maintenance.spectral import flag_post_shutdown

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#cccccc",
            "axes.edgecolor": "#444444",
            "lines.linewidth": 1.7,
        }
    )

    REPO_ROOT = Path(__file__).resolve().parent.parent
    INTERIM_DIR = REPO_ROOT / "data" / "interim"
    FIGURES_DIR = REPO_ROOT / "reports" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return (
        FIGURES_DIR,
        INTERIM_DIR,
        flag_post_shutdown,
        ims,
        pd,
        plt,
        summarize_inventory,
    )


@app.cell
def _(INTERIM_DIR, pd):
    inventory = pd.read_csv(
        INTERIM_DIR / "ims_inventory.csv", dtype={"test_id": str}, parse_dates=["timestamp"]
    )
    indicators = pd.read_csv(
        INTERIM_DIR / "ims_snapshot_indicators.csv",
        dtype={"test_id": str},
        parse_dates=["timestamp"],
    )
    inventory.groupby("test_id").size().rename("n_archivos")
    return indicators, inventory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resumen por test: documentado vs. observado
    """)
    return


@app.cell
def _(ims, inventory, pd, summarize_inventory):
    summary_rows = []
    for _test_id, spec in ims.TEST_REGISTRY.items():
        test_inventory = inventory[inventory["test_id"] == _test_id]
        summary = summarize_inventory(test_inventory, spec)
        summary_rows.append(
            {
                "test": _test_id,
                "archivos_doc": summary.documented_file_count,
                "archivos_obs": summary.observed_file_count,
                "inicio_doc": summary.documented_start,
                "inicio_obs": summary.observed_start,
                "fin_doc": summary.documented_end,
                "fin_obs": summary.observed_end,
                "canales_doc": summary.documented_channel_count,
                "canales_obs": summary.observed_channel_count,
                "filas_consistentes": summary.rows_per_file_consistent,
                "filas_moda": summary.rows_per_file_mode,
                "gap_mediana_s": summary.median_gap_seconds,
                "gap_max_s": summary.max_gap_seconds,
                "gaps_sobre_1.5x_mediana": summary.gaps_over_1_5x_median,
                "timestamps_duplicados": summary.duplicate_timestamp_count,
                "errores_lectura": summary.read_error_count,
                "archivos_con_no_finitos": summary.non_finite_file_count,
                "valores_no_finitos": summary.non_finite_value_count,
                "archivos_canales_distintos": summary.channel_count_mismatch_count,
                "archivos_saturados": summary.saturated_file_count,
                "muestras_saturadas": summary.saturated_value_count,
            }
        )
    summary_table = pd.DataFrame(summary_rows).set_index("test")
    summary_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lectura de la tabla:** en los tres tests todos los archivos tienen 20480 filas, ninguno falla
    al leerse (`errores_lectura` = 0), ninguno contiene valores no numéricos o no finitos
    (`valores_no_finitos` = 0), todos tienen el número de columnas que documenta el mapa de canales
    (`archivos_canales_distintos` = 0) y no hay timestamps duplicados. Sí hay muestras en el
    límite de adquisición (±4.999), solo en las capturas finales de cada test (sección de
    saturación más abajo). Los tests 1 y 2 coinciden
    exactamente con la documentación de NASA en conteo de archivos e inicio/fin. El test 3 no: ver
    la sección de discrepancia más abajo.

    **Frecuencia de muestreo:** el `readme` de NASA describe cada archivo como una captura de
    "1 segundo" de 20480 puntos a 20 kHz, lo que es internamente inconsistente (20480 / 20000 =
    1.024 s). Se conserva 20 kHz, el valor documentado; el cuaderno 02 muestra cómo esta
    ambigüedad afecta la lectura de las frecuencias características.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mapa canal → rodamiento/eje
    """)
    return


@app.cell
def _(ims, pd):
    channel_rows = [
        {"test": test_id, "canal": m.channel, "rodamiento": m.bearing, "eje": m.axis}
        for test_id, spec in ims.TEST_REGISTRY.items()
        for m in spec.channel_map
    ]
    pd.DataFrame(channel_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fallas documentadas por test
    """)
    return


@app.cell
def _(ims, pd):
    failure_rows = [
        {"test": test_id, "rodamiento": f.bearing, "falla_documentada": f.description}
        for test_id, spec in ims.TEST_REGISTRY.items()
        for f in spec.documented_failures
    ]
    pd.DataFrame(failure_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frecuencias características de falla (geometría Rexnord ZA-2115, 2000 RPM)
    """)
    return


@app.cell
def _(ims, pd):
    freqs = ims.characteristic_fault_frequencies()
    pd.DataFrame(
        [
            {
                "FTF_Hz": freqs.ftf,
                "BPFO_Hz": freqs.bpfo,
                "BPFI_Hz": freqs.bpfi,
                "BSF_Hz": freqs.bsf,
                "2xBSF_Hz": 2 * freqs.bsf,
            }
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cadencia de muestreo: huecos de tiempo entre snapshots consecutivos
    """)
    return


@app.cell
def _(FIGURES_DIR, ims, inventory, plt):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    for ax, (_test_id, _spec) in zip(axes, ims.TEST_REGISTRY.items(), strict=True):
        test_inv = inventory[inventory["test_id"] == _test_id].sort_values("timestamp")
        gaps = test_inv["gap_seconds"].dropna()
        median_gap = gaps.median()
        ax.plot(
            test_inv.loc[gaps.index, "elapsed_hours"],
            gaps,
            color="#3b6faa",
            linewidth=1.2,
            marker=".",
            markersize=2,
            linestyle="none",
        )
        ax.axhline(
            median_gap,
            color="#888888",
            linestyle="--",
            linewidth=1,
            label=f"mediana = {median_gap:.0f} s",
        )
        ax.set_yscale("log")
        ax.set_title(f"Test {_test_id}")
        ax.set_ylabel("hueco (s, escala log)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("horas transcurridas desde el primer snapshot")
    fig.suptitle("Huecos de tiempo entre snapshots consecutivos, por test")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_cadence_gaps.png", dpi=150)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Test 1** muestra huecos grandes recurrentes (pausas de operación): el mayor es de ≈6.2 días
    (~534 000 s), entre el 2003-10-23 y el 2003-10-29, más otros de ≈4.0 y ≈3.8 días; 24 huecos
    superan 1.5× la mediana (600 s). **Test 2** es perfectamente regular (ningún hueco supera el
    umbral). **Test 3** tiene 4 huecos por encima del umbral, el mayor de ≈7.5 horas. Implicación:
    cualquier variable de "horas transcurridas" en test 1 mezcla tiempo de operación real con tiempo
    de inactividad del banco. La cadencia tampoco es uniforme dentro de la operación (tabla
    siguiente), por lo que el índice de snapshot tampoco equivale a tiempo de operación.
    """)
    return


@app.cell
def _(inventory, pd):
    # Runs of consecutive snapshots taken at a 5-minute cadence instead of the usual 10.
    _rows = []
    for _test_id, _group in inventory.sort_values("timestamp").groupby("test_id"):
        _short = (_group["gap_seconds"] - 300).abs() <= 30
        _run_id = (_short != _short.shift()).cumsum()
        for _, _run in _group[_short].groupby(_run_id[_short]):
            _rows.append(
                {
                    "test": _test_id,
                    "inicio": _run["timestamp"].iloc[0],
                    "fin": _run["timestamp"].iloc[-1],
                    "huecos_de_300s": len(_run),
                }
            )
    cadence_runs = pd.DataFrame(_rows, columns=["test", "inicio", "fin", "huecos_de_300s"])
    cadence_runs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El `readme` documenta que en el test 1 los primeros 43 archivos se tomaron cada 5 minutos; en
    los datos ese tramo aparece como 40 huecos de 300 s seguidos, más una ráfaga corta el mismo
    día y otra el 2003-11-09. Los datos muestran además una ráfaga larga a 5 minutos el 2003-11-20
    (tabla anterior). Un eje temporal basado en el índice de snapshot contaría esas capturas como
    si representaran el doble de tiempo de operación.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Longitud de las series: horas de reloj vs. horas de operación

    Las horas de operación acumulan cada hueco entre capturas solo hasta la cadencia nominal
    (600 s), de modo que las pausas del banco no suman tiempo.
    """)
    return


@app.cell
def _(inventory, pd):
    NOMINAL_CADENCE_S = 600.0
    _rows = []
    for _test_id, _group in inventory.sort_values("timestamp").groupby("test_id"):
        _gaps = _group["gap_seconds"].fillna(0.0)
        _rows.append(
            {
                "test": _test_id,
                "archivos": len(_group),
                "horas_reloj": float(_group["elapsed_hours"].iloc[-1]),
                "horas_operacion": float(_gaps.clip(upper=NOMINAL_CADENCE_S).sum() / 3600),
            }
        )
    series_length = pd.DataFrame(_rows)
    series_length
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saturación del sistema de adquisición

    Archivos con muestras en el límite de adquisición (|x| ≥ 4.99; los registros saturan en
    ±4.999). En esas muestras la amplitud real es desconocida. Solo aparecen al final de cada
    test: 3 archivos del test 1 (9 muestras, desde las 23:13:21 del último día), 2 del test 2
    (47 muestras, entre las 05:02:39 y las 05:42:39 del último día) y 4 del test 3 (89 muestras,
    desde las 01:32:55 del último día). Las capturas tardías de los tests 1 y 3 que usa el
    cuaderno 02 están entre ellas, así que sus picos y su curtosis están recortados.
    """)
    return


@app.cell
def _(inventory):
    saturated_files = inventory.loc[
        inventory["n_saturated"] > 0, ["test_id", "file", "timestamp", "n_saturated"]
    ]
    saturated_files
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discrepancia del test 3
    """)
    return


@app.cell
def _(ims, inventory, pd, summarize_inventory):
    test3_spec = ims.TEST_REGISTRY["3"]
    test3_summary = summarize_inventory(inventory[inventory["test_id"] == "3"], test3_spec)
    pd.DataFrame(
        [
            {
                "archivos_documentados": test3_summary.documented_file_count,
                "archivos_observados": test3_summary.observed_file_count,
                "fin_documentado": test3_summary.documented_end,
                "fin_observado": test3_summary.observed_end,
                "dias_extra_observados": (
                    test3_summary.observed_end - test3_summary.documented_end
                ).total_seconds()
                / 86400,
                "archivos_hasta_fin_documentado": int(
                    (
                        inventory.loc[inventory["test_id"] == "3", "timestamp"]
                        <= test3_summary.documented_end
                    ).sum()
                ),
            }
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El directorio crudo del test 3 contiene **6324** archivos, no los 4448 que documenta el `readme`
    de NASA, y termina el **2004-04-18 02:42:55**, ~13.3 días después del fin documentado
    (2004-04-04 19:01:57). El inicio sí coincide, y exactamente 4448 archivos caen en o antes del
    fin documentado, el mismo número que documenta el `readme`. La cadencia (600 s) y la ausencia
    de duplicados se mantienen en todo el rango observado. La hipótesis más simple es que el
    `readme` describe una versión truncada de la publicación; no se puede confirmar con los datos.
    La copia usada proviene del espejo de Kaggle (`vinayak123tyagi/bearing-dataset`), donde el test
    3 está en la carpeta `3rd_test/4th_test/txt`. Esto importa para el etiquetado: la sección de
    capturas posteriores al apagado (más abajo) sugiere que la grabación termina con el banco
    detenido al final del rango **extendido**, no en la fecha documentada.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Capturas posteriores al apagado del banco
    """)
    return


@app.cell
def _(flag_post_shutdown, ims, indicators, pd):
    max_rms = (
        indicators.groupby(["test_id", "file", "timestamp"])["rms"]
        .max()
        .reset_index()
        .sort_values(["test_id", "timestamp"])
    )
    flagged_frames = []
    for _test_id in ims.TEST_REGISTRY:
        test_max_rms = (
            max_rms[max_rms["test_id"] == _test_id].sort_values("timestamp").reset_index(drop=True)
        )
        test_max_rms["post_shutdown"] = flag_post_shutdown(
            test_max_rms["rms"], baseline_n=100, fraction_of_baseline_median=0.1
        )
        flagged_frames.append(test_max_rms)
    max_rms_flagged = pd.concat(flagged_frames, ignore_index=True)
    post_shutdown = max_rms_flagged[max_rms_flagged["post_shutdown"]]
    post_shutdown
    return (post_shutdown,)


@app.cell
def _(post_shutdown):
    post_shutdown_counts = post_shutdown.groupby("test_id").size().rename("n_capturas_post_apagado")
    post_shutdown_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Usando el máximo RMS entre canales por archivo, con umbral = 10% de la mediana de los primeros
    100 archivos: **test 1** no tiene capturas post-apagado, **test 2** tiene **2** (las dos
    últimas), **test 3** tiene **1** (la última). Estas capturas casi planas (RMS cercano a 0) deben
    excluirse de cualquier análisis de degradación o de la definición de "último snapshot válido".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusiones: criterios de inclusión/exclusión propuestos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. **Excluir capturas post-apagado**: usar `flag_post_shutdown` (o el criterio equivalente) para
    descartar las últimas 1-2 filas de los tests 2 y 3 antes de calcular "tiempo hasta la falla" o
    entrenar modelos.
    2. **Tratar el fin documentado del test 3 como una versión truncada**, no como el momento de
    falla: usar el fin observado (menos las capturas post-apagado) como límite temporal real del
    test.
    3. **Los huecos de tiempo del test 1 no son errores**: son pausas de operación reales. Para
    ventanas de tiempo, decidir explícitamente si "horas transcurridas" usa tiempo de reloj (incluye
    inactividad), índice de snapshot (cadencia no uniforme) u horas de operación sin pausas (tabla
    de longitud de las series).
    4. **No hay archivos corruptos ni timestamps duplicados** en ningún test: 0 errores de lectura,
    0 valores no finitos, 20480 filas y el número de canales documentado en el 100% de los archivos
    (columnas de integridad de la tabla resumen).
    5. **La discrepancia del test 3 se documenta, no se "corrige"**: se preserva el dato crudo
    completo (6324 archivos) y se reporta la discrepancia como hallazgo, dado que no hay evidencia
    de corrupción.
    """)
    return


if __name__ == "__main__":
    app.run()
