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
            }
        )
    summary_table = pd.DataFrame(summary_rows).set_index("test")
    summary_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lectura de la tabla:** los tres tests coinciden en filas por archivo (20480, 1 s a 20 kHz) y
    no presentan timestamps duplicados. Los tests 1 y 2 coinciden exactamente con la documentación
    de NASA en conteo de archivos e inicio/fin. El test 3 no: ver la sección de discrepancia más
    abajo.
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
    **Test 1** muestra huecos grandes recurrentes (pausas de fin de semana/operación): el mayor es
    de ≈6.2 días (~534 000 s) alrededor del 2003-10-29, más otros de ≈4.0 y ≈3.8 días; 24 huecos
    superan 1.5× la mediana (600 s). **Test 2** es perfectamente regular (ningún hueco supera el
    umbral). **Test 3** tiene 4 huecos por encima del umbral, el mayor de ≈7.5 horas. Implicación:
    cualquier variable de "horas transcurridas" en test 1 mezcla tiempo de operación real con tiempo
    de inactividad del banco; para modelar degradación puede ser preferible usar el índice de
    snapshot (orden secuencial) en vez del tiempo de reloj, o documentar explícitamente esta mezcla.
    """)
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
            }
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El directorio crudo del test 3 contiene **6324** archivos, no los 4448 que documenta el `readme`
    de NASA, y termina el **2004-04-18 02:42:55**, ~13.3 días después del fin documentado
    (2004-04-04 19:01:57). El inicio sí coincide. La cadencia (600 s) y la ausencia de duplicados se
    mantienen consistentes en todo el rango observado, por lo que no parece un archivo corrupto o
    mezclado con otro test: el `readme` probablemente documenta una versión truncada de la
    publicación. Esto importa para el labeling: la sección de capturas posteriores al apagado (más
    abajo) sugiere que la falla real ocurre cerca del final de la grabación **extendida**, no en la
    fecha documentada.
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
    inactividad) o índice secuencial de snapshot (solo operación); ambos se usan en la sección de
    tendencias del cuaderno 02.
    4. **No hay archivos corruptos ni timestamps duplicados** en ningún test: las 20480 filas por
    archivo son consistentes en el 100% de los casos observados (ver `filas_consistentes` en la
    tabla resumen).
    5. **La discrepancia del test 3 se documenta, no se "corrige"**: se preserva el dato crudo
    completo (6324 archivos) y se reporta la discrepancia como hallazgo, dado que no hay evidencia
    de corrupción.
    """)
    return


if __name__ == "__main__":
    app.run()
