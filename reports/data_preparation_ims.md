# Preparación de datos — Conjunto de datos NASA IMS de rodamientos

## Resumen

Este informe documenta la Fase III (Data Preparation, CRISP-DM) del conjunto de datos de
vibraciones de rodamientos NASA IMS. El documento del proyecto define su alcance así:
"Fase III: A su vez se someten los datos a procesos de limpieza, normalización y control de
calidad, con el fin de segmentar las señales en ventanas homogéneas; a través de las
siguientes actividades: Limpieza y procesamiento [...] Normalización y estandarización [...]
Segmentación de las señales".

El resultado de la fase es un manifiesto por captura y canal
(`data/processed/ims_snapshot_manifest.csv`, 46480 filas, 46468 válidas) con las marcas de
limpieza y dos ejes temporales, un diseño de segmentación "Mixto" (captura completa para las
características espectrales, 10 segmentos de 2048 muestras para las de tiempo) y un esquema
de validación leave-one-test-out con estandarización ajustada solo con los tests de
entrenamiento. No se crea ninguna variable objetivo: el horizonte de predicción H se decide
en la Fase V, y el manifiesto guarda las horas hasta el final de cada test para derivarlo.

Todos los números de este informe provienen de la ejecución de
`notebooks/03_data_preparation.py` (notebook [marimo](https://marimo.io/), sin salidas
almacenadas en el fuente); cada sección indica la sección del cuaderno de la que provienen.
El cuaderno lee el manifiesto que genera `make manifest` a partir de las tablas de
`data/interim/` (`make indicators`), la tabla de indicadores por captura y, para la
demostración de segmentación y la comparación de longitudes de ventana, capturas crudas
puntuales de `data/raw/` (solo lectura). `make notebooks` lo exporta a
`build/notebooks/03_data_preparation.ipynb` (con salidas) y `.html`.

## Decisiones del equipo que enmarcan la fase

- **Segmentación "Mixto"** ("De acuerdo con el Mixto."): la unidad de análisis es la captura
  completa de 20480 muestras, que es también la ventana de las características espectrales;
  las características de tiempo se calculan en sus 10 segmentos contiguos de 2048 muestras y
  se resumen a una fila por captura.
- **Horizonte H como análisis de sensibilidad** ("La idea creeria que es un analisis de
  sensibilidades, por ejemplo 12, 24, 36, 48, 72 pero esto solo lo doy como ejemplo."). La
  grilla no se fija en esta fase (supuesto: se elige en la Fase V). La Fase III es
  independiente de H: guarda las horas hasta el final y no etiqueta.

## Limpieza y control de calidad (cuaderno 03, "Limpieza y control de calidad")

El manifiesto tiene una fila por (test, archivo, canal) y se construye solo a partir de las
tablas intermedias, sin releer los archivos crudos (`src/bearing_predictive_maintenance/manifest.py`).
Criterios por archivo:

1. **Captura posterior al apagado — excluida.** Regla de la Fase II: el RMS máximo entre los
   canales del archivo es menor que 0.1 veces la mediana de los primeros 100 archivos de su
   test (`spectral.flag_post_shutdown`).
2. **Error de lectura — excluido.** El inventario registra un error al leer el archivo.
3. **Archivo con saturación — se conserva, marcado.** Alguna muestra alcanza ±4.99
   (`inventory.SATURATION_LEVEL`). Los 9 archivos saturados son válidos y están a lo sumo
   1.00 h de reloj antes de la última captura válida de su test: son capturas del final de la
   vida útil, donde está la evidencia de falla, por lo que excluirlos sesgaría el conjunto.
   Sus picos y curtosis están recortados (limitación ya señalada en la Fase II).
4. **Captura posterior al fin documentado — se conserva, marcada.** El test 3 continúa más
   allá de la fecha de fin del `readme` de NASA (discrepancia documentada en la Fase II).

Una captura es **válida** cuando no es posterior al apagado ni tiene error de lectura.

| Test | Archivos | Válidos | Post-apagado (excluidos) | Error de lectura (excluidos) | Saturados (conservados) | Tras fin documentado (conservados) | Filas manifiesto | Filas válidas |
|---|---|---|---|---|---|---|---|---|
| 1 | 2156 | 2156 | 0 | 0 | 3 | 0 | 17248 | 17248 |
| 2 | 984 | 982 | 2 | 0 | 2 | 0 | 3936 | 3928 |
| 3 | 6324 | 6323 | 1 | 0 | 4 | 1876 | 25296 | 25292 |
| Total | 9464 | 9461 | 3 | 0 | 9 | 1876 | 46480 | 46468 |

La señal de cada canal se limpia además a nivel de muestra restándole su media (nivel DC,
`segmentation.remove_dc`).

## Ejes temporales (cuaderno 03, "Ejes temporales")

Cada captura tiene dos ejes en el manifiesto:

- **Horas de reloj** (`clock_hours`): tiempo transcurrido desde la primera captura del test.
- **Horas de operación** (`operating_hours`): cada hueco entre capturas consecutivas cuenta
  como máximo un paso de la cadencia nominal de 600 s, de modo que las pausas del banco no
  suman tiempo de funcionamiento. Hay una única definición compartida en el código
  (`manifest.operating_hours`).

Para ambos ejes se guardan las horas hasta la última captura válida del test
(`hours_to_end_clock`, `hours_to_end_operating`).

| Test | Horas de reloj (última válida) | Horas de operación (última válida) | Horas de pausa | Fracción de pausa | Horas de operación (todos los archivos) |
|---|---|---|---|---|---|
| 1 | 827.56 | 345.27 | 482.29 | 58% | 345.27 |
| 2 | 163.50 | 163.50 | 0.00 | 0% | 163.83 |
| 3 | 1073.09 | 1053.22 | 19.87 | 2% | 1053.38 |

La figura `reports/figures/03_time_axes.png` compara ambos ejes por test. El test 1
concentra las pausas; en el test 3 hay 19.87 h de pausa.

**Ilustración de la grilla de sensibilidad (no es una etiqueta).** Con la grilla que el
equipo dio como ejemplo, el número de capturas válidas de cada rodamiento con falla
documentada que quedan a H horas o menos de la última captura válida es:

| Test | Rodamiento | Modo de falla | Capturas válidas | H = 12 (reloj / operación) | H = 24 | H = 36 | H = 48 | H = 72 |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | pista interna | 2156 | 31 / 73 | 39 / 146 | 105 / 219 | 165 / 291 | 296 / 435 |
| 1 | 4 | elemento rodante | 2156 | 31 / 73 | 39 / 146 | 105 / 219 | 165 / 291 | 296 / 435 |
| 2 | 1 | pista externa | 982 | 73 / 73 | 145 / 145 | 217 / 217 | 289 / 289 | 433 / 433 |
| 3 | 3 | pista externa | 6323 | 73 / 73 | 145 / 145 | 217 / 217 | 289 / 289 | 390 / 433 |

El eje cambia los conteos donde hay pausas dentro de las últimas H horas: en el test 1 en
todos los H de la grilla, en el test 3 solo con H = 72 h, y nunca en el test 2. Las horas
hasta el final se miden respecto de la última captura válida del test, común a sus
rodamientos; por eso los rodamientos 3 y 4 del test 1 tienen los mismos conteos.

## Segmentación de las señales (cuaderno 03, "Segmentación de las señales")

**Diseño.** La captura completa (20480 muestras) es la unidad de análisis y la ventana
espectral; las características de tiempo se calculan en 10 segmentos contiguos, sin
solapamiento, de 2048 muestras (`segmentation.split_segments`, `segment_indicators`) y se
resumen a una fila por captura con la mediana, el percentil 90 y el máximo
(`summarize_segments`).

**Justificación espectral.** Con fs = 20000 Hz, la captura completa da una resolución de
0.977 Hz y un segmento de 2048 muestras, 9.77 Hz. La ventana de búsqueda de ±3% alrededor de
BPFO (236.40 Hz), usada en la Fase II, mide 14.18 Hz: contiene 14.5 líneas espectrales con la
captura completa y solo 1.45 con un segmento, insuficiente para ubicar el pico de falla. Por
eso las características espectrales se calculan sobre la captura completa.

**Justificación de los segmentos para el dominio del tiempo.** Los resúmenes por segmento
separan el nivel típico de la captura de la actividad impulsiva concentrada en pocos
segmentos. En la última captura válida de cada rodamiento con falla documentada (en el
test 1, eje x):

| Caso | Archivo | Muestras saturadas | RMS mediana | RMS máx. | RMS captura completa | Curtosis mediana | Curtosis p90 | Curtosis máx. | Curtosis captura completa |
|---|---|---|---|---|---|---|---|---|---|
| Test 1 - Rodamiento 3 (pista interna) | 2003.11.25.23.39.56 | 6 | 0.507 | 0.638 | 0.579 | 9.908 | 17.292 | 18.947 | 8.903 |
| Test 1 - Rodamiento 4 (elemento rodante) | 2003.11.25.23.39.56 | 0 | 0.198 | 0.206 | 0.198 | 3.591 | 3.882 | 4.166 | 3.626 |
| Test 2 - Rodamiento 1 (pista externa) | 2004.02.19.06.02.39 | 0 | 0.482 | 0.510 | 0.484 | 7.844 | 8.970 | 9.021 | 7.892 |
| Test 3 - Rodamiento 3 (pista externa) | 2004.04.18.02.32.55 | 47 | 0.746 | 0.808 | 0.757 | 9.998 | 12.712 | 13.865 | 10.060 |

El RMS varía poco entre segmentos (el máximo por segmento es entre 1.04 y 1.26 veces la
mediana), mientras que la curtosis varía más: en el rodamiento 3 del test 1 la curtosis
máxima por segmento es 1.91 veces la mediana (18.95 frente a 9.91), con una curtosis de la
captura completa de 8.90. Se guardan los tres resúmenes y la selección queda para la
Fase IV. La figura `reports/figures/03_segment_indicators.png` muestra el RMS y la curtosis
de cada segmento en estos cuatro casos, que corresponden a tres archivos distintos (los
rodamientos 3 y 4 del test 1 comparten archivo).

**Advertencia sobre la tabla:** 2 de los 3 archivos tienen muestras saturadas (±4.99) en el
canal mostrado: 6 en el canal 5 del test 1 (rodamiento 3) y 47 en el canal 3 del test 3; el
canal 1 del test 2 no tiene ninguna en esa captura. En esos casos los picos, el factor de
cresta y la curtosis están recortados, de modo que la tabla subestima la impulsividad real.

## Elección de la longitud de ventana (cuaderno 03, "Elección de la longitud de ventana")

El documento del proyecto pide ventanas "de longitud homogénea, definidas con base en
criterios de estabilidad estadística, contenido espectral y capacidad computacional". La
sección anterior cubre el contenido espectral; esta compara L = 1024, 2048, 4096, 10240 y
20480 muestras frente a los tres criterios.

**Muestra.** Para cada rodamiento con falla documentada (canal del eje x en el test 1, canal
único en los tests 2 y 3; capturas válidas del manifiesto) se toman las 20 primeras capturas
válidas (`inicio`), 20 capturas espaciadas uniformemente a lo largo del test (`a_lo_largo`,
que incluye ambos extremos) y las 20 últimas (`final`): 240 pares (captura, canal) de 174
archivos distintos, de los cuales 15 son archivos con saturación (3 en `a_lo_largo`, 12 en
`final`).

**Medidas, para cada L:** coeficiente de variación (CV, desviación estándar poblacional sobre
la media) del RMS y de la curtosis entre los segmentos contiguos y sin solapamiento de cada
captura (`segmentation.segment_indicators`), en mediana y percentil 90 por grupo (con
L = 20480 hay un solo segmento: n/a); resolución fs/L frente a la ventana de ±3% alrededor de
BPFO (14.18 Hz); y tiempo de pared mediano por captura (`time.perf_counter`) de
`segment_indicators` más un espectro de envolvente (`spectral.envelope_spectrum`, banda fija
2300-3800 Hz, solo para medir costo) por cada ventana de longitud L, extrapolado a las 46468
filas válidas del manifiesto.

| L | CV RMS mediana (inicio / final) | CV RMS p90 final | CV curtosis mediana (inicio / final) | CV curtosis p90 final | fs/L (Hz) | Líneas en ±3% BPFO | ms por captura | Ventanas por captura | Filas del conjunto si cada ventana fuera una fila | s del conjunto |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 5.6% / 4.8% | 7.7% | 10.5% / 16.9% | 42.4% | 19.531 | 0.73 | 15.59 | 20 | 929360 | 725 |
| 2048 | 3.5% / 3.0% | 6.1% | 7.6% / 10.4% | 38.9% | 9.766 | 1.45 | 9.42 | 10 | 464680 | 438 |
| 4096 | 2.2% / 1.9% | 3.8% | 5.2% / 6.8% | 23.3% | 4.883 | 2.90 | 6.35 | 5 | 232340 | 295 |
| 10240 | 0.7% / 0.8% | 1.6% | 2.2% / 2.8% | 12.3% | 1.953 | 7.26 | 4.49 | 2 | 92936 | 209 |
| 20480 | n/a | n/a | n/a | n/a | 0.977 | 14.52 | 3.89 | 1 | 46468 | 181 |

Los tiempos se midieron en una sola máquina y en un solo hilo; cambian algunos puntos
porcentuales entre ejecuciones, por lo que valen como órdenes de magnitud.

- **Estabilidad estadística.** El CV baja al crecer L porque cada estimación usa más
  muestras. Con L = 2048 el CV mediano del RMS es 3.5% al inicio y 3.0% al final, coherente
  con el 2-5.5% de la Fase II (`reports/eda_nasa_ims.md`, "Estabilidad de los segmentos"); el
  de la curtosis es 7.6% y 10.4%, con un percentil 90 final de 38.9%, y el mayor CV mediano
  final de la curtosis es el del rodamiento 3 del test 1 (30.0%), en línea con el 32% de la
  Fase II. El CV de la curtosis sube del inicio al final en todas las L con más de un
  segmento: parte de la variación es la actividad impulsiva del defecto, no ruido de
  estimación. Por sí solo, este criterio favorece ventanas más largas (con L = 10240 el CV
  final es 0.8% en RMS y 2.8% en curtosis), pero esas ventanas promedian la impulsividad y con
  2 segmentos ya no admiten resúmenes robustos como el percentil 90.
- **Contenido espectral.** Solo la captura completa da más de 10 líneas dentro de la ventana
  de ±3% de BPFO (14.52); L = 10240 da 7.26 y L = 2048, 1.45.
- **Capacidad computacional.** No discrimina: el costo por captura es de milisegundos y el
  conjunto completo se procesa en a lo sumo unos 12 min con cualquier L. El diseño "Mixto"
  (indicadores en 10 segmentos de 2048 más un espectro de envolvente de la captura completa)
  costó 3.87 ms por captura, unos 180 s para el conjunto, y produce una fila por captura con
  11 indicadores × 3 resúmenes de tiempo. Las ventanas cortas multiplican filas y valores
  (L = 1024 daría 929360 filas) sin ganancia espectral.

**Conclusión.** El diseño "Mixto" queda respaldado, con un matiz: L = 2048 no es la longitud
más estable (4096 y 10240 dan CV menores), sino el compromiso que mantiene el CV mediano del
RMS por debajo de 5% y conserva 10 segmentos para los resúmenes mediana, p90 y máximo que
capturan la impulsividad tardía. L = 4096 (5 segmentos) es una alternativa razonable para la
Fase IV si se prefiere estabilidad sobre detalle. Para el espectro, los datos favorecen
claramente la captura completa.

## Normalización y estandarización sin fuga (cuaderno 03, "Normalización y estandarización sin fuga")

Las capturas de un mismo test están fuertemente autocorrelacionadas: una partición aleatoria
por filas pondría capturas casi idénticas en entrenamiento y prueba. Los tres tests son las
unidades independientes, por lo que la validación usa **leave-one-test-out**
(`folds.leave_one_test_out`) sobre las filas válidas del manifiesto:

| Test fuera | Tests de entrenamiento | Filas de entrenamiento | Filas de prueba | Modos de falla en entrenamiento | Modos de falla en prueba |
|---|---|---|---|---|---|
| 1 | 2, 3 | 29220 | 17248 | pista externa | elemento rodante, pista interna |
| 2 | 1, 3 | 42540 | 3928 | elemento rodante, pista externa, pista interna | pista externa |
| 3 | 1, 2 | 21176 | 25292 | elemento rodante, pista externa, pista interna | pista externa |

Cuando el test 1 queda fuera, sus dos modos de falla (pista interna y elemento rodante) no
aparecen en el entrenamiento: esa partición evalúa el modelo sobre tipos de falla que nunca
vio, lo que debe considerarse al interpretar sus métricas.

**Reglas contra la fuga de información:**

1. Ningún test se reparte entre entrenamiento y prueba.
2. La estandarización (media y desviación estándar poblacional) se ajusta **solo con las
   filas de los tests de entrenamiento** de cada partición (`folds.standardize_by_fold`,
   `fit_standardizer`) y luego se aplica a todas las filas. Una columna con desviación cero
   en entrenamiento se centra sin reescalar.
3. Lo mismo vale para cualquier otro parámetro aprendido de los datos en las fases
   siguientes (umbrales, selección de características, tolerancias de búsqueda espectral).
   En particular, las bandas de envolvente de la Fase II se eligieron mirando capturas del
   final de la vida útil, y la tolerancia de ±3% alrededor de las frecuencias de falla salió
   de esa misma exploración: ambas deben fijarse a priori (antes de ver el test de prueba) o
   reajustarse dentro de cada partición solo con los tests de entrenamiento.
4. **Columnas del manifiesto que no son características.** Sirven solo para filtrar,
   particionar y etiquetar; la Fase IV y la Fase V **no** deben usarlas como entradas del
   modelo, porque filtran información del final de la vida útil o de la identidad:
   - `hours_to_end_clock`, `hours_to_end_operating`: son la distancia al final, de la que se
     deriva la etiqueta.
   - `clock_hours`, `operating_hours`: sumadas con las horas hasta el final dan una constante
     por test, y los tests tienen duraciones distintas, de modo que revelan cuánto falta.
   - `saturated_file`: solo es verdadera en archivos de la última hora (o menos) de cada test,
     cuya saturación está en los canales que fallan.
   - `after_documented_end`, `documented_failure`, `failure_mode`: describen el desenlace del
     test o del rodamiento.
   - `valid`, `post_shutdown`: marcas de limpieza para filtrar.
   - `test_id`, `bearing`, `channel`, `timestamp`: identidad y fecha (el rodamiento 3 falla en
     2 de los 3 tests, de modo que `bearing` correlaciona con la falla).

Demostración con RMS y curtosis de la tabla de indicadores: en cada partición las filas de
entrenamiento quedan con media ≈ 0 (|media| máxima 5.7e-16) y desviación estándar ≈ 1
(desvío máximo respecto de 1: 1.1e-16), mientras que el test fuera no:

| Test fuera | Característica | Media (entrenamiento) | Desv. est. (entrenamiento) | Media (prueba) | Desv. est. (prueba) |
|---|---|---|---|---|---|
| 1 | rms | 0.000 | 1.000 | 0.988 | 0.912 |
| 1 | kurtosis | 0.000 | 1.000 | 2.430 | 7.002 |
| 2 | rms | 0.000 | 1.000 | 0.437 | 1.460 |
| 2 | kurtosis | 0.000 | 1.000 | -0.013 | 0.416 |
| 3 | rms | 0.000 | 1.000 | -1.020 | 0.798 |
| 3 | kurtosis | 0.000 | 1.000 | -0.354 | 0.104 |

El caso más alejado es la curtosis con el test 1 fuera (media 2.43, desviación estándar
7.00). Esa diferencia es la que el modelo enfrenta con un test nunca visto; estandarizar con
todos los datos la ocultaría y daría una estimación optimista.

## Entregables para la Fase IV (cuaderno 03, "Cierre")

- **Manifiesto** `data/processed/ims_snapshot_manifest.csv` (46480 filas, 46468 válidas;
  generado con `make manifest`, no versionado), con las columnas `test_id`, `file`,
  `timestamp`, `channel`, `bearing`, `axis`, `clock_hours`, `operating_hours`,
  `post_shutdown`, `read_error`, `saturated_file`, `after_documented_end`, `valid`,
  `hours_to_end_clock`, `hours_to_end_operating`, `documented_failure` y `failure_mode`. Las
  características se generan solo para las filas con `valid` verdadero; ninguna de las
  columnas listadas en la regla 4 contra la fuga es una característica del modelo.
- **Segmentación** (`segmentation.py`): `remove_dc`, `split_segments`, `segment_indicators`
  y `summarize_segments` (mediana, p90, máximo); las características espectrales se calculan
  sobre la captura completa.
- **Particiones** (`folds.py`): `leave_one_test_out` (3 particiones) y
  `standardize_by_fold` / `fit_standardizer`, ajustadas solo con los tests de entrenamiento.

## Decisiones abiertas para la Fase V

1. La grilla de horizontes H del análisis de sensibilidad: 12, 24, 36, 48 y 72 h es un
   ejemplo del equipo, no una grilla fijada.
2. El eje temporal para definir H (horas de reloj u horas de operación). Los ejes difieren
   en los tests 1 y 3, sobre todo en el test 1 (58% de pausa).
3. El etiquetado del rodamiento 4 del test 1 (elemento rodante), cuya huella es débil y no
   monótona según la Fase II, y el de los rodamientos sin falla documentada, cuyo estado al
   final del test es desconocido.
4. Conservar las 1876 capturas del test 3 posteriores a su fin documentado, hoy conservadas
   y marcadas (`after_documented_end`); la última captura válida del test 3
   (2004.04.18.02.32.55) pertenece a ese tramo.
