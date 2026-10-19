# EDA — Conjunto de datos NASA IMS de rodamientos

## Resumen

Este informe documenta la Fase II (Data Understanding, CRISP-DM) del conjunto de datos
de vibraciones de rodamientos NASA IMS. Se verificó la integridad de los tres tests
crudos, se detectó y documentó una discrepancia relevante en el test 3 respecto al
`readme` de NASA, se identificaron capturas posteriores al apagado del banco de pruebas
que deben excluirse del análisis, y se exploraron indicadores de tiempo y frecuencia para
los cuatro rodamientos con falla documentada. El hallazgo más relevante para el
modelado: los defectos de pista (interna y externa) dejan una huella clara tanto en
indicadores de tiempo (curtosis) como en el espectro de envolvente, mientras que el
defecto de elemento rodante del rodamiento 4 (test 1) **no** muestra una subida de
curtosis ni un pico limpio y dominante en su frecuencia característica (BSF), aunque sí
se observa un incremento real, más débil y ruidoso, de la amplitud de envolvente en esa
frecuencia. Todos los números de este informe provienen de
`notebooks/01_data_inventory.py` y `notebooks/02_signal_exploration.py` (notebooks
[marimo](https://marimo.io/), sin salidas almacenadas en el fuente), ejecutados de
principio a fin sobre los datos crudos completos; `make notebooks` los exporta a
`build/notebooks/*.ipynb` (con salidas) y `*.html` para su lectura sin necesidad de
ejecutar marimo.

## Fuente y estructura de los datos

El conjunto de datos NASA IMS contiene tres tests de rodamientos hasta la falla
("run-to-failure"), cada uno una carpeta de archivos de snapshot con nombre igual a su
marca de tiempo de captura (`YYYY.MM.DD.HH.MM.SS`), sin cabecera, separados por
tabulador, con terminación de línea CRLF y 20480 filas por archivo (1 segundo a 20 kHz).
El test 1 registra 8 canales (2 por rodamiento: ejes x/y); los tests 2 y 3 registran 4
canales (1 por rodamiento). La velocidad de eje documentada es 2000 RPM y la geometría
del rodamiento es Rexnord ZA-2115, de donde se derivan las frecuencias características de
falla (`ims.characteristic_fault_frequencies`, notebook 01, sección "Frecuencias
características"): FTF ≈ 14.8 Hz, BPFO ≈ 236.4 Hz, BPFI ≈ 296.9 Hz, BSF ≈ 139.9 Hz.

Fallas documentadas por NASA: test 1 — rodamiento 3 (pista interna) y rodamiento 4
(elemento rodante); test 2 — rodamiento 1 (pista externa); test 3 — rodamiento 3 (pista
externa). El código reutilizable de carga y metadatos vive en
`src/bearing_predictive_maintenance/ims.py` (registro de tests, mapa canal→rodamiento,
geometría), con indicadores de tiempo en `indicators.py` y de frecuencia en
`spectral.py`, todos con pruebas unitarias (`tests/test_ims.py`,
`tests/test_indicators.py`, `tests/test_spectral.py`).

## Inventario e integridad (notebook 01)

Tabla documentado vs. observado (notebook 01, sección "Resumen por test"):

| Test | Archivos doc. | Archivos obs. | Inicio doc./obs. | Fin doc. | Fin obs. | Filas/archivo consistentes | Timestamps duplicados |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2156 | 2156 | coincide | 2003-11-25 23:39:56 | coincide | sí (100%) | 0 |
| 2 | 984 | 984 | coincide | 2004-02-19 06:22:39 | coincide | sí (100%) | 0 |
| 3 | 4448 | **6324** | coincide | 2004-04-04 19:01:57 | **2004-04-18 02:42:55** | sí (100%) | 0 |

En todos los tests las 20480 filas por archivo son consistentes al 100% de los casos
observados y no hay timestamps duplicados: no hay evidencia de archivos corruptos o
mezclados entre tests.

**Cadencia de muestreo** (notebook 01, sección "Cadencia de muestreo"): la mediana de
huecos entre snapshots consecutivos es 600 s en los tres tests. El test 1 tiene huecos
grandes recurrentes (pausas de operación reales, no errores): el mayor es de ≈6.2 días
(534 333 s, alrededor del 2003-10-29), más otros de ≈4.0 y ≈3.8 días; 24 huecos superan
1.5× la mediana. El test 2 es perfectamente regular (ningún hueco supera el umbral). El
test 3 tiene 4 huecos por encima del umbral (el mayor ≈7.5 horas).

**Discrepancia del test 3** (notebook 01, sección "Discrepancia del test 3"): el
directorio crudo contiene 6324 archivos, no los 4448 documentados en el `readme` de
NASA, y termina el 2004-04-18 02:42:55, **13.32 días** después del fin documentado
(2004-04-04 19:01:57). El inicio coincide exactamente. La cadencia y la ausencia de
duplicados se mantienen consistentes en todo el rango observado, lo que descarta un
archivo corrupto o mal mezclado: el `readme` de NASA probablemente documenta una versión
truncada de la publicación del conjunto de datos.

**Capturas posteriores al apagado** (notebook 01, sección "Capturas posteriores al
apagado", usando `spectral.flag_post_shutdown` con umbral = 10% de la mediana del RMS
máximo entre canales de los primeros 100 archivos): el test 1 no tiene ninguna; el test 2
tiene **2** (las dos últimas: 2004-02-19 06:12:39 y 06:22:39, RMS máximo ≈0.004 y ≈0.002,
frente a ≈0.46-0.48 en la captura anterior); el test 3 tiene **1** (2004-04-18 02:42:55,
RMS máximo ≈0.004, frente a ≈0.76 en la captura anterior). Estas capturas casi planas
indican que el banco ya estaba apagado antes de terminar de grabar, y deben excluirse de
cualquier análisis de degradación o del cálculo de "última captura válida".

## Hallazgos del análisis exploratorio (notebook 02)

**Formas de onda y distribuciones de amplitud** (notebook 02, secciones "Formas de onda
crudas" y "Distribuciones de amplitud"): para los cuatro rodamientos con falla
documentada, la captura tardía (última válida, antes del apagado) muestra amplitudes
varias veces mayores que la captura temprana (primer snapshot del test), con picos
impulsivos visibles a simple vista en la forma de onda — consistente con el desarrollo de
un defecto de rodamiento.

**Tendencias de degradación** (notebook 02, sección "Tendencias de degradación"): RMS,
curtosis y factor de cresta muestran una subida clara, tardía y concentrada en las
últimas horas, para los rodamientos 3 (test 1), 1 (test 2) y 3 (test 3). El **rodamiento
4 del test 1 no muestra una subida de curtosis ni de factor de cresta comparable** a la
del rodamiento 3, replicando la observación de T3. Para el test 1, se comparó el eje de
horas de reloj transcurridas contra el índice secuencial de snapshot, dado que este test
tiene huecos de tiempo grandes (arriba): la forma de la tendencia es similar en ambos
ejes, pero las horas de reloj comprimen visualmente el tramo final (donde hay pausas
largas). Para cuantificar tiempo de operación acumulado se recomienda el índice
secuencial de snapshot (o una versión de "horas de operación" sin huecos), no las horas
de reloj crudas.

**Correlación entre ejes y entre indicadores** (notebook 02, sección "Correlación entre
ejes x/y"): la correlación de Pearson entre el RMS del eje x y del eje y del test 1 es
alta para los cuatro rodamientos (0.98, 0.98, 0.99 y 0.93 para los rodamientos 1 a 4
respectivamente), lo que indica una redundancia considerable entre ambos ejes para este
indicador. La matriz de correlación entre los diez indicadores de tiempo (test 1) muestra
dos grupos fuertemente correlacionados: uno de amplitud (RMS, desviación estándar, pico,
pico-a-pico, energía; |r| entre 0.63 y 0.97) y otro de impulsividad (curtosis, factor de
cresta, factor de impulso; |r| entre 0.82 y 1.00); la asimetría (skewness) no está
correlacionada con ningún otro indicador (|r| ≤ 0.22 en todos los casos).

**Espectro y espectro de envolvente** (notebook 02, sección "Espectro y espectro de
envolvente"; la banda de filtrado para la envolvente se eligió de forma explícita y
reproducible, buscando dónde crece más la energía espectral cruda entre la captura
temprana y la tardía, no por referencia de literatura):

| Rodamiento (fallo doc.) | Frecuencia objetivo | Banda de envolvente elegida | Pico limpio en la frecuencia objetivo |
| --- | --- | --- | --- |
| Test 1, rodamiento 3 (pista interna) | BPFI ≈ 296.9 Hz | 4300-5800 Hz | Parcial: actividad elevada cerca de BPFI, pero sin un pico único y dominante |
| Test 1, rodamiento 4 (elemento rodante) | BSF ≈ 139.9 Hz | 3300-4800 Hz | **No**: el pico dominante está cerca de 65 Hz, no en BSF ni en 2×BSF |
| Test 2, rodamiento 1 (pista externa) | BPFO ≈ 236.4 Hz | 2300-3800 Hz | **Sí**: pico limpio y dominante exactamente en BPFO |
| Test 3, rodamiento 3 (pista externa) | BPFO ≈ 236.4 Hz | 3800-5300 Hz | **Sí**: pico limpio y dominante exactamente en BPFO |

En los dos casos de pista externa, el espectro crudo tardío tiene picos dominantes que
**no** coinciden con BPFO (parecen resonancias estructurales entre 700-900 Hz): el
defecto solo se revela con claridad al demodular (espectro de envolvente), el
comportamiento textbook de por qué esta técnica es necesaria para el diagnóstico de
rodamientos.

**Evolución temporal de la amplitud de envolvente en la frecuencia documentada**
(notebook 02, sección "Evolución temporal..."; submuestreo declarado: test 1 cada 5
archivos, test 2 cada 3, test 3 cada 15, hasta la última captura válida): para los
rodamientos 1 (test 2) y 3 (test 3), la amplitud en BPFO permanece plana y baja durante
casi todo el test y crece abruptamente solo en las últimas horas. Para el rodamiento 3
del test 1 (BPFI), la subida es más gradual y menos limpia. **Para el rodamiento 4 del
test 1 (BSF), la amplitud de envolvente sí muestra una tendencia ascendente real** —de una
línea base ≈0.002-0.004 a valores ≈0.008-0.012 en las últimas ~100 horas, un aumento de
≈3-4×— aunque ruidosa, sin la subida limpia y monótona de los otros tres casos. Esto
significa que **el defecto de elemento rodante sí deja una huella medible, solo que más
débil y difusa que los defectos de pista**, y no debería descartarse como "sin señal" al
diseñar características para el modelado, pese a la ausencia de subida en curtosis.

**Inicio exploratorio de degradación (onset)** (notebook 02, sección "Inicio exploratorio
de degradación"; detector `spectral.detect_onset`, explícitamente exploratorio, no una
regla de etiquetado final). Estimaciones de horas antes del final válido, con
sensibilidad a los parámetros:

| Rodamiento | n_sigma=2.5, persist=3 | n_sigma=3.0, persist=5 | n_sigma=4.0, persist=10 |
| --- | --- | --- | --- |
| Test 1, rodamiento 3 (BPFI) | 66.1 h | 11.9 h | sin detección |
| Test 1, rodamiento 4 (BSF) | 129.1 h | 112.4 h | 98.7 h |
| Test 2, rodamiento 1 (BPFO) | 65.0 h | 65.0 h | 60.0 h |
| Test 3, rodamiento 3 (BPFO) | 42.5 h | 40.0 h | 40.0 h |

La estimación varía considerablemente según el umbral (de 11.9 a 66.1 horas para el
rodamiento 3 del test 1, según el parámetro), lo que confirma que este detector **no**
debe usarse como frontera exacta para una etiqueta de producción, solo como referencia
exploratoria de orden de magnitud.

## Criterios de inclusión/exclusión propuestos

1. **Excluir capturas posteriores al apagado**: las 2 últimas capturas del test 2 y la
   última del test 3 tienen amplitud casi nula y deben descartarse del cálculo de
   "última captura válida" y de cualquier ventana de entrenamiento.
2. **Tratar el fin documentado del test 3 como una versión truncada**, no como el momento
   de la falla: usar el fin observado (2004-04-18, menos la captura post-apagado) como
   límite temporal real de ese test, dado que no hay evidencia de corrupción y la
   degradación observada (RMS, curtosis, envolvente) crece precisamente hacia ese fin
   extendido.
3. **No tratar los huecos de tiempo del test 1 como errores**: son pausas de operación
   reales. Cualquier variable de "tiempo transcurrido" debe declarar explícitamente si
   usa horas de reloj (incluye inactividad) o índice secuencial de snapshot (solo
   operación).
4. **Conservar los 6324 archivos del test 3** como dato válido: no hay evidencia de
   corrupción, solo una discrepancia de documentación que se reporta, no se "corrige".
5. **No hay archivos corruptos ni timestamps duplicados** en ningún test; las 20480
   filas por archivo son consistentes en el 100% de los casos.

## Implicaciones para preparación y modelado

- **Variable objetivo**: el patrón observado apoya una etiqueta de tipo "falla dentro de
  una ventana de H horas", calculada hacia atrás desde la última captura válida de cada
  rodamiento con falla documentada. El horizonte H es una decisión de negocio/modelado,
  no solo estadística: las estimaciones de onset (arriba) dan un rango plausible pero
  varían con los parámetros del detector.
- **Ventanas**: para el test 1, se recomienda usar el índice secuencial de snapshot (o
  una versión de "horas de operación" sin huecos) en vez de horas de reloj crudas, dado
  que los huecos de tiempo distorsionan la lectura de cuánto tiempo de operación precede
  a la falla.
- **Partición sin fuga**: debe hacerse por rodamiento/test completo, nunca mezclando
  snapshots del mismo rodamiento fallado entre entrenamiento y validación/prueba, dado
  que los snapshots consecutivos están altamente correlacionados (ver correlación de
  indicadores). Los cuatro rodamientos de un mismo test comparten eje, carga y fecha de
  adquisición, por lo que separar rodamientos del mismo test entre entrenamiento y prueba
  también puede filtrar información del banco; la opción más conservadora es una
  validación dejando un test fuera (*leave-one-test-out*). Con solo cuatro rodamientos
  con falla documentada (dos en el test 1, uno en el test 2 y uno en el test 3), la
  estimación del desempeño tendrá alta varianza y debe reportarse por test.
- **Desbalance de clases estimado** (candidatos, no decisiones; notebook 02, sección de
  cierre), positivos = snapshots dentro de las últimas H horas del rango válido de cada
  rodamiento fallado:

  | Rodamiento | H=24h | H=48h | H=72h |
  | --- | --- | --- | --- |
  | Test 1, rodamiento 3 (BPFI) | 1.8% (39/2156) | 7.7% (165/2156) | 13.7% (296/2156) |
  | Test 1, rodamiento 4 (BSF) | 1.8% (39/2156) | 7.7% (165/2156) | 13.7% (296/2156) |
  | Test 2, rodamiento 1 (BPFO) | 14.8% (145/982) | 29.4% (289/982) | 44.1% (433/982) |
  | Test 3, rodamiento 3 (BPFO) | 2.3% (145/6323) | 4.6% (289/6323) | 6.2% (390/6323) |

  El desbalance varía mucho entre tests según su duración total y su cadencia: el test 2
  es mucho más corto (982 snapshots válidos) que el test 3 (6323), por lo que la misma
  ventana de 72 horas representa proporciones de positivos muy distintas (44.1% vs 6.2%).
  Esto es relevante si se entrena un modelo combinando los tres tests.

  En el test 1, el horizonte de 24 h en horas de reloj contiene solo 39 capturas porque
  las últimas 13 h incluyen dos pausas de adquisición (≈10.8 h, que termina el
  2003-11-25 10:47:32, y ≈7.1 h, que termina el 2003-11-25 23:13:21). Un horizonte
  expresado en horas de operación o en número de capturas daría proporciones distintas.

  Además, estas proporciones son por rodamiento fallado. Si se entrena con todos los
  rodamientos (incluidos los que no fallan, cuyas capturas serían todas negativas), la
  proporción global de positivos será considerablemente menor. Por ejemplo, con H = 24 h
  el test 3 aporta 145 positivos frente a 6323 × 4 = 25 292 capturas por rodamiento
  (≈0.6 %). Esto obliga a usar métricas robustas al desbalance (sensibilidad, F1,
  PR-AUC) además de la exactitud.
- **Características prometedoras**: dado el fuerte solapamiento entre RMS/pico/energía
  (redundantes entre sí) y entre curtosis/factor de cresta/factor de impulso (redundantes
  entre sí), se recomienda no incluir todas las variantes de cada grupo como
  características independientes. Las amplitudes de envolvente en las frecuencias
  características documentadas son la familia más directamente vinculada al mecanismo de
  falla y deberían priorizarse junto con curtosis/factor de cresta.

## Limitaciones y preguntas abiertas

1. ¿Qué horizonte H (24/48/72 h u otro) se adopta como variable objetivo, dado el
   desbalance de clases estimado arriba y su variación entre tests?
2. ¿Se usa el índice secuencial de snapshot o una versión de "horas de operación" sin
   huecos como eje temporal de referencia para las ventanas del test 1?
3. ¿Se conserva el test 3 completo (6324 archivos) para el modelado, dado que su fin
   documentado no coincide con el fin observado? Este informe recomienda conservarlo,
   pero es una decisión que debe confirmar el equipo.
4. El rodamiento 4 del test 1 (defecto de elemento rodante) no muestra una subida de
   curtosis ni un pico limpio en su frecuencia característica, aunque sí una tendencia de
   amplitud de envolvente real y más débil: debe decidirse explícitamente cómo tratar
   este caso en el etiquetado y en el diseño de características, en vez de asumir que es
   un caso positivo "típico".
5. La elección de bandas de envolvente (tabla en "Hallazgos") se hizo con una heurística
   simple y de un solo caso por rodamiento (temprano vs. tardío); no se validó con más
   casos ni se contrastó contra literatura especializada en diagnóstico de rodamientos
   (por verificar, si se desea reforzar la elección de banda con referencias externas).
   El detector de onset exploratorio tampoco se validó contra un método de referencia
   externo — es una primera lectura, no una regla validada de la disciplina.
6. El submuestreo usado en la evolución temporal (cada 5/3/15 archivos) reduce el costo
   de cómputo pero también la resolución temporal exacta del onset; una versión posterior
   podría usar el conjunto completo si el presupuesto de cómputo lo permite.
