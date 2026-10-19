# EDA — Conjunto de datos NASA IMS de rodamientos

## Resumen

Este informe documenta la Fase II (Data Understanding, CRISP-DM) del conjunto de datos
de vibraciones de rodamientos NASA IMS. Se verificó la integridad de los tres tests
crudos, se detectó y documentó una discrepancia relevante en el test 3 respecto al
`readme` de NASA, se identificaron capturas posteriores al apagado del banco de pruebas
que deben excluirse del análisis, y se exploraron indicadores de tiempo y frecuencia para
los cuatro rodamientos con falla documentada. El hallazgo más relevante para el
modelado: solo los dos defectos de pista externa (tests 2 y 3) dejan una huella clara y
dominante en el espectro de envolvente al final del test. Los dos defectos del test 1 son
más débiles: el de pista interna (rodamiento 3) crece solo 1.4× en la envolvente y su
subida final de curtosis depende de unas pocas muestras extremas; el de elemento rodante
(rodamiento 4) tiene un episodio de curtosis robusto entre 150 y 72 horas antes del final
que luego desaparece, y su amplitud en BSF crece solo ≈1.9× y sin un pico dominante. Todos los números de este informe provienen de
`notebooks/01_data_inventory.py` y `notebooks/02_signal_exploration.py` (notebooks
[marimo](https://marimo.io/), sin salidas almacenadas en el fuente). El cuaderno 01 lee
las tablas de `data/interim/` que genera `make indicators` a partir de los datos crudos
completos; el cuaderno 02 lee además capturas crudas, con el submuestreo que se declara en
cada sección. `make notebooks` los exporta a `build/notebooks/*.ipynb` (con salidas) y
`*.html` para su lectura sin necesidad de ejecutar marimo.

## Fuente y estructura de los datos

La copia usada proviene del espejo de Kaggle `vinayak123tyagi/bearing-dataset`, que
incluye el `readme` de NASA (`data/raw/Readme Document for IMS Bearing Data.pdf`). El
conjunto contiene tres tests de rodamientos hasta la falla ("run-to-failure"), cada uno
una carpeta de archivos de snapshot con nombre igual a su marca de tiempo de captura
(`YYYY.MM.DD.HH.MM.SS`), sin cabecera, separados por tabulador, con terminación de línea
CRLF y 20480 filas por archivo. En el espejo, el test 3 está en la carpeta
`3rd_test/4th_test/txt`. El test 1 registra 8 canales (2 por rodamiento: ejes x/y); los
tests 2 y 3 registran 4 canales (1 por rodamiento). El `readme` documenta 2000 RPM, una
carga radial de 6000 lbs, acelerómetros PCB 353B33 y una frecuencia de muestreo de 20 kHz,
y describe cada archivo como una captura de "1 segundo" de 20480 puntos, lo que es
internamente inconsistente (20480 / 20000 = 1.024 s). Se conserva 20 kHz (ver
"Desviación de las frecuencias características").

El `readme` no da la geometría del rodamiento. Para el Rexnord ZA-2115 se usan los
valores de Qiu et al. (2006), *Wavelet filter-based weak signature detection method and
its application on rolling element bearing prognostics*, Journal of Sound and Vibration
289: 16 rodillos, diámetro primitivo 2.815 in, diámetro del rodillo 0.331 in y ángulo de
contacto 15.17°. De ellos se derivan las frecuencias características de falla
(`ims.characteristic_fault_frequencies`, notebook 01, sección "Frecuencias
características"): FTF ≈ 14.8 Hz, BPFO ≈ 236.4 Hz, BPFI ≈ 296.9 Hz, BSF ≈ 139.9 Hz.

Fallas documentadas por NASA: test 1 — rodamiento 3 (pista interna) y rodamiento 4
(elemento rodante); test 2 — rodamiento 1 (pista externa); test 3 — rodamiento 3 (pista
externa). El código reutilizable de carga y metadatos vive en
`src/bearing_predictive_maintenance/ims.py` (registro de tests, mapa canal→rodamiento,
geometría), con indicadores de tiempo en `indicators.py` y de frecuencia en
`spectral.py`, todos con pruebas unitarias (`tests/test_ims.py`,
`tests/test_indicators.py`, `tests/test_spectral.py`). Los indicadores de amplitud (RMS,
pico, pico-a-pico, energía y factores de cresta, impulso y forma) se calculan sobre la
señal sin su media: los canales del test 1 tienen un offset DC de ≈−0.115 g, del mismo
orden que su vibración, que de otro modo ocultaría la mitad de la tendencia de
degradación. Con esta corrección el RMS coincide con la desviación estándar.

## Inventario e integridad (notebook 01)

Tabla documentado vs. observado (notebook 01, sección "Resumen por test"):

| Test | Archivos doc. | Archivos obs. | Inicio doc./obs. | Fin doc. | Fin obs. | Filas/archivo consistentes | Timestamps duplicados |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2156 | 2156 | coincide | 2003-11-25 23:39:56 | coincide | sí (100%) | 0 |
| 2 | 984 | 984 | coincide | 2004-02-19 06:22:39 | coincide | sí (100%) | 0 |
| 3 | 4448 | **6324** | coincide | 2004-04-04 19:01:57 | **2004-04-18 02:42:55** | sí (100%) | 0 |

En los tres tests, todos los archivos tienen 20480 filas, ninguno falla al leerse
(0 errores de lectura), ninguno contiene valores no numéricos o no finitos (0 valores no
finitos), todos tienen el número de columnas que documenta el mapa de canales y no hay
timestamps duplicados (columnas de integridad de la misma tabla): no hay evidencia de
archivos corruptos o mezclados entre tests.

**Saturación** (notebook 01, sección "Saturación del sistema de adquisición"): algunas
muestras alcanzan el límite de adquisición (±4.999), solo en las capturas finales de cada
test: 3 archivos del test 1 (9 muestras, desde las 23:13:21 del último día), 2 del test 2
(47 muestras, entre las 05:02:39 y las 05:42:39) y 4 del test 3 (89 muestras, desde las
01:32:55). En esas muestras la amplitud real es desconocida, y las capturas tardías que
usa el notebook 02 en los tests 1 y 3 están entre ellas.

**Longitud de las series** (notebook 01, sección "Longitud de las series"): el test 1 dura
827.6 horas de reloj pero solo 345.3 horas de operación (acumulando cada hueco hasta la
cadencia nominal de 600 s); el test 2, 163.8 h en ambos casos; el test 3, 1073.3 horas de
reloj y 1053.4 de operación.

**Cadencia de muestreo** (notebook 01, sección "Cadencia de muestreo"): la mediana de
huecos entre snapshots consecutivos es 600 s en los tres tests. El test 1 tiene huecos
grandes recurrentes (pausas de operación reales, no errores): el mayor es de ≈6.2 días
(534 333 s, entre el 2003-10-23 y el 2003-10-29), más otros de ≈4.0 y ≈3.8 días; 24
huecos superan 1.5× la mediana. El test 2 es perfectamente regular (ningún hueco supera
el umbral). El test 3 tiene 4 huecos por encima del umbral (el mayor ≈7.5 horas). La
cadencia tampoco es uniforme dentro de la operación del test 1: además de los primeros
≈43 archivos tomados cada 5 minutos que documenta el `readme`, hay una segunda ráfaga a
5 minutos el 2003-11-20, entre las 11:54 y las 20:39 (106 archivos, 105 huecos de 300 s).

**Discrepancia del test 3** (notebook 01, sección "Discrepancia del test 3"): el
directorio crudo contiene 6324 archivos, no los 4448 documentados en el `readme` de
NASA, y termina el 2004-04-18 02:42:55, **13.32 días** después del fin documentado
(2004-04-04 19:01:57). El inicio coincide exactamente, y **exactamente 4448 archivos**
caen en o antes del fin documentado. La cadencia y la ausencia de duplicados se mantienen
en todo el rango observado. La hipótesis más simple es que el `readme` describe una
versión truncada de la publicación; los datos no permiten confirmarla.

**Capturas posteriores al apagado** (notebook 01, sección "Capturas posteriores al
apagado", usando `spectral.flag_post_shutdown` con umbral = 10% de la mediana del RMS
máximo entre canales de los primeros 100 archivos): el test 1 no tiene ninguna; el test 2
tiene **2** (las dos últimas: 2004-02-19 06:12:39 y 06:22:39); el test 3 tiene **1**
(2004-04-18 02:42:55). Su RMS es casi nulo frente al de la captura anterior. Estas
capturas indican que el banco ya estaba apagado antes de terminar de grabar, y deben
excluirse de cualquier análisis de degradación o del cálculo de "última captura válida".

## Hallazgos del análisis exploratorio (notebook 02)

**Formas de onda y distribuciones de amplitud** (notebook 02, secciones "Formas de onda
crudas" y "Distribuciones de amplitud"): para los cuatro rodamientos con falla
documentada, la captura tardía (última válida, antes del apagado) muestra amplitudes
varias veces mayores que la captura temprana (primer snapshot del test), con picos
impulsivos visibles a simple vista en la forma de onda — consistente con el desarrollo de
un defecto de rodamiento.

**Estadísticos descriptivos** (notebook 02, sección "Estadísticos descriptivos por test y
rodamiento"): sobre las capturas válidas, la mediana del RMS va de 0.055 a 0.115 según el
rodamiento y la de la curtosis de 3.12 a 4.32, con rangos intercuartílicos estrechos
(p. ej. RMS 0.066-0.070 en el rodamiento 3 del test 3): la mayor parte de cada test es
estable, y la degradación ocupa una fracción pequeña de las capturas.

**Tendencias de degradación** (notebook 02, sección "Tendencias de degradación"): RMS,
curtosis y factor de cresta muestran una subida clara, tardía y concentrada en las
últimas horas, para los rodamientos 3 (test 1), 1 (test 2) y 3 (test 3). El
**rodamiento 4 del test 1 tiene un comportamiento distinto** (notebook 02, sección
"Curtosis por rodamiento al final del test 1"): entre 150 y 72 h antes de la última
captura válida, 190 de sus 381 capturas superan una curtosis de 8 (línea base ≈4.3), con
un máximo de 59.4 a ≈93 h del final dentro de ese tramo; en las últimas 72 h ninguna
captura supera 8. El rodamiento 3 hace lo contrario: 146 de sus 296 capturas de las
últimas 72 h superan 8 (máximo 74.6 a ≈64 h del final). Los rodamientos 1 y 2 nunca
superan 8. En la figura de tendencias el episodio del rodamiento 4 se ve atenuado porque se
promedian los dos ejes. Al quitar las 10 muestras más extremas de cada captura (de 20480),
el episodio del rodamiento 4 se mantiene en 108 de las 190 capturas, pero la subida final
del rodamiento 3 cae de 146 a 6: la curtosis tardía del rodamiento 3 la producen unos
pocos impactos aislados, y los máximos de 74.6 de ambos rodamientos son de ese tipo.

Para el test 1 se compararon dos ejes temporales, horas de reloj e índice secuencial de
snapshot, dado que este test tiene huecos de tiempo grandes (arriba): la forma de la
tendencia es similar en ambos, pero las horas de reloj comprimen visualmente el tramo
final. Ninguno equivale a tiempo de operación: las horas de reloj incluyen las pausas, y
el índice de snapshot cuenta igual las capturas tomadas cada 5 y cada 10 minutos. Se
recomienda construir **horas de operación** acumulando cada hueco solo hasta la cadencia
nominal (600 s).

**Correlación entre ejes y entre indicadores** (notebook 02, sección "Correlación entre
ejes x/y"): la correlación de Pearson entre el RMS del eje x y del eje y del test 1 es
alta para los cuatro rodamientos (0.93, 0.95, 0.99 y 0.99 para los rodamientos 1 a 4),
lo que indica una redundancia considerable entre ambos ejes para este indicador. En la
matriz de correlación entre los diez indicadores de tiempo (test 1):

- RMS, desviación estándar y energía son redundantes entre sí (|r| 0.90-1.00).
- Pico y pico-a-pico (|r| 0.99 entre sí) se parecen más a la familia de impulsividad
  (|r| 0.76-0.89 con curtosis, factor de cresta y factor de impulso) que al RMS
  (|r| 0.69-0.74).
- Curtosis, factor de cresta y factor de impulso forman un grupo (|r| 0.81-1.00).
- El factor de forma queda en un punto intermedio: |r| 0.57-0.64 con la familia de
  impulsividad y 0.33-0.36 con RMS y energía. La asimetría no se correlaciona con ningún
  otro indicador (|r| ≤ 0.16).

Estas correlaciones mezclan los cuatro rodamientos y todos los momentos del test, que
comparten una tendencia, lo que las infla; no miden la redundancia entre rodamientos del
mismo eje.

**Autocorrelación temporal** (notebook 02, sección "Autocorrelación temporal"): el RMS de
los rodamientos fallados conserva una autocorrelación de 0.95-0.995 entre capturas
consecutivas y de 0.45-0.85 a 144 capturas (≈24 h); la curtosis, de 0.52-0.69 entre
capturas consecutivas. Capturas vecinas de un mismo rodamiento son casi copias entre sí.

**Estabilidad de los segmentos** (notebook 02, sección "Estabilidad de los segmentos"):

- *Dentro de cada captura*, dividiendo las 20480 muestras en 10 segmentos de 2048
  (≈0.1 s), el RMS varía entre 2% y 5.5% entre segmentos (coeficiente de variación
  mediano), tanto en las 20 primeras como en las 20 últimas capturas válidas. La curtosis
  varía 7-10% al inicio, pero llega a 32% al final del test 1 (rodamiento 3) y a 15% al
  final del test 3, porque los impactos del defecto no aparecen en todos los segmentos.
- *En la línea base*, los tests 2 y 3 son estables: la mediana de RMS y curtosis cambia
  menos de 1.2% entre la primera y la segunda mitad del primer 20% de capturas válidas.
  El test 1 no: el RMS de los rodamientos 3 y 4 sube 20-23% dentro de ese tramo y la
  curtosis del rodamiento 4 sube 22%, lo que apunta a un periodo de asentamiento o deriva.

**Espectro y espectro de envolvente** (notebook 02, sección "Espectro y espectro de
envolvente"; la banda de filtrado para la envolvente se eligió de forma explícita y
reproducible, buscando dónde crece más la energía espectral cruda entre la captura
temprana y la tardía, no por referencia de literatura). Los picos se miden con
`spectral.fault_peak`, que busca el máximo dentro de ±3% de la frecuencia teórica:

| Rodamiento (fallo doc.) | Frecuencia teórica | Banda de envolvente | Pico observado (desviación) | ¿Domina la envolvente? |
| --- | --- | --- | --- | --- |
| Test 1, rodamiento 3 (pista interna) | BPFI ≈ 296.9 Hz | 4300-5800 Hz | 293.9 Hz (−1.0%) | **No**: los mayores picos están en 131.8 y 260.7 Hz |
| Test 1, rodamiento 4 (elemento rodante) | BSF ≈ 139.9 Hz | 3300-4800 Hz | 137.7 Hz (−1.6%), pequeño | **No**: domina 65.4 Hz, cercano a 2× la frecuencia del eje (66.7 Hz) |
| Test 2, rodamiento 1 (pista externa) | BPFO ≈ 236.4 Hz | 2300-3800 Hz | 231.4 Hz (−2.1%), armónico en 461.9 Hz | **Sí** |
| Test 3, rodamiento 3 (pista externa) | BPFO ≈ 236.4 Hz | 3800-5300 Hz | 230.5 Hz (−2.5%), armónico en 461.9 Hz | **Sí** |

En los dos casos de pista externa, el espectro crudo tardío está dominado por otras
frecuencias: 693 y 925 Hz en el test 2, y contenido de baja frecuencia (13.7 y 57.6 Hz)
seguido de 924 y 692 Hz en el test 3. El defecto solo se revela con claridad al demodular
(espectro de envolvente), el comportamiento esperado de un defecto que modula una
resonancia.

**Desviación de las frecuencias características.** Todas las líneas observadas caen 1-3%
por debajo de las frecuencias cinemáticas. El deslizamiento de la jaula no basta para
explicarlo: baja BPFO pero sube BPFI, porque BPFO + BPFI = 16 × frecuencia del eje. Con
20 kHz, las líneas de pista externa (−2.1% y −2.5%) son compatibles con deslizamiento,
pero la de pista interna (−1.0%) requiere además un eje ≈1% más lento que los 2000 RPM
nominales (con 1980 RPM el BPFI teórico es 293.96 Hz, frente a 293.9 Hz observado). Con
20.48 kHz, la otra lectura posible del `readme`, la línea de pista interna (+1.4%) sí
sería compatible con deslizamiento, pero las de pista externa quedarían sin él (+0.3%).
Las líneas del eje en los espectros son demasiado débiles para medir su velocidad, así
que los datos no resuelven la ambigüedad: se conserva el valor documentado, 20 kHz.

Los picos se buscan en una ventana relativa de ±3% alrededor de cada frecuencia teórica.
Esa tolerancia se eligió **después** de observar que el pico de pista externa del test 3
(230.5 Hz) quedaba fuera de la ventana fija de ±5 Hz que se usaba antes; es un parámetro
informado por los datos, y ese pico queda a solo 0.5% del borde de la nueva ventana.

**Evolución temporal de la amplitud de envolvente en la frecuencia documentada**
(notebook 02, sección "Evolución temporal..."; submuestreo declarado: test 1 cada 5
archivos, test 2 cada 3, test 3 cada 15, hasta la última captura válida). Comparando la
media del primer 20% de cada serie con la mediana y el percentil 90 de las últimas 100 h:

| Rodamiento | Línea base (media) | Últimas 100 h (mediana) | Razón mediana | Razón p90 |
| --- | --- | --- | --- | --- |
| Test 1, rodamiento 3 (BPFI) | 0.0036 | 0.0051 | 1.4× | 2.4× |
| Test 1, rodamiento 4 (BSF) | 0.0035 | 0.0065 | 1.9× | 2.9× |
| Test 2, rodamiento 1 (BPFO) | 0.0026 | 0.0151 | 5.7× | 19.2× |
| Test 3, rodamiento 3 (BPFO) | 0.0018 | 0.0047 | 2.6× | 50.8× |

Solo en los rodamientos de pista externa la amplitud permanece baja durante casi todo el
test y crece fuerte al final (en el test 3 la subida se concentra en las últimas horas,
de ahí la diferencia entre mediana y p90). En el test 1 ambos defectos crecen poco: la
pista interna 1.4× y el elemento rodante 1.9× en mediana. Junto con su episodio de
curtosis, esto indica que **el defecto de elemento rodante sí deja huella en ambos
dominios, aunque débil y no concentrada al final**, y no debería descartarse como "sin
señal" al diseñar características para el modelado. En el 7-19% de los puntos de cada
serie el máximo cae en el bin del borde de la ventana de ±3%: ahí el valor no es una línea
espectral sino el borde de la búsqueda, lo que añade ruido sobre todo al test 1 y al
test 3.

**Inicio exploratorio de degradación (onset)** (notebook 02, sección "Inicio exploratorio
de degradación"; detector `spectral.detect_onset`, explícitamente exploratorio, no una
regla de etiquetado final). El detector busca solo después de su ventana de línea base.
Estimaciones de horas antes de la última captura válida, con sensibilidad a los
parámetros:

| Rodamiento | n_sigma=2.5, persist=3 | n_sigma=3.0, persist=5 | n_sigma=4.0, persist=10 |
| --- | --- | --- | --- |
| Test 1, rodamiento 3 (BPFI) | 66.1 h | 34.2 h | sin detección |
| Test 1, rodamiento 4 (BSF) | 129.1 h | 112.4 h | 98.7 h |
| Test 2, rodamiento 1 (BPFO) | 74.5 h | 73.5 h | 69.5 h |
| Test 3, rodamiento 3 (BPFO) | 133.2 h | 41.2 h | 41.2 h |

En el test 1, la línea base del detector (el primer 20% de la serie) cae en el tramo
inicial que la sección de estabilidad muestra como no estable. La estimación varía
considerablemente según el umbral (de 41.2 a 133.2 horas para el rodamiento 3 del test 3,
según el parámetro), lo que confirma que este detector **no**
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
   usa horas de reloj (incluye inactividad), índice de snapshot (cadencia no uniforme) u
   horas de operación sin pausas.
4. **Conservar los 6324 archivos del test 3** como dato válido: no hay evidencia de
   corrupción, solo una discrepancia de documentación que se reporta, no se "corrige".
5. **No hay archivos corruptos ni timestamps duplicados** en ningún test: 0 errores de
   lectura, 0 valores no finitos, 20480 filas y el número de canales documentado en el
   100% de los archivos.

## Implicaciones para preparación y modelado

- **Variable objetivo**: el patrón observado apoya una etiqueta de tipo "falla dentro de
  una ventana de H horas", calculada hacia atrás desde la última captura válida de cada
  rodamiento con falla documentada. El horizonte H es una decisión de negocio/modelado,
  no solo estadística: las estimaciones de onset (arriba) dan un rango plausible pero
  varían con los parámetros del detector.
- **Eje temporal**: para el test 1 se recomiendan horas de operación sin pausas en vez de
  horas de reloj crudas o del índice de snapshot (ver "Tendencias de degradación").
- **Ventanas**: dentro de una captura, ventanas de 2048 muestras (≈0.1 s) dan un RMS
  estable (2-5.5% de variación); la curtosis en ventanas cortas será ruidosa en la etapa
  tardía (hasta 32%), lo que debe pesar al elegir la longitud de ventana en la Fase III.
- **Línea base**: la línea base del test 1 no es estable (RMS +20-23% dentro del primer
  20% del test). Un umbral calibrado solo con el inicio de ese test se dispararía antes de
  tiempo.
- **Partición sin fuga**: debe hacerse por test completo, nunca mezclando capturas del
  mismo rodamiento entre entrenamiento y validación/prueba, dado que la autocorrelación
  temporal de los indicadores es muy alta (0.95-0.995 entre capturas consecutivas para el
  RMS). Los cuatro rodamientos de un mismo test comparten eje, carga y fecha de
  adquisición, por lo que separar rodamientos del mismo test entre entrenamiento y prueba
  también puede filtrar información del banco; la opción más conservadora es una
  validación dejando un test fuera (*leave-one-test-out*). Con solo cuatro rodamientos
  con falla documentada en tres corridas independientes, la estimación del desempeño
  tendrá alta varianza y debe reportarse por test. Además, los modos de falla no se
  repiten entre tests: el test 1 tiene pista interna y elemento rodante, y los tests 2 y
  3, pista externa, así que cada partición evalúa un modo de falla que puede no estar en
  entrenamiento.
- **Parámetros elegidos con la serie completa**: las bandas de envolvente se derivaron de
  la captura de fin de vida, la tolerancia de ±3% de la búsqueda de picos se eligió tras
  ver el pico de pista externa, y los umbrales de onset y de apagado usan la propia serie.
  Si se reutilizan en la Fase III son fuga de la etiqueta: deben fijarse a priori o
  ajustarse solo con los datos de entrenamiento de cada partición.
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
  las últimas 24 h incluyen dos pausas de adquisición (≈10.8 h, que termina el
  2003-11-25 10:47:32, y ≈7.1 h, que termina el 2003-11-25 23:13:21), que suman ≈17.9 h.
  Medido en horas de operación, el mismo horizonte contiene 146 capturas (291 con 48 h y
  435 con 72 h); en el test 3, el horizonte de 72 h pasa de 390 a 433 capturas por su
  pausa de ≈7.5 h. El eje temporal elegido cambia directamente el desbalance.

  Además, estas proporciones son por rodamiento fallado. Si se entrena con todos los
  rodamientos (incluidos los que no fallan, cuyas capturas serían todas negativas), la
  proporción global de positivos será considerablemente menor. Por ejemplo, con H = 24 h
  el test 3 aporta 145 positivos frente a 6323 × 4 = 25 292 capturas entre sus cuatro
  rodamientos (≈0.57%). Esto obliga a usar métricas robustas al desbalance (sensibilidad,
  F1, PR-AUC) además de la exactitud.
- **Características prometedoras**: una variable de energía (RMS; desviación estándar y
  energía son redundantes con ella), una de impulsividad (curtosis o factor de cresta; el
  pico se agrupa con esta familia), preferiblemente en una versión robusta a muestras
  aisladas, y las amplitudes de envolvente en las frecuencias características, buscadas
  con la ventana relativa de ±3%. Las amplitudes de envolvente son la familia más
  directamente vinculada al mecanismo de falla, pero solo mostraron una huella clara en
  pista externa: no deben ser la única familia, porque los dos defectos del test 1 se
  verían poco. Conviene no incluir todas las variantes redundantes.

## Limitaciones y preguntas abiertas

1. ¿Qué horizonte H (24/48/72 h u otro) se adopta como variable objetivo, dado el
   desbalance de clases estimado arriba y su variación entre tests?
2. ¿Qué eje temporal se adopta para las ventanas del test 1? Este informe recomienda
   horas de operación sin pausas.
3. ¿Se conserva el test 3 completo (6324 archivos) para el modelado, dado que su fin
   documentado no coincide con el fin observado? Este informe recomienda conservarlo,
   pero es una decisión que debe confirmar el equipo.
4. El rodamiento 4 del test 1 (defecto de elemento rodante) tiene un episodio de
   curtosis entre 150 y 72 h antes del final que luego desaparece, y una huella débil en
   BSF: debe decidirse explícitamente cómo tratar este caso en el etiquetado y en el
   diseño de características, en vez de asumir que es un caso positivo "típico".
5. Los rodamientos sin falla documentada se etiquetarían como negativos en todo el test,
   pero el `readme` indica que todas las fallas ocurrieron después de superar la vida de
   diseño, así que su estado al final del test es desconocido. ¿Se aceptan como negativos
   o se excluyen sus últimas capturas?
6. La frecuencia de muestreo es ambigua (20 kHz vs. 20480 muestras por "1 segundo"). No
   cambia los indicadores de tiempo, pero sí la lectura de las frecuencias
   características, y ninguna de las dos lecturas explica todas las líneas solo con
   deslizamiento. La ventana relativa de ±3% absorbe la diferencia, pero fue elegida
   mirando los datos.
7. La elección de bandas de envolvente se hizo con una heurística simple y de un solo
   caso por rodamiento (temprano vs. tardío), mirando la captura de fin de vida; no se
   validó con más casos ni se contrastó contra literatura especializada. El detector de
   onset exploratorio tampoco se validó contra un método de referencia externo — es una
   primera lectura, no una regla validada de la disciplina.
8. El submuestreo usado en la evolución temporal (cada 5/3/15 archivos) reduce el costo
   de cómputo pero también la resolución temporal exacta del onset; una versión posterior
   podría usar el conjunto completo si el presupuesto de cómputo lo permite.
9. Las capturas finales de cada test tienen muestras saturadas (±4.999): sus picos y su
   curtosis están recortados, justo donde la degradación es mayor.
10. Este informe cubre solo NASA IMS. El análisis exploratorio del conjunto CWRU que
   menciona el documento del proyecto no tiene código ni resultados en este repositorio.
