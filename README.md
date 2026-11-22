# AgentRisk

Monitor de riesgo para agentes LLM multi-turno. Proyecto Final de Inteligencia Artificial (UTB), basado en **SemEval 2027, Tarea 10 (AgentRisk)**. Trabajo individual de Christian Menco.

Dado el historial de una interacción entre un usuario y un agente, y la acción que el agente propone ejecutar, el sistema debe estimar el riesgo de esa acción:

| Subtarea | Qué se predice | Métrica |
|---|---|---|
| S1 | Severidad del riesgo (1 a 5) | MAE (menor es mejor) |
| S2a | Dominio de daño (1 o 2 de 11 etiquetas) | Jaccard medio |
| S2b | Origen del riesgo (1 o 2 de 7 etiquetas) | Jaccard medio |
| S3 | Intervención guiada por política | F1, con políticas y casos propios (pendiente) |

**Estado: Hito 3 (en curso).** Datos, particiones por grupos, líneas base L0 y L1 y M1 (DeBERTa-v3 base multitarea) evaluados en `dev` y con validación cruzada de 5 folds, y umbrales de decisión elegidos con los folds y confirmados en `dev`. Faltan M2, S3, el análisis de errores, las métricas por subgrupo, el prototipo y el informe.

## Cómo reproducir los resultados

Requiere Python 3.14 y Git. Probado en Windows (PowerShell) con Python 3.14.5. Desde una carpeta vacía:

```powershell
# 1. Clonar el repositorio
git clone https://github.com/criss132/AgentRisk.git; cd AgentRisk
# 2. Crear el entorno e instalar dependencias
python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt
# 3. Descargar los datos (obligatorio: los datos no están en el repositorio)
python data/download_data.py
# 4. Particiones y líneas base
python src/make_splits.py; python src/baseline_L0.py; python src/baseline_L1.py; python src/baseline_L1_s2.py
# 5. Líneas base con los mismos folds y selección de umbrales (usa las predicciones de M1 guardadas en experiments/)
python src/baselines_cv.py; python src/seleccionar_umbrales.py
```

En Linux o macOS, la activación del entorno es `source .venv/bin/activate`.

**M1 (requiere GPU).** Se entrenó en Kaggle con una GPU T4: clonar el repositorio, ejecutar `python data/download_data.py` y luego `python src/train_m1.py` (unos 6 minutos). Opciones: `--rapido` hace una prueba corta, `--fold N` entrena sin el fold N y predice sobre él, `--seed S` cambia la semilla y `--guardar` guarda los pesos en `experiments/checkpoints/`. Los resultados se escriben en `experiments/`. En Kaggle no se instala el `requirements.txt` fijado, porque reemplazaría las versiones de PyTorch del entorno.

Los scripts deben ejecutarse desde la raíz del repositorio. Cada uno imprime sus métricas y guarda un JSON en `experiments/`.

## Datos

- **Origen:** dataset `AgentRisk/agentrisk-data-2027` en Hugging Face ([enlace](https://huggingface.co/datasets/AgentRisk/agentrisk-data-2027)).
- **Licencia:** la ficha del dataset no declara licencia (revisada el 3 de noviembre de 2026). Por eso los datos **no se redistribuyen**: `data/raw/` está en `.gitignore` y se descarga con `data/download_data.py`.
- **Archivos usados:** `data/train.jsonl`, `data/dev.jsonl` y `metadata/split_manifest.jsonl`.
- **Tamaño:** 568 ejemplos de entrenamiento y 143 de desarrollo, agrupados en 397 escenarios (`scenario_group`: 302 en train y 95 en dev, sin ningún grupo compartido entre ambos).
- **Campos de cada ejemplo:** `example_id`, `formatted_interaction` (texto completo de la interacción, con una media de unas 490 palabras), `proposed_action`, `risk_severity`, `risk_domains` y `risk_drivers`.

Resumen de lo que mostró la exploración (`notebooks/01_exploracion.ipynb`):

- La severidad está desbalanceada: en train hay 265 ejemplos de severidad 1, 33 de 2, 127 de 3, 136 de 4 y solo 7 de 5.
- El 46 % de los ejemplos no tiene daño ni riesgo material (etiquetas "No Material Harm" y "No Material Risk"), así que la clase mayoritaria da un Jaccard cercano a 0.46.
- Hay dominios muy raros, como Discriminación/Equidad (1 ejemplo) y Medio ambiente (2).

## Particiones

`src/make_splits.py` reparte `train` en 5 folds con `StratifiedGroupKFold` (semilla 42), de modo que todos los ejemplos de un mismo escenario queden en el mismo fold y no haya fuga entre entrenamiento y validación. El resultado queda en `data/splits/train_folds.csv` (solo `example_id` y `fold`). Comprobaciones: ningún grupo aparece en más de un fold, los folds tienen entre 113 y 114 ejemplos y la distribución de severidad es parecida en todos.

## Resultados

Todo se entrena con `train` y se evalúa en `dev`. Semilla 42 en L0 y L1; M1 con las semillas 42, 43 y 44.

| Modelo | S1 MAE | S2a Jaccard | S2b Jaccard | Subestimación severa* |
|---|---|---|---|---|
| L0: mediana y etiqueta más frecuente | 1.217 | 0.455 | 0.455 | 100 % |
| L1: TF-IDF y regresión logística | 0.762 | no evaluado | no evaluado | 45.9 % |
| L1 con `class_weight="balanced"` | 0.531 | 0.492 | 0.486 | **13.5 %** |
| M1: DeBERTa-v3 base multitarea (media ± desv. de 3 semillas) | **0.378 ± 0.042** | **0.695 ± 0.008** | **0.679 ± 0.010** | 14.4 ± 4.1 % |

\* Porcentaje de ejemplos de `dev` con severidad real 4 o 5 que el modelo predice como 1 o 2 (37 ejemplos).

Frente a los umbrales de éxito del Hito 1, tomando L1 balanceado como mejor baseline (M1 con la media de 3 semillas):

| Criterio | Umbral | M1 | ¿Cumple? |
|---|---|---|---|
| S1 MAE | al menos 20 % menos que 0.531 (0.425 o menos) | 0.378 ± 0.042 (−29 %); peor semilla 0.420 | Sí, en las 3 semillas |
| S2a Jaccard | +0.10 | +0.203 | Sí |
| S2b Jaccard | +0.10 | +0.193 | Sí |
| Subestimación severa | 15 % o menos | 14.4 ± 4.1 % (18.9, 13.5 y 10.8 según la semilla) | En promedio sí; 1 de 3 semillas lo supera |

Cómo leerlo:

- L1 con pesos balanceados reduce el MAE un 56 % respecto a L0. Por tanto, el mejor baseline pasa a ser 0.531 y, según los umbrales del Hito 1, el modelo M1 debía llegar a un MAE de 0.425 o menos.
- La configuración balanceada se eligió mirando `dev`, así que su resultado es algo optimista. Para el informe final la selección se hará con los folds y `dev` quedará como evaluación.
- `dev` es pequeño (143 ejemplos, 37 severos): diferencias de pocos ejemplos mueven las métricas varios puntos.

- En S2a y S2b, L1 mejora a L0 solo unas 0.03 a 0.04 (0.492 y 0.486 frente a 0.455). Con TF-IDF el texto distingue mucho peor los dominios y los orígenes del riesgo que la severidad.
- La regla de decisión de S2 (siempre la etiqueta más probable, y la segunda si su probabilidad es al menos 0.3) con L1 predice casi 2 etiquetas por ejemplo, cuando en `dev` hay 1.2 (S2a) y 1.4 (S2b). Al ajustar el umbral con los folds, su Jaccard sube a 0.71 y 0.64 (ver «Validación cruzada y ajuste de umbrales»).

Sobre M1 (semillas 42, 43 y 44, última época, sin elegir nada con `dev`):

- Hiperparámetros: `max_len` 512 con truncamiento por la izquierda para conservar la acción propuesta, tasa de aprendizaje 2e-5, lote efectivo 16, 9 épocas y fp16. Cada entrenamiento tarda unos 350 s en una T4. Repetir la semilla 42 en la corrida de validación cruzada dio exactamente las mismas métricas que la primera vez.
- En S1 el MAE queda bajo el umbral en las 3 semillas (de 0.336 a 0.420), aunque esa dispersión es del mismo orden que el margen. La mejora en S2 es grande (+0.20 y +0.19) y estable entre semillas (desviación menor que 0.01).
- La subestimación severa es la métrica más inestable: 18.9 %, 13.5 % y 10.8 % según la semilla, con solo 37 casos severos en `dev` (cada caso son 2.7 puntos). Su media (14.4 %) queda bajo el 15 %, pero `dev` no basta para afirmarlo con seguridad.
- Validación cruzada de 5 folds (predicciones fuera de muestra sobre los 568 ejemplos de `train`, semilla 42): MAE 0.379, subestimación severa 13.3 % (19 de 143 casos severos), Jaccard 0.698 en S2a y 0.696 en S2b. Por fold, la subestimación va de 3.4 % a 21.4 %. Con 143 casos severos esta estimación es más estable que la de `dev`. Las líneas base aún no se evaluaron con los mismos folds.
- Con el umbral fijo de 0.3, M1 predice 1.0 (S2a) y 1.35 (S2b) etiquetas por ejemplo (semilla 42), cerca de las 1.2 y 1.39 reales de `dev`. Los umbrales se ajustaron después con los folds (sección siguiente).
- La subestimación severa se atacó después eligiendo con los folds una regla de decisión sensible al riesgo en S1 (predecir severidad 4 o más si la probabilidad acumulada de 4 y 5 supera un umbral); ver la sección siguiente.

Resultados completos con hiperparámetros: `experiments/l0_resultados.json`, `experiments/l1_s1_resultados.json` y `experiments/l1_s2_resultados.json`. De M1: `experiments/m1_resultados.json` (semilla 42), `m1_seed43_resultados.json`, `m1_seed44_resultados.json` y `m1_cv_fold{0-4}_seed42.json`; las probabilidades por ejemplo están en los `.jsonl` con el mismo nombre. Los pesos de M1 (semilla 42) están en `experiments/checkpoints/`, que no se sube a GitHub.

## Validación cruzada y ajuste de umbrales

L0 y L1 se evaluaron con los mismos 5 folds que M1 (`src/baselines_cv.py`). Media ± desviación entre folds, con predicciones fuera de muestra sobre los 568 ejemplos de `train` (semilla 42):

| Modelo | S1 MAE | S2a Jaccard | S2b Jaccard | Subestimación severa |
|---|---|---|---|---|
| L0 | 1.206 ± 0.013 | 0.461 ± 0.004 | 0.461 ± 0.004 | 100 % |
| L1 (balanceado) | 0.539 ± 0.024 | 0.465 ± 0.027 | 0.508 ± 0.030 | 16.2 ± 9.8 % |
| M1 | 0.379 ± 0.041 | 0.698 ± 0.027 | 0.696 ± 0.029 | 13.4 ± 6.9 % |

Cada fold tiene unos 28 casos severos, así que la subestimación severa varía mucho de un fold a otro y la diferencia entre M1 y L1 en esa métrica no es concluyente.

`src/seleccionar_umbrales.py` elige con esas predicciones fuera de muestra (sin mirar `dev`) la regla de decisión de cada modelo y la confirma en `dev`:

- **S1:** predecir severidad 4 o más si P(4) + P(5) supera un umbral tau. Se elige el tau de menor MAE con subestimación severa de 10 % o menos (criterio fijado antes de ver las curvas).
- **S2:** la etiqueta más probable y la segunda si su probabilidad supera un umbral, entre 0.10 y 0.60 (o ninguna segunda etiqueta). Se usa la media de vecindad de ventana 3 (2 en los extremos) para no escoger picos aislados.

Resultado en `dev` (M1: media ± desviación de 3 semillas; L1: una sola corrida):

| Modelo y regla | S1 MAE | S2a Jaccard | S2b Jaccard | Subestimación severa |
|---|---|---|---|---|
| M1, regla por defecto (argmax y 0.3) | 0.378 ± 0.042 | 0.695 ± 0.008 | 0.679 ± 0.010 | 14.4 ± 4.1 % |
| M1, umbrales elegidos (tau 0.225; S2a 0.40; S2b 0.25) | 0.345 ± 0.048 | 0.702 ± 0.005 | 0.681 ± 0.011 | 7.2 ± 3.1 % |
| L1, regla por defecto | 0.531 | 0.492 | 0.486 | 13.5 % |
| L1, umbrales elegidos (tau 0.30; S2a 0.55; S2b 0.50) | 0.755 | 0.706 | 0.642 | 5.4 % |

Cómo leerlo:

- La subestimación severa de M1 baja en las dos evaluaciones (13.3 % a 9.8 % en los folds, 14.4 % a 7.2 % en `dev`). Es el efecto sólido del ajuste.
- El MAE de M1 empeora en los folds (0.379 a 0.410) y mejora en `dev` (0.378 a 0.345). Los modelos de los folds usan el 80 % de `train` y los de `dev` el 100 %, así que sus probabilidades no son idénticas. El efecto del ajuste sobre el MAE es incierto.
- En S2, ajustar el umbral de M1 casi no cambia nada (+0.007 en S2a y +0.002 en S2b), menos que la desviación entre semillas. Al ampliar la cuadrícula a 0.10 y 0.15, el umbral de S2b pasó de 0.20 (extremo de la cuadrícula) a 0.25.
- L1 cambia mucho al ajustar S2: con 0.3 predecía casi 2 etiquetas por ejemplo y su Jaccard era 0.49; con 0.55 y 0.50 sube a 0.71 y 0.64. La línea base sin ajustar era injustamente débil. Con la misma regla de ajuste, M1 no supera a L1 en S2a (0.702 frente a 0.706) y lo supera por 0.039 en S2b, menos del +0.10 del Hito 1. El +0.20 de la tabla anterior se cumple solo frente al L1 con umbral fijo de 0.3, tal como se definió en el Hito 1.
- En S1 la ventaja se mantiene: con ambos modelos bajo el 10 % de subestimación, M1 tiene un MAE de 0.345 y L1 de 0.755.
- Límites: las métricas de los folds tras elegir umbrales son optimistas (se eligen y se miden en los mismos datos), así que la cifra honesta es la de `dev`, que tiene solo 143 ejemplos (37 severos). L1 es una sola corrida. El umbral de S2a de L1 (0.55) queda cerca del límite superior de la cuadrícula (0.60).

Resultados: `experiments/l0_cv_resultados.json`, `experiments/l1_cv_resultados.json` y `experiments/umbrales_resultados.json`.

## Estructura del repositorio

```
data/download_data.py      descarga de los datos desde Hugging Face
data/splits/               particiones por grupos (sin texto del dataset)
src/make_splits.py         genera los 5 folds
src/baseline_L0.py         línea base L0 (S1, S2a, S2b)
src/baseline_L1.py         línea base L1 (S1)
src/baseline_L1_s2.py      línea base L1 (S2a y S2b, multietiqueta)
src/train_m1.py            M1: DeBERTa-v3 base multitarea (GPU; se ejecutó en Kaggle)
src/baselines_cv.py         L0 y L1 evaluados con los mismos 5 folds
src/seleccionar_umbrales.py elige umbrales en los folds y los confirma en dev (M1 y L1)
notebooks/                 exploración de los datos
experiments/               resultados en JSON, con la semilla usada
experiments/checkpoints/   pesos de M1 (no se suben a GitHub)
docs/bitacora_ia.md        bitácora de uso de IA
```

## Pendiente para el Hito 3

- M2: modelo de lenguaje de pesos abiertos con pocos ejemplos (Ollama, local).
- S3: políticas y casos propios.
- Análisis de errores, métricas por subgrupo, prototipo e informe.

## Uso de IA

Todo el uso de IA está registrado en [`docs/bitacora_ia.md`](docs/bitacora_ia.md). Claude propuso pasos, explicó conceptos, revisó código y redactó la documentación a pedido del estudiante. El uso de otras herramientas de IA y la ayuda de terceros también se registran en esa bitácora.
