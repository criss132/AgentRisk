# AgentRisk

Monitor de riesgo para agentes LLM multi-turno. Proyecto Final de Inteligencia Artificial (UTB), basado en **SemEval 2027, Tarea 10 (AgentRisk)**. Trabajo individual de Christian Menco.

Dado el historial de una interacción entre un usuario y un agente, y la acción que el agente propone ejecutar, el sistema debe estimar el riesgo de esa acción:

| Subtarea | Qué se predice | Métrica |
|---|---|---|
| S1 | Severidad del riesgo (1 a 5) | MAE (menor es mejor) |
| S2a | Dominio de daño (1 o 2 de 11 etiquetas) | Jaccard medio |
| S2b | Origen del riesgo (1 o 2 de 7 etiquetas) | Jaccard medio |
| S3 | Intervención guiada por política | F1, con políticas y casos propios (pendiente) |

**Estado: Hito 2 (avance).** Datos preparados, particiones por grupos, línea base L0 en S1, S2a y S2b, y línea base L1 en S1, S2a y S2b. El resto del plan (M1, M2 y S3) corresponde al Hito 3.

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
```

En Linux o macOS, la activación del entorno es `source .venv/bin/activate`.

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

## Líneas base y primeros resultados

Todo se entrena con `train` y se evalúa en `dev`. Semilla fija: 42.

| Modelo | S1 MAE | S2a Jaccard | S2b Jaccard | Subestimación severa* |
|---|---|---|---|---|
| L0: mediana y etiqueta más frecuente | 1.217 | 0.455 | 0.455 | 100 % |
| L1: TF-IDF y regresión logística | 0.762 | no evaluado | no evaluado | 45.9 % |
| L1 con `class_weight="balanced"` | **0.531** | **0.492** | **0.486** | **13.5 %** |

\* Porcentaje de ejemplos de `dev` con severidad real 4 o 5 que el modelo predice como 1 o 2 (37 ejemplos).

Cómo leerlo:

- L1 con pesos balanceados reduce el MAE un 56 % respecto a L0. Por tanto, el mejor baseline pasa a ser 0.531 y, según los umbrales del Hito 1, el modelo M1 deberá llegar a un MAE de unos 0.425 o menos.
- La configuración balanceada se eligió mirando `dev`, así que su resultado es algo optimista. Para el informe final la selección se hará con los folds y `dev` quedará como evaluación.
- `dev` es pequeño (143 ejemplos, 37 severos): diferencias de pocos ejemplos mueven las métricas varios puntos.

- En S2a y S2b, L1 mejora a L0 solo unas 0.03 a 0.04 (0.492 y 0.486 frente a 0.455). Con TF-IDF el texto distingue mucho peor los dominios y los orígenes del riesgo que la severidad. Para el modelo M1 el umbral del Hito 1 (+0.10) significa llegar a un Jaccard de unos 0.59.
- La regla de decisión de S2 (siempre la etiqueta más probable, y la segunda si su probabilidad es al menos 0.3) usa un umbral fijo sin ajustar. Con él el modelo predice casi 2 etiquetas por ejemplo, cuando en `dev` hay 1.2 (S2a) y 1.4 (S2b), así que el Jaccard actual probablemente se pueda mejorar eligiendo el umbral con los folds.

Resultados completos con hiperparámetros: `experiments/l0_resultados.json`, `experiments/l1_s1_resultados.json` y `experiments/l1_s2_resultados.json`.

## Estructura del repositorio

```
data/download_data.py      descarga de los datos desde Hugging Face
data/splits/               particiones por grupos (sin texto del dataset)
src/make_splits.py         genera los 5 folds
src/baseline_L0.py         línea base L0 (S1, S2a, S2b)
src/baseline_L1.py         línea base L1 (S1)
src/baseline_L1_s2.py      línea base L1 (S2a y S2b, multietiqueta)
notebooks/                 exploración de los datos
experiments/               resultados en JSON, con la semilla usada
docs/bitacora_ia.md        bitácora de uso de IA
```

## Pendiente para el Hito 3

- Elegir el umbral de la segunda etiqueta de S2 con los folds, en lugar de fijarlo en 0.3.
- M1: DeBERTa-v3 base multitarea; M2: modelo de lenguaje de pesos abiertos con pocos ejemplos.
- S3: políticas y casos propios.
- Selección de hiperparámetros con validación cruzada por grupos, análisis de errores, prototipo e informe.

## Uso de IA

Todo el uso de IA está registrado en [`docs/bitacora_ia.md`](docs/bitacora_ia.md). Por norma del curso, el código lo escribe el estudiante: la IA solo guía los pasos y revisa.
