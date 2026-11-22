# AgentRisk

Monitor de riesgo para agentes LLM multi-turno. Proyecto Final de Inteligencia Artificial (UTB), basado en **SemEval 2027, Tarea 10 (AgentRisk)**. Trabajo individual de Christian Menco.

Dado el historial de una interacción entre un usuario y un agente, y la acción que el agente propone ejecutar, el sistema debe estimar el riesgo de esa acción:

| Subtarea | Qué se predice | Métrica |
|---|---|---|
| S1 | Severidad del riesgo (1 a 5) | MAE (menor es mejor) |
| S2a | Dominio de daño (1 o 2 de 11 etiquetas) | Jaccard medio |
| S2b | Origen del riesgo (1 o 2 de 7 etiquetas) | Jaccard medio |
| S3 | Intervención guiada por política | F1, con políticas y casos propios (pendiente) |

**Estado: Hito 3 (en curso).** Datos, particiones por grupos, líneas base L0 y L1 y primer resultado de M1 (DeBERTa-v3 base multitarea) en `dev`, con una sola semilla. Faltan M2, S3, el ajuste de umbrales y semillas con los folds, el análisis de errores, el prototipo y el informe.

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

**M1 (requiere GPU).** Se entrenó en Kaggle con una GPU T4: clonar el repositorio, ejecutar `python data/download_data.py` y luego `python src/train_m1.py` (unos 6 minutos; la opción `--rapido` hace una prueba corta). Escribe `experiments/m1_resultados.json` y `experiments/m1_dev_predicciones.jsonl`. En Kaggle no se instala el `requirements.txt` fijado, porque reemplazaría las versiones de PyTorch del entorno.

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

Todo se entrena con `train` y se evalúa en `dev`. Semilla fija: 42.

| Modelo | S1 MAE | S2a Jaccard | S2b Jaccard | Subestimación severa* |
|---|---|---|---|---|
| L0: mediana y etiqueta más frecuente | 1.217 | 0.455 | 0.455 | 100 % |
| L1: TF-IDF y regresión logística | 0.762 | no evaluado | no evaluado | 45.9 % |
| L1 con `class_weight="balanced"` | 0.531 | 0.492 | 0.486 | **13.5 %** |
| M1: DeBERTa-v3 base multitarea (semilla 42) | **0.420** | **0.703** | **0.676** | 18.9 % |

\* Porcentaje de ejemplos de `dev` con severidad real 4 o 5 que el modelo predice como 1 o 2 (37 ejemplos).

Frente a los umbrales de éxito del Hito 1, tomando L1 balanceado como mejor baseline:

| Criterio | Umbral | M1 | ¿Cumple? |
|---|---|---|---|
| S1 MAE | al menos 20 % menos que 0.531 (0.425 o menos) | 0.420 (−21 %) | Sí, por poco |
| S2a Jaccard | +0.10 | +0.211 | Sí |
| S2b Jaccard | +0.10 | +0.190 | Sí |
| Subestimación severa | 15 % o menos | 18.9 % (7 de 37) | **No** |

Cómo leerlo:

- L1 con pesos balanceados reduce el MAE un 56 % respecto a L0. Por tanto, el mejor baseline pasa a ser 0.531 y, según los umbrales del Hito 1, el modelo M1 debía llegar a un MAE de 0.425 o menos.
- La configuración balanceada se eligió mirando `dev`, así que su resultado es algo optimista. Para el informe final la selección se hará con los folds y `dev` quedará como evaluación.
- `dev` es pequeño (143 ejemplos, 37 severos): diferencias de pocos ejemplos mueven las métricas varios puntos.

- En S2a y S2b, L1 mejora a L0 solo unas 0.03 a 0.04 (0.492 y 0.486 frente a 0.455). Con TF-IDF el texto distingue mucho peor los dominios y los orígenes del riesgo que la severidad.
- La regla de decisión de S2 (siempre la etiqueta más probable, y la segunda si su probabilidad es al menos 0.3) usa un umbral fijo sin ajustar. Con él el modelo predice casi 2 etiquetas por ejemplo, cuando en `dev` hay 1.2 (S2a) y 1.4 (S2b), así que el Jaccard actual probablemente se pueda mejorar eligiendo el umbral con los folds.

Sobre M1:

- Es una sola corrida (semilla 42). Se evalúa la última época (9), sin elegir nada con `dev`. Hiperparámetros: `max_len` 512 con truncamiento por la izquierda para conservar la acción propuesta, tasa de aprendizaje 2e-5, lote efectivo 16 y fp16. Tardó 372 s en una T4.
- La mejora en S2 es grande (+0.21 y +0.19), mucho más que la de L1 sobre L0. En S1 el margen es estrecho: el MAE pasa el umbral por 0.005 y con 37 casos severos un solo ejemplo mueve la subestimación 2.7 puntos.
- La subestimación severa no cumple el umbral: M1 acierta más en promedio, pero falla en 2 casos severos más que L1. Es la métrica que más importa para un monitor de riesgo.
- Con el umbral fijo de 0.3, M1 predice 1.0 (S2a) y 1.35 (S2b) etiquetas por ejemplo, cerca de las 1.2 y 1.39 reales de `dev`: la segunda etiqueta casi nunca supera 0.3 en S2a. El umbral no se ajustó todavía.
- Para reducir la subestimación sin mirar `dev`, el plan es una regla de decisión sensible al riesgo en S1 (predecir severidad 4 o más si la probabilidad acumulada de 4 y 5 supera un umbral), y elegir ese umbral y el de S2 con predicciones por folds. Además se repetirá con más semillas para reportar media y desviación.

Resultados completos con hiperparámetros: `experiments/l0_resultados.json`, `experiments/l1_s1_resultados.json`, `experiments/l1_s2_resultados.json` y `experiments/m1_resultados.json`. Las probabilidades de M1 en `dev` están en `experiments/m1_dev_predicciones.jsonl`.

## Estructura del repositorio

```
data/download_data.py      descarga de los datos desde Hugging Face
data/splits/               particiones por grupos (sin texto del dataset)
src/make_splits.py         genera los 5 folds
src/baseline_L0.py         línea base L0 (S1, S2a, S2b)
src/baseline_L1.py         línea base L1 (S1)
src/baseline_L1_s2.py      línea base L1 (S2a y S2b, multietiqueta)
src/train_m1.py            M1: DeBERTa-v3 base multitarea (GPU; se ejecutó en Kaggle)
notebooks/                 exploración de los datos
experiments/               resultados en JSON, con la semilla usada
docs/bitacora_ia.md        bitácora de uso de IA
```

## Pendiente para el Hito 3

- M1 con validación cruzada por folds y varias semillas (media y desviación), y elegir con ellos el umbral de la segunda etiqueta de S2 y la regla de decisión de S1 para bajar la subestimación severa.
- M2: modelo de lenguaje de pesos abiertos con pocos ejemplos (Ollama, local).
- S3: políticas y casos propios.
- Análisis de errores, prototipo e informe.

## Uso de IA

Todo el uso de IA está registrado en [`docs/bitacora_ia.md`](docs/bitacora_ia.md). Por norma del curso, el código lo escribe el estudiante: la IA solo guía los pasos y revisa.
