# Revisión del código (25 de noviembre de 2026)

Revisión hecha por Claude a pedido de Christian sobre los 16 scripts de `data/` y `src/`. **No se cambió ninguna lógica**: solo se añadieron docstrings (módulo, funciones y clases). Los hallazgos de abajo quedan para que Christian decida si los aplica.

## Verificación de lo que sí se cambió

- Los 16 archivos compilan (`python -m py_compile`).
- El árbol sintáctico (AST) de cada archivo, sin contar docstrings, es idéntico antes y después.
- `git diff --ignore-cr-at-eol`: 91 líneas añadidas y 0 eliminadas. Se conservaron los finales de línea CRLF.
- `pyflakes` no encontró errores de ejecución, solo los avisos de la sección siguiente.

## Hallazgos que conviene corregir

| # | Dónde | Qué pasa | Sugerencia |
|---|---|---|---|
| 1 | `prototipo.py`, `cargar_con_respaldo` | Si falla la carga de M1 por cualquier motivo (por ejemplo, faltan los pesos), pasa a L1 y reentrena al arrancar. Solo lo avisa una línea de texto y `Modelo activo : L1_fallback`. | Antes de grabar el video o probar el clon limpio, comprobar que dice `Modelo activo : M1`. |
| 2 | `prototipo.py`, `torch.load` | Carga el archivo de pesos sin `weights_only=True`, que evita ejecutar código al cargar un archivo ajeno. | Añadir `weights_only=True` (los pesos son solo tensores). |
| 3 | `baseline_L0.py:5`, `baseline_L1.py:5`, `m2_s3.py:40`, `prototipo.py:6` | Imports sin usar (`numpy`, `INTERVENCIONES`, `datetime`). | Quitarlos. |
| 4 | `baseline_L0.py:51` | f-string sin variables. | Quitar la `f`. |
| 5 | `jaccard` (6 archivos) | La rama `if union == 0: return 0.0` es inalcanzable, porque el caso de dos listas vacías ya devuelve 1.0. | Quitarla al unificar la función. |
| 6 | `make_splits.py` | `import os` dentro de `main`; el `merge` interno podría descartar ejemplos sin avisar; el texto "ideal cerca de 114 por fold" está fijo. | Mover el import arriba y comprobar que quedan 568 filas. |
| 7 | `download_data.py` | No maneja errores de red y no imprime qué descarga. | Añadir un mensaje por archivo. |
| 8 | `m2_s3.py` | No guarda la respuesta cruda del LLM, solo la intervención traducida. Por eso no se pudo diagnosticar la correlación negativa con la referencia. | Guardar el campo `response` en el JSONL. |
| 9 | `train_m1.py`, `set_seed` | No fija el determinismo de cuDNN, así que dos corridas con la misma semilla en GPU pueden diferir un poco. | Declararlo en el informe (ya se dice que las cifras no son bit a bit idénticas) o fijar `torch.backends.cudnn.deterministic = True`. |

## Optimizaciones posibles

- **Código duplicado.** `jaccard`, `seleccionar_etiquetas`, `PARAMS_VECTORIZER`, `PARAMS_ESTIMATOR` y las rutas se repiten en varios archivos. Un `src/comun.py` los unificaría.
- **Prototipo y entrenamiento acoplados.** `prototipo.py` importa `train_m1.py`, que fija `CUDA_VISIBLE_DEVICES=0` al importarse y trae dependencias de entrenamiento. Mover `DebertaMultiTask` y `seleccionar_etiquetas` a un módulo común lo evitaría.
- **Tokenización.** `MultiTaskDataset` tokeniza cada texto en cada época. Tokenizar una vez antes de entrenar ahorra CPU; el relleno a 512 casi no desperdicia nada porque 142 de 143 textos superan ese largo.
- **Respaldo L1.** Reentrena en cada arranque. Se puede guardar con `joblib` y reutilizar.
- **Inferencia.** `torch.inference_mode()` en lugar de `torch.no_grad()` usa algo menos de memoria.
- **Dependencias.** `requirements.txt` incluye paquetes de desarrollo (`ipykernel`, `debugpy`, `jedi`). Si se añade Gradio, fijar su versión.

## Dudas resueltas

- **Jaccard de L0 igual en S2a y S2b (0.455).** Se explica por los datos: las etiquetas más frecuentes (`No Material Harm` y `No Material Risk`) aparecen en exactamente los 65 casos seguros de dev, y predecir solo esa etiqueta da Jaccard 1 en esos casos y 0 en el resto (65/143 = 0.4545).
- **0 respuestas inválidas de M2** en S1 y S2 (zero-shot y few-shot) y en S3 (A y B): sus malos resultados no vienen del *fallback* de severidad 2.
