"""Línea base L1 para S2a y S2b: TF-IDF + regresión logística uno-contra-resto.

Se evalúa en dev con Jaccard y los resultados se guardan en experiments/l1_s2_resultados.json.
"""
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# 1. Constantes y rutas
SEMILLA = 42
UMBRAL_SEGUNDA_ETIQUETA = 0.3
L0_REF_JACCARD = 0.455

TRAIN_PATH = "data/raw/data/train.jsonl"
DEV_PATH = "data/raw/data/dev.jsonl"
OUTPUT_DIR = "experiments"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "l1_s2_resultados.json")

PARAMS_VECTORIZER = {
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "max_features": 10000,
    "min_df": 2,
}

PARAMS_ESTIMATOR = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": SEMILLA,
}

TAREAS = [
    {"task": "S2a", "column": "risk_domains"},
    {"task": "S2b", "column": "risk_drivers"},
]


def jaccard(real, pred):
    """Jaccard entre etiquetas reales y predichas (métrica de S2a y S2b)."""
    set_real = set(real)
    set_pred = set(pred)

    if len(set_real) == 0 and len(set_pred) == 0:
        return 1.0

    union = len(set_real.union(set_pred))
    if union == 0:
        return 0.0

    return len(set_real.intersection(set_pred)) / union


def seleccionar_etiquetas(probs_fila, classes, umbral):
    """
    Ordena probabilidades de mayor a menor, incluye siempre la top-1
    y añade la segunda solo si su probabilidad >= umbral.
    """
    indices_ordenados = np.argsort(probs_fila)[::-1]
    elegidos_idx = [indices_ordenados[0]]

    if len(indices_ordenados) > 1:
        segundo_idx = indices_ordenados[1]
        if probs_fila[segundo_idx] >= umbral:
            elegidos_idx.append(segundo_idx)

    return [classes[idx] for idx in elegidos_idx]


def main():
    """Entrena L1 para S2a y S2b en train, evalúa en dev y guarda los resultados."""
    print("=== EJECUCIÓN DEL MODELO BASE L1 (S2a y S2b: Multilabel) ===")

    # 1. Cargar datos
    train = pd.read_json(TRAIN_PATH, lines=True)
    dev = pd.read_json(DEV_PATH, lines=True)

    X_train = train["formatted_interaction"]
    X_dev = dev["formatted_interaction"]

    resultados_tareas = {}

    # 2. Bucle sobre risk_domains (S2a) y risk_drivers (S2b)
    for config_tarea in TAREAS:
        task_name = config_tarea["task"]
        col = config_tarea["column"]

        print(f"\n{'=' * 55}")
        print(f"Tarea {task_name} — Columna: {col}")
        print(f"{'=' * 55}")

        y_train_listas = train[col]
        y_dev_listas = dev[col]

        # Ajustar MultiLabelBinarizer solo con train
        mlb = MultiLabelBinarizer()
        Y_train = mlb.fit_transform(y_train_listas)
        clases = mlb.classes_

        # Diagnóstico rápido de positivos por clase en train
        positivos_por_clase = pd.Series(Y_train.sum(axis=0), index=clases).sort_values(ascending=False)
        print("\nPositivos en train por etiqueta:")
        print(positivos_por_clase.to_string())

        # Pipeline: TF-IDF -> OneVsRest(LogisticRegression)
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
            ("clf", OneVsRestClassifier(LogisticRegression(**PARAMS_ESTIMATOR))),
        ])

        pipeline.fit(X_train, Y_train)

        # 3. Probabilidades y regla de selección en dev
        probs_dev = pipeline.predict_proba(X_dev)
        preds_dev = [
            seleccionar_etiquetas(fila_probs, clases, UMBRAL_SEGUNDA_ETIQUETA)
            for fila_probs in probs_dev
        ]

        # Frecuencia de etiquetas predichas en dev
        conteo_predichas = pd.Series(preds_dev).explode().value_counts()
        print("\nFrecuencia de etiquetas predichas en dev:")
        print(conteo_predichas.to_string())

        # 4. Evaluar en dev comparando listas de texto reales vs predichas
        scores_jaccard = [
            jaccard(real_tags, pred_tags)
            for real_tags, pred_tags in zip(y_dev_listas, preds_dev)
        ]
        jaccard_medio = float(np.mean(scores_jaccard))
        diff_vs_l0 = jaccard_medio - L0_REF_JACCARD

        print(f"\nJaccard medio en dev ({task_name}): {jaccard_medio:.4f} (vs L0 ~{L0_REF_JACCARD:.3f}: {diff_vs_l0:+.4f})")

        resultados_tareas[task_name] = {
            "column": col,
            "jaccard_mean": round(jaccard_medio, 4),
            "diff_vs_l0_ref": round(diff_vs_l0, 4),
            "train_positives_per_class": {k: int(v) for k, v in positivos_por_clase.items()},
            "dev_predicted_counts": {k: int(v) for k, v in conteo_predichas.items()},
        }

    # 5. Guardar resultados en JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    salida = {
        "model": "L1",
        "tasks": ["S2a", "S2b"],
        "seed": SEMILLA,
        "threshold_second_label": UMBRAL_SEGUNDA_ETIQUETA,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": {
            "vectorizer": PARAMS_VECTORIZER,
            "classifier": {
                "wrapper": "OneVsRestClassifier",
                "estimator": "LogisticRegression",
                **PARAMS_ESTIMATOR,
            },
        },
        "results": resultados_tareas,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=4, ensure_ascii=False)

    print(f"\nResultados guardados en: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()