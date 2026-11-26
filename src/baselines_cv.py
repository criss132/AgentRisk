"""Validación cruzada de 5 folds de L0 y L1 sobre train, con los mismos folds que M1.

Usa data/splits/train_folds.csv y guarda l0_cv_resultados.json y l1_cv_resultados.json.
"""
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# Constantes y rutas
SEMILLA = 42
UMBRAL_SEGUNDA_ETIQUETA = 0.3

TRAIN_PATH = "data/raw/data/train.jsonl"
SPLITS_PATH = "data/splits/train_folds.csv"
OUTPUT_DIR = "experiments"
L0_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "l0_cv_resultados.json")
L1_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "l1_cv_resultados.json")

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


def seleccionar_etiquetas(probs_fila, classes, umbral=UMBRAL_SEGUNDA_ETIQUETA):
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


def calcular_metricas_4(
    y_eval_s1_orig,
    pred_s1,
    y_eval_s2a_listas,
    preds_s2a,
    y_eval_s2b_listas,
    preds_s2b,
):
    """Calcula las mismas 4 métricas que M1: MAE S1, subestimación severa, Jaccard S2a y S2b."""
    mae_s1 = float(np.mean(np.abs(y_eval_s1_orig - pred_s1)))

    severe_mask = y_eval_s1_orig >= 4
    if np.sum(severe_mask) > 0:
        underestimation_pct = float(np.mean(pred_s1[severe_mask] <= 2) * 100.0)
    else:
        underestimation_pct = 0.0

    jaccard_s2a = float(np.mean([jaccard(r, p) for r, p in zip(y_eval_s2a_listas, preds_s2a)]))
    jaccard_s2b = float(np.mean([jaccard(r, p) for r, p in zip(y_eval_s2b_listas, preds_s2b)]))

    return {
        "s1_mae": round(mae_s1, 4),
        "severe_underestimation_pct": round(underestimation_pct, 2),
        "s2a_jaccard": round(jaccard_s2a, 4),
        "s2b_jaccard": round(jaccard_s2b, 4),
    }


def resumir_folds(per_fold_metrics: list[dict]) -> dict:
    """Calcula media y desviación estándar (ddof=1) para cada métrica a través de los folds."""
    metric_keys = ["s1_mae", "severe_underestimation_pct", "s2a_jaccard", "s2b_jaccard"]
    summary = {}
    for key in metric_keys:
        vals = np.array([m[key] for m in per_fold_metrics], dtype=np.float64)
        decimals = 2 if key == "severe_underestimation_pct" else 4
        summary[key] = {
            "mean": round(float(np.mean(vals)), decimals),
            "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, decimals),
        }
    return summary


def evaluar_l0_fold(train_fold_df: pd.DataFrame, val_fold_df: pd.DataFrame) -> dict:
    """Entrena L0 exclusivamente en train_fold_df y evalúa en val_fold_df."""
    l0_s1 = float(train_fold_df["risk_severity"].median())
    l0_s2a = train_fold_df["risk_domains"].explode().dropna().value_counts().index[0]
    l0_s2b = train_fold_df["risk_drivers"].explode().dropna().value_counts().index[0]

    n_val = len(val_fold_df)
    pred_s1 = np.full(n_val, l0_s1, dtype=np.float64)
    preds_s2a = [[l0_s2a] for _ in range(n_val)]
    preds_s2b = [[l0_s2b] for _ in range(n_val)]

    return calcular_metricas_4(
        y_eval_s1_orig=val_fold_df["risk_severity"].to_numpy(),
        pred_s1=pred_s1,
        y_eval_s2a_listas=val_fold_df["risk_domains"].tolist(),
        preds_s2a=preds_s2a,
        y_eval_s2b_listas=val_fold_df["risk_drivers"].tolist(),
        preds_s2b=preds_s2b,
    )


def evaluar_l1_fold(
    train_fold_df: pd.DataFrame,
    val_fold_df: pd.DataFrame,
    mlb_s2a: MultiLabelBinarizer,
    mlb_s2b: MultiLabelBinarizer,
    fold_idx: int,
) -> dict:
    """Entrena L1 (S1, S2a, S2b) en train_fold_df y evalúa en val_fold_df."""
    X_train = train_fold_df["formatted_interaction"]
    X_val = val_fold_df["formatted_interaction"]

    # S1: Pipeline TF-IDF + LogisticRegression balanceada (vocabulario solo de train_fold_df)
    pipe_s1 = Pipeline([
        ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
        ("clf", LogisticRegression(**PARAMS_ESTIMATOR)),
    ])
    pipe_s1.fit(X_train, train_fold_df["risk_severity"])
    pred_s1 = pipe_s1.predict(X_val)

    # S2a y S2b: MLB ajustado en todo train (columnas consistentes), pero matrices Y y TF-IDF solo del fold
    tareas_s2 = [
        ("S2a", "risk_domains", mlb_s2a),
        ("S2b", "risk_drivers", mlb_s2b),
    ]
    preds_por_tarea = {}

    for task_name, col, mlb in tareas_s2:
        Y_train_fold = mlb.transform(train_fold_df[col])
        clases = mlb.classes_

        # Diagnóstico útil si alguna clase rara no tiene positivos en este subconjunto de train
        ceros_en_train = [clases[i] for i in range(len(clases)) if Y_train_fold[:, i].sum() == 0]
        if ceros_en_train:
            print(f"  [Aviso Fold {fold_idx} | {task_name}] Clases sin positivos en train del fold: {ceros_en_train}")

        pipe_s2 = Pipeline([
            ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
            ("clf", OneVsRestClassifier(LogisticRegression(**PARAMS_ESTIMATOR))),
        ])
        pipe_s2.fit(X_train, Y_train_fold)

        probs_val = pipe_s2.predict_proba(X_val)
        preds_por_tarea[task_name] = [
            seleccionar_etiquetas(fila_probs, clases, UMBRAL_SEGUNDA_ETIQUETA)
            for fila_probs in probs_val
        ]

    return calcular_metricas_4(
        y_eval_s1_orig=val_fold_df["risk_severity"].to_numpy(),
        pred_s1=pred_s1,
        y_eval_s2a_listas=val_fold_df["risk_domains"].tolist(),
        preds_s2a=preds_por_tarea["S2a"],
        y_eval_s2b_listas=val_fold_df["risk_drivers"].tolist(),
        preds_s2b=preds_por_tarea["S2b"],
    )


def main():
    """Ejecuta la validación cruzada de L0 y L1 y guarda los resultados por fold y el resumen."""
    print("=== VALIDACIÓN CRUZADA (5 FOLDS) — BASELINES L0 Y L1 ===")

    # 1. Cargar train.jsonl y train_folds.csv y unir por example_id
    train_df = pd.read_json(TRAIN_PATH, lines=True)
    splits_df = pd.read_csv(SPLITS_PATH)

    merged_df = train_df.merge(splits_df[["example_id", "fold"]], on="example_id", how="inner")
    if len(merged_df) != len(train_df):
        print(f"[Aviso] Filas tras merge: {len(merged_df)} vs train original: {len(train_df)}")

    folds = sorted(merged_df["fold"].unique().tolist())
    print(f"Folds detectados: {folds} | Total ejemplos: {len(merged_df)}")

    # Ajustar MultiLabelBinarizer con todo train (igual que M1) para mantener mismas columnas entre folds
    mlb_s2a = MultiLabelBinarizer()
    mlb_s2a.fit(merged_df["risk_domains"])

    mlb_s2b = MultiLabelBinarizer()
    mlb_s2b.fit(merged_df["risk_drivers"])

    l0_per_fold = {}
    l1_per_fold = {}

    # 2. Entrenar en fold != N y evaluar en fold == N
    for fold_idx in folds:
        train_fold_df = merged_df[merged_df["fold"] != fold_idx].reset_index(drop=True)
        val_fold_df = merged_df[merged_df["fold"] == fold_idx].reset_index(drop=True)

        print(f"\n--- Fold {fold_idx} (train={len(train_fold_df)}, val={len(val_fold_df)}) ---")

        metrics_l0 = evaluar_l0_fold(train_fold_df, val_fold_df)
        metrics_l1 = evaluar_l1_fold(train_fold_df, val_fold_df, mlb_s2a, mlb_s2b, fold_idx)

        l0_per_fold[f"fold_{fold_idx}"] = metrics_l0
        l1_per_fold[f"fold_{fold_idx}"] = metrics_l1

        print(f"  L0 -> {metrics_l0}")
        print(f"  L1 -> {metrics_l1}")

    l0_summary = resumir_folds(list(l0_per_fold.values()))
    l1_summary = resumir_folds(list(l1_per_fold.values()))

    # 7. Guardar l0_cv_resultados.json y l1_cv_resultados.json en experiments/
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    l0_payload = {
        "model": "L0",
        "cv_folds": len(folds),
        "seed": SEMILLA,
        "date": fecha_str,
        "per_fold": l0_per_fold,
        "summary": l0_summary,
    }

    l1_payload = {
        "model": "L1",
        "cv_folds": len(folds),
        "seed": SEMILLA,
        "date": fecha_str,
        "hyperparameters": {
            "vectorizer": PARAMS_VECTORIZER,
            "classifier": {
                "wrapper": "OneVsRestClassifier",
                "estimator": "LogisticRegression",
                **PARAMS_ESTIMATOR,
            },
            "threshold_second_label": UMBRAL_SEGUNDA_ETIQUETA,
        },
        "per_fold": l1_per_fold,
        "summary": l1_summary,
    }

    with open(L0_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(l0_payload, f, indent=4, ensure_ascii=False)

    with open(L1_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(l1_payload, f, indent=4, ensure_ascii=False)

    print(f"\nGuardado: {L0_OUTPUT_FILE}")
    print(f"Guardado: {L1_OUTPUT_FILE}")

    print("\n=== RESUMEN 5-FOLD CV (MEDIA ± DEVIACIÓN) ===")
    for nombre, res in [("L0", l0_summary), ("L1", l1_summary)]:
        print(
            f"{nombre}: "
            f"MAE={res['s1_mae']['mean']:.4f}±{res['s1_mae']['std']:.4f} | "
            f"Subest.Severa={res['severe_underestimation_pct']['mean']:.2f}±{res['severe_underestimation_pct']['std']:.2f}% | "
            f"Jaccard_S2a={res['s2a_jaccard']['mean']:.4f}±{res['s2a_jaccard']['std']:.4f} | "
            f"Jaccard_S2b={res['s2b_jaccard']['mean']:.4f}±{res['s2b_jaccard']['std']:.4f}"
        )

if __name__ == "__main__":
    main()