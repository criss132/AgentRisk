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
SEEDS_DEV = [42, 43, 44]
UMBRAL_DEFAULT_S2 = 0.30

TRAIN_PATH = "data/raw/data/train.jsonl"
DEV_PATH = "data/raw/data/dev.jsonl"
SPLITS_PATH = "data/splits/train_folds.csv"
EXPERIMENTS_DIR = "experiments"
OUTPUT_FILE = os.path.join(EXPERIMENTS_DIR, "umbrales_resultados.json")

# Cuadrícula ampliada hacia abajo (0.10, 0.15) para detectar si el óptimo está por debajo de 0.20
GRID_UMBRALES_S2 = [None, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# Cuadrícula de tau con pasos de 0.025 redondeada a 3 decimales
GRID_TAU_S1 = [round(float(x), 3) for x in np.linspace(0.15, 0.85, 29)]

# Criterio declarado antes de mirar la curva
CRITERIO_S1 = "menor_mae_con_subestimacion_severa_menor_o_igual_10pct"
MAX_SUBESTIMACION_PERMITIDA = 10.0

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
    """Calcula el índice de Jaccard entre dos listas de etiquetas como conjuntos."""
    set_real = set(real)
    set_pred = set(pred)
    if len(set_real) == 0 and len(set_pred) == 0:
        return 1.0
    union = len(set_real.union(set_pred))
    if union == 0:
        return 0.0
    return len(set_real.intersection(set_pred)) / union


def normalizar_probs_s2(probs_raw, classes_ordenadas: list[str]) -> dict[str, float]:
    """Acepta probs_s2 como dict {etiqueta: prob} o como lista alineada con classes_ordenadas."""
    if isinstance(probs_raw, dict):
        return {str(k): float(v) for k, v in probs_raw.items()}
    if isinstance(probs_raw, (list, np.ndarray)):
        if len(probs_raw) != len(classes_ordenadas):
            raise ValueError(
                f"Longitud de lista de probabilidades ({len(probs_raw)}) no coincide con clases ({len(classes_ordenadas)})."
            )
        return {str(cls): float(p) for cls, p in zip(classes_ordenadas, probs_raw)}
    raise TypeError(f"Formato no soportado para probabilidades S2: {type(probs_raw)}")


def seleccionar_etiquetas_con_umbral(probs_raw, umbral: float | None, classes_ordenadas: list[str]) -> list[str]:
    """
    Ordena las etiquetas por probabilidad descendente.
    Incluye siempre la top-1 y añade la segunda solo si umbral is not None y prob_2 >= umbral.
    """
    probs_dict = normalizar_probs_s2(probs_raw, classes_ordenadas)
    orden = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)
    elegidas = [orden[0][0]]
    if umbral is not None and len(orden) > 1:
        segunda_etiqueta, segunda_prob = orden[1]
        if segunda_prob >= umbral:
            elegidas.append(segunda_etiqueta)
    return elegidas


def predecir_s1_argmax(probs_s1: list[float] | np.ndarray) -> int:
    """Regla por defecto para S1: argmax en escala 1..5."""
    p = np.asarray(probs_s1, dtype=np.float64)
    return int(np.argmax(p) + 1)


def predecir_s1_con_tau(probs_s1: list[float] | np.ndarray, tau: float) -> int:
    """
    probs_s1 corresponde a severidades [1, 2, 3, 4, 5] en índices [0, 1, 2, 3, 4].
    Si P(sev 4) + P(sev 5) > tau, predice al menos 4 (elige entre 4 y 5 según argmax);
    en caso contrario, predice el argmax entre las severidades 1..3.
    """
    p = np.asarray(probs_s1, dtype=np.float64)
    p_alta_sev = float(p[3] + p[4])
    if p_alta_sev > tau:
        return 4 if p[3] >= p[4] else 5
    return int(np.argmax(p[:3]) + 1)


def calcular_metricas_s1(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_real - y_pred)))
    severe_mask = y_real >= 4
    if np.sum(severe_mask) > 0:
        subest_pct = float(np.mean(y_pred[severe_mask] <= 2) * 100.0)
    else:
        subest_pct = 0.0
    return {
        "s1_mae": round(mae, 4),
        "severe_underestimation_pct": round(subest_pct, 2),
    }


def evaluar_predicciones_df(
    df: pd.DataFrame,
    classes_s2a: list[str],
    classes_s2b: list[str],
    umbral_s2a: float | None,
    umbral_s2b: float | None,
    tau_s1: float | None = None,
) -> dict:
    """
    Evalúa un DataFrame que contiene risk_severity, risk_domains, risk_drivers,
    probs_s1, probs_s2a y probs_s2b. Si tau_s1 is None, usa argmax en S1.
    """
    y_real_s1 = df["risk_severity"].to_numpy()
    if tau_s1 is None:
        y_pred_s1 = np.array([predecir_s1_argmax(p) for p in df["probs_s1"]], dtype=np.int64)
    else:
        y_pred_s1 = np.array([predecir_s1_con_tau(p, tau_s1) for p in df["probs_s1"]], dtype=np.int64)

    m_s1 = calcular_metricas_s1(y_real_s1, y_pred_s1)

    preds_s2a = [
        seleccionar_etiquetas_con_umbral(p, umbral_s2a, classes_s2a)
        for p in df["probs_s2a"]
    ]
    preds_s2b = [
        seleccionar_etiquetas_con_umbral(p, umbral_s2b, classes_s2b)
        for p in df["probs_s2b"]
    ]

    j_s2a = float(np.mean([jaccard(r, p) for r, p in zip(df["risk_domains"], preds_s2a)]))
    j_s2b = float(np.mean([jaccard(r, p) for r, p in zip(df["risk_drivers"], preds_s2b)]))

    return {
        "s1_mae": m_s1["s1_mae"],
        "severe_underestimation_pct": m_s1["severe_underestimation_pct"],
        "s2a_jaccard": round(j_s2a, 4),
        "s2b_jaccard": round(j_s2b, 4),
    }


def cargar_m1_oof_cv(train_df: pd.DataFrame, splits_df: pd.DataFrame, seed: int = SEMILLA) -> pd.DataFrame:
    """Une las predicciones OOF de M1 en los 5 folds con las etiquetas reales de train."""
    records = []
    for fold_idx in range(5):
        path = os.path.join(EXPERIMENTS_DIR, f"m1_cv_fold{fold_idx}_seed{seed}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo OOF de M1: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records.append(rec)

    oof_df = pd.DataFrame(records)
    merged = train_df.merge(splits_df[["example_id", "fold"]], on="example_id", how="inner")
    merged = merged.merge(oof_df[["example_id", "probs_s1", "probs_s2a", "probs_s2b"]], on="example_id", how="inner")
    return merged


def cargar_m1_dev_semilla(dev_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Une las predicciones de M1 en dev para una semilla dada con las etiquetas reales de dev."""
    if seed == 42:
        path = os.path.join(EXPERIMENTS_DIR, "m1_dev_predicciones.jsonl")
    else:
        path = os.path.join(EXPERIMENTS_DIR, f"m1_seed{seed}_dev_predicciones.jsonl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de predicciones de M1 en dev: {path}")

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    preds_df = pd.DataFrame(records)
    return dev_df.merge(preds_df[["example_id", "probs_s1", "probs_s2a", "probs_s2b"]], on="example_id", how="inner")


def generar_l1_oof_y_dev(
    merged_train_folds: pd.DataFrame,
    dev_df: pd.DataFrame,
    classes_s1_orig: list[int],
    classes_s2a: list[str],
    classes_s2b: list[str],
    seed: int = SEMILLA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entrena L1 (corrida única con `seed`) en cada fold para obtener OOF en los 568 ejemplos
    y en todo train para evaluar en dev, devolviendo probabilidades en el mismo formato.
    """
    folds = sorted(merged_train_folds["fold"].unique().tolist())
    oof_records = []
    est_params = {**PARAMS_ESTIMATOR, "random_state": seed}

    for fold_idx in folds:
        tr_f = merged_train_folds[merged_train_folds["fold"] != fold_idx].reset_index(drop=True)
        va_f = merged_train_folds[merged_train_folds["fold"] == fold_idx].reset_index(drop=True)

        pipe_s1 = Pipeline([
            ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
            ("clf", LogisticRegression(**est_params)),
        ])
        pipe_s1.fit(tr_f["formatted_interaction"], tr_f["risk_severity"])
        probs_s1_raw = pipe_s1.predict_proba(va_f["formatted_interaction"])

        probs_s1_aligned = np.zeros((len(va_f), len(classes_s1_orig)), dtype=np.float64)
        for idx_c, c_val in enumerate(classes_s1_orig):
            if c_val in pipe_s1.classes_:
                col_pos = list(pipe_s1.classes_).index(c_val)
                probs_s1_aligned[:, idx_c] = probs_s1_raw[:, col_pos]

        mlb_a = MultiLabelBinarizer(classes=classes_s2a)
        Y_a_tr = mlb_a.fit_transform(tr_f["risk_domains"])
        pipe_a = Pipeline([
            ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
            ("clf", OneVsRestClassifier(LogisticRegression(**est_params))),
        ])
        pipe_a.fit(tr_f["formatted_interaction"], Y_a_tr)
        probs_a_val = pipe_a.predict_proba(va_f["formatted_interaction"])

        mlb_b = MultiLabelBinarizer(classes=classes_s2b)
        Y_b_tr = mlb_b.fit_transform(tr_f["risk_drivers"])
        pipe_b = Pipeline([
            ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
            ("clf", OneVsRestClassifier(LogisticRegression(**est_params))),
        ])
        pipe_b.fit(tr_f["formatted_interaction"], Y_b_tr)
        probs_b_val = pipe_b.predict_proba(va_f["formatted_interaction"])

        for i, ex_id in enumerate(va_f["example_id"]):
            oof_records.append({
                "example_id": ex_id,
                "probs_s1": probs_s1_aligned[i].tolist(),
                "probs_s2a": {cls: float(probs_a_val[i, j]) for j, cls in enumerate(classes_s2a)},
                "probs_s2b": {cls: float(probs_b_val[i, j]) for j, cls in enumerate(classes_s2b)},
            })

    l1_oof_df = merged_train_folds.merge(pd.DataFrame(oof_records), on="example_id", how="inner")

    # Entrenar L1 en todo train y evaluar en dev (corrida única con `seed`)
    pipe_s1_full = Pipeline([
        ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
        ("clf", LogisticRegression(**est_params)),
    ])
    pipe_s1_full.fit(merged_train_folds["formatted_interaction"], merged_train_folds["risk_severity"])
    p_s1_dev_raw = pipe_s1_full.predict_proba(dev_df["formatted_interaction"])

    p_s1_dev_aligned = np.zeros((len(dev_df), len(classes_s1_orig)), dtype=np.float64)
    for idx_c, c_val in enumerate(classes_s1_orig):
        if c_val in pipe_s1_full.classes_:
            col_pos = list(pipe_s1_full.classes_).index(c_val)
            p_s1_dev_aligned[:, idx_c] = p_s1_dev_raw[:, col_pos]

    mlb_a_full = MultiLabelBinarizer(classes=classes_s2a)
    Y_a_full = mlb_a_full.fit_transform(merged_train_folds["risk_domains"])
    pipe_a_full = Pipeline([
        ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
        ("clf", OneVsRestClassifier(LogisticRegression(**est_params))),
    ])
    pipe_a_full.fit(merged_train_folds["formatted_interaction"], Y_a_full)
    p_a_dev = pipe_a_full.predict_proba(dev_df["formatted_interaction"])

    mlb_b_full = MultiLabelBinarizer(classes=classes_s2b)
    Y_b_full = mlb_b_full.fit_transform(merged_train_folds["risk_drivers"])
    pipe_b_full = Pipeline([
        ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
        ("clf", OneVsRestClassifier(LogisticRegression(**est_params))),
    ])
    pipe_b_full.fit(merged_train_folds["formatted_interaction"], Y_b_full)
    p_b_dev = pipe_b_full.predict_proba(dev_df["formatted_interaction"])

    l1_dev_df = dev_df.copy()
    l1_dev_df["probs_s1"] = [p_s1_dev_aligned[i].tolist() for i in range(len(l1_dev_df))]
    l1_dev_df["probs_s2a"] = [
        {cls: float(p_a_dev[i, j]) for j, cls in enumerate(classes_s2a)}
        for i in range(len(l1_dev_df))
    ]
    l1_dev_df["probs_s2b"] = [
        {cls: float(p_b_dev[i, j]) for j, cls in enumerate(classes_s2b)}
        for i in range(len(l1_dev_df))
    ]

    return l1_oof_df, l1_dev_df


def evaluar_cuadricula_s2(
    df: pd.DataFrame,
    col_etiquetas: str,
    col_probs: str,
    classes_ordenadas: list[str],
    umbrales: list,
) -> list[dict]:
    """Evalúa Jaccard agregado y por fold para cada umbral en la cuadrícula."""
    resultados_grid = []
    folds = sorted(df["fold"].unique().tolist()) if "fold" in df.columns else []

    for umbral in umbrales:
        etiqueta_umbral = "nunca_segunda_etiqueta" if umbral is None else umbral
        preds = [
            seleccionar_etiquetas_con_umbral(probs_raw, umbral, classes_ordenadas)
            for probs_raw in df[col_probs]
        ]
        scores = [jaccard(real, pred) for real, pred in zip(df[col_etiquetas], preds)]
        jaccard_agregado = float(np.mean(scores))

        por_fold = {}
        for f_idx in folds:
            mask = (df["fold"] == f_idx).to_numpy()
            scores_f = [s for s, m in zip(scores, mask) if m]
            por_fold[f"fold_{f_idx}"] = round(float(np.mean(scores_f)), 4)

        std_folds = float(np.std(list(por_fold.values()), ddof=1)) if len(por_fold) > 1 else 0.0

        resultados_grid.append({
            "threshold": etiqueta_umbral,
            "jaccard_mean": round(jaccard_agregado, 4),
            "jaccard_std_across_folds": round(std_folds, 4),
            "per_fold": por_fold,
        })

    return resultados_grid


def elegir_umbral_s2_estable(grid_s2: list[dict]) -> tuple[float | None, str, bool]:
    """
    Elige un umbral estable evitando picos aislados usando la media de vecindad
    (ventana de hasta 3 vecinos; en los extremos promedia 2 vecinos).
    Retorna (umbral_elegido, motivo, cae_en_extremo_inferior).
    """
    numericos = [item for item in grid_s2 if isinstance(item["threshold"], (int, float))]
    nunca_item = next((item for item in grid_s2 if item["threshold"] == "nunca_segunda_etiqueta"), None)

    mejor_umbral: float | None = None
    mejor_idx_num: int = -1
    mejor_score_suave: float = -1.0
    mejor_clave: tuple[float, float, float] | None = None
    mejor_n_vecinos: int = 0

    for i, item in enumerate(numericos):
        vecinos_idx = range(max(0, i - 1), min(len(numericos), i + 2))
        vecinos = [numericos[j]["jaccard_mean"] for j in vecinos_idx]
        score_suave = float(np.mean(vecinos))

        clave_candidata = (
            round(score_suave, 6),
            round(float(item["jaccard_mean"]), 6),
            -round(float(item["jaccard_std_across_folds"]), 6),
        )

        if mejor_clave is None or clave_candidata > mejor_clave:
            mejor_clave = clave_candidata
            mejor_score_suave = score_suave
            mejor_umbral = float(item["threshold"])
            mejor_idx_num = i
            mejor_n_vecinos = len(list(vecinos_idx))

    if nunca_item is not None and float(nunca_item["jaccard_mean"]) > mejor_score_suave + 0.005:
        return None, "nunca_segunda_etiqueta supera con margen la media de vecindad de umbrales", False

    cae_en_extremo_inferior = mejor_idx_num == 0
    borde_nota = " (ventana de 2 vecinos en extremo)" if mejor_n_vecinos < 3 else " (ventana de 3 vecinos)"
    aviso_extremo = (
        " [AVISO: cae en el extremo inferior de la cuadrícula; revisar si la curva sigue subiendo por debajo]"
        if cae_en_extremo_inferior
        else ""
    )
    motivo = (
        f"mayor score suavizado de vecindad{borde_nota} con desempate por Jaccard puntual y menor std entre folds"
        f"{aviso_extremo}"
    )
    return mejor_umbral, motivo, cae_en_extremo_inferior


def evaluar_cuadricula_s1(df: pd.DataFrame, taus: list[float]) -> list[dict]:
    """Evalúa MAE y subestimación severa en OOF para cada tau."""
    y_real = df["risk_severity"].to_numpy()
    folds = sorted(df["fold"].unique().tolist()) if "fold" in df.columns else []
    resultados = []

    for tau in taus:
        y_pred = np.array([predecir_s1_con_tau(p, tau) for p in df["probs_s1"]], dtype=np.int64)
        metricas_agg = calcular_metricas_s1(y_real, y_pred)

        por_fold = {}
        for f_idx in folds:
            mask = (df["fold"] == f_idx).to_numpy()
            por_fold[f"fold_{f_idx}"] = calcular_metricas_s1(y_real[mask], y_pred[mask])

        resultados.append({
            "tau": tau,
            **metricas_agg,
            "per_fold": por_fold,
        })

    return resultados


def elegir_tau_s1(grid_s1: list[dict], max_subest: float = MAX_SUBESTIMACION_PERMITIDA) -> tuple[float, str]:
    """Aplica el criterio declarado: menor MAE con subestimación severa <= max_subest."""
    candidatos = [row for row in grid_s1 if row["severe_underestimation_pct"] <= max_subest]
    if candidatos:
        elegido = min(candidatos, key=lambda r: (r["s1_mae"], r["severe_underestimation_pct"], r["tau"]))
        motivo = f"menor MAE ({elegido['s1_mae']}) cumpliendo subestimación <= {max_subest}%"
        return float(elegido["tau"]), motivo

    elegido = min(grid_s1, key=lambda r: (r["severe_underestimation_pct"], r["s1_mae"], r["tau"]))
    motivo = f"ningún tau alcanzó <= {max_subest}%; se eligió el de menor subestimación ({elegido['severe_underestimation_pct']}%)"
    return float(elegido["tau"]), motivo


def resumir_semillas(per_seed: dict[str, dict]) -> dict:
    resumen = {}
    for metrica in ["s1_mae", "severe_underestimation_pct", "s2a_jaccard", "s2b_jaccard"]:
        vals = np.array([per_seed[s][metrica] for s in per_seed], dtype=np.float64)
        dec = 2 if metrica == "severe_underestimation_pct" else 4
        resumen[metrica] = {
            "mean": round(float(np.mean(vals)), dec),
            "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, dec),
        }
    return resumen


def procesar_m1(
    oof_df: pd.DataFrame,
    dev_por_semilla: dict[int, pd.DataFrame],
    classes_s2a: list[str],
    classes_s2b: list[str],
) -> dict:
    """Evalúa M1 sin ajuste en OOF, selecciona umbrales en OOF y confirma en dev (3 semillas)."""
    sin_ajuste_oof = evaluar_predicciones_df(
        df=oof_df,
        classes_s2a=classes_s2a,
        classes_s2b=classes_s2b,
        umbral_s2a=UMBRAL_DEFAULT_S2,
        umbral_s2b=UMBRAL_DEFAULT_S2,
        tau_s1=None,
    )

    grid_s2a = evaluar_cuadricula_s2(oof_df, "risk_domains", "probs_s2a", classes_s2a, GRID_UMBRALES_S2)
    grid_s2b = evaluar_cuadricula_s2(oof_df, "risk_drivers", "probs_s2b", classes_s2b, GRID_UMBRALES_S2)

    umbral_s2a, motivo_s2a, ext_s2a = elegir_umbral_s2_estable(grid_s2a)
    umbral_s2b, motivo_s2b, ext_s2b = elegir_umbral_s2_estable(grid_s2b)

    grid_s1 = evaluar_cuadricula_s1(oof_df, GRID_TAU_S1)
    tau_s1, motivo_s1 = elegir_tau_s1(grid_s1, max_subest=MAX_SUBESTIMACION_PERMITIDA)

    con_ajuste_oof_optimista = evaluar_predicciones_df(
        df=oof_df,
        classes_s2a=classes_s2a,
        classes_s2b=classes_s2b,
        umbral_s2a=umbral_s2a,
        umbral_s2b=umbral_s2b,
        tau_s1=tau_s1,
    )

    dev_sin_ajuste_per_seed = {}
    dev_con_ajuste_per_seed = {}

    for s, df_dev_s in dev_por_semilla.items():
        dev_sin_ajuste_per_seed[f"seed_{s}"] = evaluar_predicciones_df(
            df=df_dev_s,
            classes_s2a=classes_s2a,
            classes_s2b=classes_s2b,
            umbral_s2a=UMBRAL_DEFAULT_S2,
            umbral_s2b=UMBRAL_DEFAULT_S2,
            tau_s1=None,
        )
        dev_con_ajuste_per_seed[f"seed_{s}"] = evaluar_predicciones_df(
            df=df_dev_s,
            classes_s2a=classes_s2a,
            classes_s2b=classes_s2b,
            umbral_s2a=umbral_s2a,
            umbral_s2b=umbral_s2b,
            tau_s1=tau_s1,
        )

    print("\n--- Resumen M1 ---")
    print(f"  Umbrales elegidos en OOF -> S2a: {umbral_s2a} | S2b: {umbral_s2b} | tau_S1: {tau_s1}")
    print(f"  OOF sin ajuste (argmax, umbral=0.3) : {sin_ajuste_oof}")
    print(f"  OOF con ajuste [OPTIMISTA por eleg.] : {con_ajuste_oof_optimista}")
    print(f"  DEV sin ajuste (media 3 semillas)    : {resumir_semillas(dev_sin_ajuste_per_seed)}")
    print(f"  DEV con ajuste [HONESTA] (media 3s)  : {resumir_semillas(dev_con_ajuste_per_seed)}")

    return {
        "evaluation_scheme": "3_seeds_dev_confirmation",
        "chosen_thresholds": {
            "s2a_second_label_threshold": umbral_s2a,
            "s2b_second_label_threshold": umbral_s2b,
            "s1_tau": tau_s1,
            "at_lower_grid_bound": {
                "s2a": ext_s2a,
                "s2b": ext_s2b,
            },
            "reasons": {
                "s2a": motivo_s2a,
                "s2b": motivo_s2b,
                "s1": motivo_s1,
            },
        },
        "oof_evaluation": {
            "without_tuning_default_rule": sin_ajuste_oof,
            "with_chosen_thresholds_optimistic_note": (
                "Métrica optimista porque los umbrales se seleccionaron sobre este mismo conjunto OOF; "
                "usar dev como evaluación honesta."
            ),
            "with_chosen_thresholds_optimistic": con_ajuste_oof_optimista,
        },
        "oof_grid": {
            "s2a_domains": grid_s2a,
            "s2b_drivers": grid_s2b,
            "s1_severity": grid_s1,
        },
        "dev_confirmation": {
            "without_tuning_default_rule": {
                "per_seed": dev_sin_ajuste_per_seed,
                "summary": resumir_semillas(dev_sin_ajuste_per_seed),
            },
            "with_chosen_thresholds": {
                "per_seed": dev_con_ajuste_per_seed,
                "summary": resumir_semillas(dev_con_ajuste_per_seed),
            },
        },
    }


def procesar_l1_corrida_unica(
    oof_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    classes_s2a: list[str],
    classes_s2b: list[str],
    seed: int = SEMILLA,
) -> dict:
    """Evalúa L1 (corrida única) sin ajuste en OOF, selecciona umbrales en OOF y confirma en dev."""
    sin_ajuste_oof = evaluar_predicciones_df(
        df=oof_df,
        classes_s2a=classes_s2a,
        classes_s2b=classes_s2b,
        umbral_s2a=UMBRAL_DEFAULT_S2,
        umbral_s2b=UMBRAL_DEFAULT_S2,
        tau_s1=None,
    )

    grid_s2a = evaluar_cuadricula_s2(oof_df, "risk_domains", "probs_s2a", classes_s2a, GRID_UMBRALES_S2)
    grid_s2b = evaluar_cuadricula_s2(oof_df, "risk_drivers", "probs_s2b", classes_s2b, GRID_UMBRALES_S2)

    umbral_s2a, motivo_s2a, ext_s2a = elegir_umbral_s2_estable(grid_s2a)
    umbral_s2b, motivo_s2b, ext_s2b = elegir_umbral_s2_estable(grid_s2b)

    grid_s1 = evaluar_cuadricula_s1(oof_df, GRID_TAU_S1)
    tau_s1, motivo_s1 = elegir_tau_s1(grid_s1, max_subest=MAX_SUBESTIMACION_PERMITIDA)

    con_ajuste_oof_optimista = evaluar_predicciones_df(
        df=oof_df,
        classes_s2a=classes_s2a,
        classes_s2b=classes_s2b,
        umbral_s2a=umbral_s2a,
        umbral_s2b=umbral_s2b,
        tau_s1=tau_s1,
    )

    dev_sin_ajuste = evaluar_predicciones_df(
        df=dev_df,
        classes_s2a=classes_s2a,
        classes_s2b=classes_s2b,
        umbral_s2a=UMBRAL_DEFAULT_S2,
        umbral_s2b=UMBRAL_DEFAULT_S2,
        tau_s1=None,
    )
    dev_con_ajuste = evaluar_predicciones_df(
        df=dev_df,
        classes_s2a=classes_s2a,
        classes_s2b=classes_s2b,
        umbral_s2a=umbral_s2a,
        umbral_s2b=umbral_s2b,
        tau_s1=tau_s1,
    )

    print("\n--- Resumen L1 (corrida única) ---")
    print(f"  Umbrales elegidos en OOF -> S2a: {umbral_s2a} | S2b: {umbral_s2b} | tau_S1: {tau_s1}")
    print(f"  OOF sin ajuste (argmax, umbral=0.3) : {sin_ajuste_oof}")
    print(f"  OOF con ajuste [OPTIMISTA por eleg.] : {con_ajuste_oof_optimista}")
    print(f"  DEV sin ajuste (corrida única)     : {dev_sin_ajuste}")
    print(f"  DEV con ajuste [HONESTA] (corr. ún.): {dev_con_ajuste}")

    return {
        "evaluation_scheme": "single_run",
        "seed": seed,
        "chosen_thresholds": {
            "s2a_second_label_threshold": umbral_s2a,
            "s2b_second_label_threshold": umbral_s2b,
            "s1_tau": tau_s1,
            "at_lower_grid_bound": {
                "s2a": ext_s2a,
                "s2b": ext_s2b,
            },
            "reasons": {
                "s2a": motivo_s2a,
                "s2b": motivo_s2b,
                "s1": motivo_s1,
            },
        },
        "oof_evaluation": {
            "without_tuning_default_rule": sin_ajuste_oof,
            "with_chosen_thresholds_optimistic_note": (
                "Métrica optimista porque los umbrales se seleccionaron sobre este mismo conjunto OOF; "
                "usar dev como evaluación honesta."
            ),
            "with_chosen_thresholds_optimistic": con_ajuste_oof_optimista,
        },
        "oof_grid": {
            "s2a_domains": grid_s2a,
            "s2b_drivers": grid_s2b,
            "s1_severity": grid_s1,
        },
        "dev_confirmation": {
            "without_tuning_default_rule": {
                "single_run": dev_sin_ajuste,
            },
            "with_chosen_thresholds": {
                "single_run": dev_con_ajuste,
            },
        },
    }


def main():
    print("=== SELECCIÓN DE UMBRALES EN OOF (5 FOLDS) Y CONFIRMACIÓN EN DEV (M1 Y L1) ===")

    train_df = pd.read_json(TRAIN_PATH, lines=True)
    dev_df = pd.read_json(DEV_PATH, lines=True)
    splits_df = pd.read_csv(SPLITS_PATH)

    merged_train_folds = train_df.merge(splits_df[["example_id", "fold"]], on="example_id", how="inner")
    print(f"Ejemplos en train con fold: {len(merged_train_folds)} | Ejemplos en dev: {len(dev_df)}")

    classes_s1_orig = sorted(merged_train_folds["risk_severity"].unique().tolist())

    mlb_a = MultiLabelBinarizer()
    mlb_a.fit(merged_train_folds["risk_domains"])
    classes_s2a = list(mlb_a.classes_)

    mlb_b = MultiLabelBinarizer()
    mlb_b.fit(merged_train_folds["risk_drivers"])
    classes_s2b = list(mlb_b.classes_)

    # 1. M1 OOF y dev (3 semillas)
    m1_oof_df = cargar_m1_oof_cv(train_df, splits_df, seed=SEMILLA)
    m1_dev_por_semilla = {s: cargar_m1_dev_semilla(dev_df, seed=s) for s in SEEDS_DEV}
    reporte_m1 = procesar_m1(m1_oof_df, m1_dev_por_semilla, classes_s2a, classes_s2b)

    # 2. L1: corrida única en OOF (5 folds) y dev
    l1_oof_df, l1_dev_df = generar_l1_oof_y_dev(
        merged_train_folds=merged_train_folds,
        dev_df=dev_df,
        classes_s1_orig=classes_s1_orig,
        classes_s2a=classes_s2a,
        classes_s2b=classes_s2b,
        seed=SEMILLA,
    )
    reporte_l1 = procesar_l1_corrida_unica(l1_oof_df, l1_dev_df, classes_s2a, classes_s2b, seed=SEMILLA)

    # 3. Guardar experiments/umbrales_resultados.json
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    payload = {
        "seed_oof": SEMILLA,
        "seeds_dev_confirmation_m1": SEEDS_DEV,
        "l1_evaluation": "corrida única",
        "default_rule": {
            "s1": "argmax",
            "s2_second_label_threshold": UMBRAL_DEFAULT_S2,
        },
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_oof_examples": int(len(merged_train_folds)),
        "n_dev_examples": int(len(dev_df)),
        "methodological_notes": {
            "oof_optimism": "Las métricas OOF tras elegir umbrales son optimistas; reportar dev como cifra honesta.",
            "s2_window_edges": "La ventana de suavizado usa hasta 3 vecinos (2 vecinos en los extremos de la cuadrícula).",
            "s2_lower_bound_check": "Se incluyen 0.10 y 0.15 para comprobar si el óptimo de S2b sigue subiendo por debajo de 0.20.",
            "s1_tau_grid": "Cuadrícula de tau redondeada a 3 decimales para mantener pasos de 0.025.",
            "l1_single_run": "L1 se reporta como corrida única (sin media ni std entre semillas).",
        },
        "selection_criteria": {
            "s2": "estabilidad en cuadrícula de umbrales (ventana de vecindad para evitar picos aislados)",
            "s1": CRITERIO_S1,
            "s1_max_severe_underestimation_pct": MAX_SUBESTIMACION_PERMITIDA,
            "s1_rule": "predict >= 4 if P(sev 4) + P(sev 5) > tau",
        },
        "models": {
            "M1": reporte_m1,
            "L1": reporte_l1,
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    print(f"\nResultados guardados en: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()