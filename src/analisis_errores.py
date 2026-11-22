import os
import sys
import json
from collections import Counter
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.prototipo import (
    DEFAULT_CKPT_DIR,
    UMBRALES_FILE,
    _predecir_severidad_con_tau,
    cargar_umbrales,
    formatear_texto_caso,
)
from src.train_m1 import (
    DEV_PATH,
    TRAIN_PATH,
    MAX_LEN_DEFAULT,
    MODEL_NAME,
    UMBRAL_SEGUNDA_ETIQUETA,
    jaccard,
    seleccionar_etiquetas,
)

EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
M1_DEV_PREDS_PATH = os.path.join(EXPERIMENTS_DIR, "m1_dev_predicciones.jsonl")
M1_RESULTS_PATH = os.path.join(EXPERIMENTS_DIR, "m1_resultados.json")
# Todos los ejemplos de dev, con texto del dataset (no se sube al repositorio: ver .gitignore).
ERRORES_CSV_PATH = os.path.join(EXPERIMENTS_DIR, "errores_dev.csv")
# Solo los casos seleccionados y sin texto del dataset; aquí se escribe la causa a mano.
SELECCIONADOS_CSV_PATH = os.path.join(EXPERIMENTS_DIR, "errores_dev_seleccionados.csv")

# Métricas por subgrupo (IA responsable): se guardan sin texto del dataset, así que sí se pueden subir.
SUBGRUPOS_CSV_PATH = os.path.join(EXPERIMENTS_DIR, "metricas_subgrupos_dev.csv")
# Un subgrupo con menos ejemplos que esto se marca como no confiable.
MIN_N_CONFIABLE = 20

# Regla de selección, fijada antes de mirar los casos (se documenta en el README).
N_POR_REGLA = 5
MINIMO_CASOS = 10

# Categorías de causa definidas antes de la clasificación manual (`sesgo_clase_frecuente` se añadió después de ver los casos).
CATEGORIAS_CAUSA = [
    "ambiguedad_etiqueta",       # la etiqueta real es discutible o hay otra igual de razonable
    "inicio_recortado",          # el límite de 512 tokens recortó el inicio de la trayectoria
    "senal_no_explicita",        # el riesgo no se ve en el texto de la acción o la observación
    "etiqueta_poco_frecuente",   # la etiqueta real casi no aparece en train
    "posible_error_anotacion",   # la etiqueta real parece incorrecta
    "sesgo_clase_frecuente",     # el modelo se inclina por una etiqueta frecuente (añadida después de ver los casos)
    "otra",
]

# Referencia para comprobar que este script reproduce la cifra ya reportada (semilla 42, umbrales elegidos).
TOL_MAE = 0.002
TOL_PCT = 0.05
TOL_JACCARD = 0.002


def cargar_predicciones_m1_dev(ruta_jsonl: str = M1_DEV_PREDS_PATH) -> dict[str, dict]:
    if not os.path.exists(ruta_jsonl):
        raise FileNotFoundError(f"No se encontró {ruta_jsonl}")
    preds = {}
    with open(ruta_jsonl, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            rec = json.loads(linea)
            preds[str(rec["example_id"])] = rec
    return preds


def obtener_clases_s2(preds_map: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene las clases ordenadas de S2a y S2b desde m1_resultados.json o desde la primera predicción."""
    if os.path.exists(M1_RESULTS_PATH):
        with open(M1_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        clases = data.get("classes", {})
        s2a = clases.get("s2a_domains")
        s2b = clases.get("s2b_drivers")
        if s2a and s2b:
            return np.array(s2a), np.array(s2b)

    primera = next(iter(preds_map.values()))
    return np.array(list(primera["probs_s2a"].keys())), np.array(list(primera["probs_s2b"].keys()))


def cargar_tokenizador_local(ckpt_dir: str = DEFAULT_CKPT_DIR, max_len: int = MAX_LEN_DEFAULT):
    """Carga el tokenizador guardado con truncamiento por la izquierda; si no existe, cae al base."""
    tok_dir = os.path.join(ckpt_dir, "tokenizer")
    if os.path.isdir(tok_dir):
        tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True, truncation_side="left")
    else:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, truncation_side="left")
    tok.truncation_side = "left"
    tok.model_max_length = max_len
    return tok


def extraer_ultima_accion(row: pd.Series, texto_formateado: str) -> str:
    """
    Intenta extraer la última acción propuesta del registro si existe en campos estructurados;
    si no, toma el tramo final de la interacción formateada.
    """
    for campo in ("proposed_action", "final_action", "last_action", "action"):
        if campo in row and pd.notna(row[campo]) and str(row[campo]).strip():
            return str(row[campo]).strip()

    # Buscar bloques típicos al final de la trayectoria (ej. Assistant / Action / Turno final)
    patrones = ["Proposed Action:", "Final Action:", "Action:", "Assistant:"]
    for pat in patrones:
        idx = texto_formateado.rfind(pat)
        if idx != -1:
            fragmento = texto_formateado[idx + len(pat) :].strip()
            if fragmento:
                return fragmento

    # Si no hay marcador explícito, tomar los últimos 600 caracteres y devolver sus primeros 300
    return texto_formateado[-600:].strip()


def imprimir_matriz_confusion_s1(y_real: np.ndarray, y_pred: np.ndarray):
    """Imprime la matriz de confusión 5x5 para severidades 1..5."""
    niveles = [1, 2, 3, 4, 5]
    matriz = np.zeros((5, 5), dtype=int)
    for r, p in zip(y_real, y_pred):
        r_int = int(r)
        p_int = int(p)
        if 1 <= r_int <= 5 and 1 <= p_int <= 5:
            matriz[r_int - 1, p_int - 1] += 1

    print("\n=== Matriz de confusión S1 (filas = real 1..5, columnas = predicha 1..5) ===")
    encabezado = "Real \\ Pred | " + " | ".join(f"S{c}" for c in niveles) + " | Total"
    print(encabezado)
    print("-" * len(encabezado))
    for i, r_lvl in enumerate(niveles):
        fila_str = " | ".join(f"{matriz[i, j]:2d}" for j in range(5))
        print(f"     S{r_lvl}     | {fila_str} | {matriz[i].sum():5d}")


def reportar_confusiones_s2(df_errores: pd.DataFrame, col_real: str, col_pred: str, top_k: int = 8):
    """
    Calcula por etiqueta los falsos positivos (FP) y falsos negativos (FN),
    y los pares (etiqueta_real, etiqueta_predicha) más frecuentes cuando hay desacuerdo.
    """
    fp_counter = Counter()
    fn_counter = Counter()
    pares_error = Counter()

    for reales_str, preds_str in zip(df_errores[col_real], df_errores[col_pred]):
        reales = [x.strip() for x in str(reales_str).split(" | ") if x.strip()]
        preds = [x.strip() for x in str(preds_str).split(" | ") if x.strip()]
        set_r = set(reales)
        set_p = set(preds)

        for lbl in set_p - set_r:
            fp_counter[lbl] += 1
        for lbl in set_r - set_p:
            fn_counter[lbl] += 1

        if set_r != set_p:
            for r_lbl in set_r - set_p:
                for p_lbl in set_p - set_r:
                    pares_error[(r_lbl, p_lbl)] += 1

    todas_etiquetas = sorted(set(fp_counter.keys()) | set(fn_counter.keys()))
    ranking_etiquetas = sorted(
        todas_etiquetas,
        key=lambda lbl: (fp_counter[lbl] + fn_counter[lbl], fn_counter[lbl], fp_counter[lbl]),
        reverse=True,
    )[:top_k]

    print(f"\n=== Top {top_k} etiquetas con más confusiones en {col_real} ===")
    for lbl in ranking_etiquetas:
        fp = fp_counter[lbl]
        fn = fn_counter[lbl]
        print(f"  {lbl:<32} -> total_err={fp + fn:3d} (FN={fn:2d}, FP={fp:2d})")

    if pares_error:
        print(f"\n=== Pares (real -> predicha) más frecuentes en desacuerdos de {col_real} ===")
        for (r_lbl, p_lbl), cnt in pares_error.most_common(min(top_k, len(pares_error))):
            print(f"  Real '{r_lbl}' -> Pred '{p_lbl}': {cnt} veces")


def seleccionar_casos(df: pd.DataFrame, n: int = N_POR_REGLA) -> dict[str, list[str]]:
    """
    Regla fijada de antemano (desempate siempre por example_id ascendente):
      1. mayor_error_S1: los n con mayor error absoluto en S1.
      2. severo_subestimado: de los severos subestimados, los n con mayor P(severidad 1-3).
      3. s2a_jaccard_0: los n primeros (por example_id) con Jaccard 0 en S2a.
      4. s2b_jaccard_0: los n primeros (por example_id) con Jaccard 0 en S2b.
    Un caso puede cumplir varias reglas; se cuenta una vez y se guardan todos los motivos.
    """
    motivos: dict[str, list[str]] = {}

    def marcar(sub: pd.DataFrame, motivo: str):
        for ex in sub["example_id"]:
            motivos.setdefault(str(ex), []).append(motivo)

    marcar(
        df.sort_values(["error_absoluto", "example_id"], ascending=[False, True]).head(n),
        "mayor_error_S1",
    )
    marcar(
        df[df["severo_subestimado"]]
        .sort_values(["prob_s1_bajo", "example_id"], ascending=[False, True])
        .head(n),
        "severo_subestimado",
    )
    marcar(df[df["s2a_jaccard"] == 0].sort_values("example_id").head(n), "s2a_jaccard_0")
    marcar(df[df["s2b_jaccard"] == 0].sort_values("example_id").head(n), "s2b_jaccard_0")
    return motivos


def cargar_causas_previas(ruta: str = SELECCIONADOS_CSV_PATH) -> dict[str, dict]:
    """Conserva las causas escritas a mano si el script se vuelve a ejecutar."""
    if not os.path.exists(ruta):
        return {}
    previo = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    if "example_id" not in previo.columns:
        return {}
    out = {}
    for _, r in previo.iterrows():
        out[str(r["example_id"])] = {
            "causa": str(r.get("causa", "")).strip(),
            "comentario": str(r.get("comentario", "")).strip(),
        }
    return out


def reportar_por_longitud(df: pd.DataFrame):
    """
    Error por cuartil de longitud. Casi todos los ejemplos superan los 512 tokens (se recorta el inicio de
    la trayectoria), así que no sirve comparar truncados con no truncados; se compara por longitud.
    """
    print(f"\n=== Errores por cuartil de longitud en tokens (M1 solo ve los últimos {MAX_LEN_DEFAULT}) ===")
    cuartil = pd.qcut(df["longitud_tokens"], 4, duplicates="drop")
    for intervalo, g in df.groupby(cuartil, observed=True):
        severos = int((g["sev_real"] >= 4).sum())
        sub = int(g["severo_subestimado"].sum())
        print(
            f"  {str(intervalo):<18}: n={len(g):3d} | MAE S1={g['error_absoluto'].mean():.4f} | "
            f"sesgo={g['error_con_signo'].mean():+.3f} | J2a={g['s2a_jaccard'].mean():.3f} | "
            f"J2b={g['s2b_jaccard'].mean():.3f} | severos subestimados={sub}/{severos}"
        )
    for col in ("error_absoluto", "s2a_jaccard", "s2b_jaccard"):
        rho = df["longitud_tokens"].rank().corr(df[col].rank())
        print(f"  Spearman longitud vs {col}: {rho:+.3f}")
    print("  (n pequeño por cuartil: son pistas, no pruebas)")


def _fila_subgrupo(tipo: str, nombre: str, g: pd.DataFrame) -> dict:
    return {
        "tipo_subgrupo": tipo,
        "subgrupo": nombre,
        "n": int(len(g)),
        "confiable": bool(len(g) >= MIN_N_CONFIABLE),
        "mae_s1": round(float(g["error_absoluto"].mean()), 4),
        "sesgo_s1": round(float(g["error_con_signo"].mean()), 4),
        "jaccard_s2a": round(float(g["s2a_jaccard"].mean()), 4),
        "jaccard_s2b": round(float(g["s2b_jaccard"].mean()), 4),
        "severos": int((g["sev_real"] >= 4).sum()),
        "severos_subestimados": int(g["severo_subestimado"].sum()),
    }


def reportar_por_subgrupo(df: pd.DataFrame):
    """
    Métricas por subgrupo (criterio de IA responsable). Subgrupos fijados antes de mirar los resultados:
      - severidad real (1, 2, 3 y 4-5)
      - cuartil de longitud en tokens
      - dominio de daño real (S2a) y factor de riesgo real (S2b): un ejemplo cuenta en cada etiqueta que tiene
    Es una sola semilla (42) sobre dev; los subgrupos con menos de MIN_N_CONFIABLE ejemplos se marcan.
    """
    filas = [_fila_subgrupo("todos", "todos", df)]

    por_sev = [("1", df["sev_real"] == 1), ("2", df["sev_real"] == 2), ("3", df["sev_real"] == 3), ("4-5", df["sev_real"] >= 4)]
    for nombre, mascara in por_sev:
        if mascara.any():
            filas.append(_fila_subgrupo("severidad_real", nombre, df[mascara]))

    cuartil = pd.qcut(df["longitud_tokens"], 4, duplicates="drop")
    for intervalo, g in df.groupby(cuartil, observed=True):
        filas.append(_fila_subgrupo("cuartil_longitud", str(intervalo), g))

    for tipo, col in (("dominio_real", "s2a_real"), ("factor_real", "s2b_real")):
        listas = df[col].map(lambda s: [x.strip() for x in str(s).split(" | ") if x.strip()])
        etiquetas = sorted({lbl for fila in listas for lbl in fila})
        bloque = [
            _fila_subgrupo(tipo, lbl, df[listas.map(lambda fila, lbl=lbl: lbl in fila)])
            for lbl in etiquetas
        ]
        filas.extend(sorted(bloque, key=lambda r: -r["n"]))

    tabla = pd.DataFrame(filas)
    tabla.to_csv(SUBGRUPOS_CSV_PATH, index=False, encoding="utf-8")

    print(f"\n=== Métricas por subgrupo (M1, semilla 42, dev; * = n < {MIN_N_CONFIABLE}, no confiable) ===")
    print(f"{'tipo':<17}{'subgrupo':<40}{'n':>4}  {'MAE S1':>7} {'sesgo':>7} {'J2a':>6} {'J2b':>6}  severos sub./total")
    for _, r in tabla.iterrows():
        marca = " " if r["confiable"] else "*"
        print(
            f"{r['tipo_subgrupo']:<17}{r['subgrupo']:<40}{r['n']:>4}{marca} {r['mae_s1']:>7.3f} {r['sesgo_s1']:>+7.3f} "
            f"{r['jaccard_s2a']:>6.3f} {r['jaccard_s2b']:>6.3f}  {r['severos_subestimados']}/{r['severos']}"
        )
    print(f"Archivo generado: {SUBGRUPOS_CSV_PATH}")
    print("  (los datos no traen atributos de personas: se mide entre tipos de caso, no entre grupos humanos)")


def verificar_contra_referencia(df: pd.DataFrame):
    """Compara con la cifra de la semilla 42 y los umbrales elegidos guardada en umbrales_resultados.json."""
    try:
        with open(UMBRALES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ref = data["models"]["M1"]["dev_confirmation"]["with_chosen_thresholds"]["per_seed"]["seed_42"]
    except (OSError, KeyError, json.JSONDecodeError):
        print("\n[Verificación] No se encontró la referencia de la semilla 42 en umbrales_resultados.json.")
        return

    severos = df["sev_real"] >= 4
    pct = 100.0 * df.loc[severos, "severo_subestimado"].mean() if severos.any() else 0.0
    comparaciones = [
        ("MAE S1", df["error_absoluto"].mean(), ref["s1_mae"], TOL_MAE),
        ("Severos subestimados %", pct, ref["severe_underestimation_pct"], TOL_PCT),
        ("Jaccard S2a", df["s2a_jaccard"].mean(), ref["s2a_jaccard"], TOL_JACCARD),
        ("Jaccard S2b", df["s2b_jaccard"].mean(), ref["s2b_jaccard"], TOL_JACCARD),
    ]
    print("\n=== Verificación contra la semilla 42 con umbrales elegidos (umbrales_resultados.json) ===")
    todo_ok = True
    for nombre, calculado, esperado, tol in comparaciones:
        ok = abs(calculado - esperado) <= tol
        todo_ok = todo_ok and ok
        print(f"  {nombre:<24} calculado={calculado:.4f} | referencia={esperado:.4f} | {'OK' if ok else 'DESAJUSTE'}")
    if not todo_ok:
        print("  Hay un desajuste: revisar formatear_texto_caso, los umbrales o el archivo de predicciones antes de usar el CSV.")


def imprimir_frecuencias_train():
    """Frecuencia de cada etiqueta de S2 en train, para juzgar si una etiqueta es poco frecuente."""
    ruta = TRAIN_PATH if os.path.isabs(TRAIN_PATH) else os.path.join(REPO_ROOT, TRAIN_PATH)
    if not os.path.exists(ruta):
        print(f"\n[Frecuencias] No se encontró {ruta}")
        return
    train_df = pd.read_json(ruta, lines=True)
    n = len(train_df)
    for col, titulo in (("risk_domains", "S2a (dominios)"), ("risk_drivers", "S2b (factores)")):
        cont = Counter(lbl for fila in train_df[col] for lbl in fila)
        print(f"\n=== Frecuencia en train de las etiquetas de {titulo} (n={n} ejemplos) ===")
        for lbl, c in sorted(cont.items(), key=lambda kv: -kv[1]):
            print(f"  {lbl:<40} {c:4d} ({100.0 * c / n:4.1f} %)")


def main():
    ruta_dev = DEV_PATH if os.path.isabs(DEV_PATH) else os.path.join(REPO_ROOT, DEV_PATH)
    dev_df = pd.read_json(ruta_dev, lines=True)
    preds_map = cargar_predicciones_m1_dev(M1_DEV_PREDS_PATH)

    umbrales = cargar_umbrales(UMBRALES_FILE)
    tau_s1 = umbrales.get("s1_tau", None)
    umbral_s2a = umbrales.get("s2a_second_label_threshold", UMBRAL_SEGUNDA_ETIQUETA)
    umbral_s2b = umbrales.get("s2b_second_label_threshold", UMBRAL_SEGUNDA_ETIQUETA)

    classes_s2a, classes_s2b = obtener_clases_s2(preds_map)
    tokenizer = cargar_tokenizador_local(DEFAULT_CKPT_DIR, max_len=MAX_LEN_DEFAULT)

    print(
        f"Análisis de errores M1 (semilla 42) en dev | "
        f"tau_S1={tau_s1} | umbral_S2a={umbral_s2a} | umbral_S2b={umbral_s2b}"
    )

    filas = []
    for _, row in dev_df.iterrows():
        ex_id = str(row["example_id"])
        if ex_id not in preds_map:
            continue

        rec = preds_map[ex_id]
        texto_formateado = formatear_texto_caso(row.to_dict())

        # Conteo de tokens sin truncar para saber longitud real y si supera 512
        tokens_sin_truncar = tokenizer(
            texto_formateado, truncation=False, add_special_tokens=True, verbose=False
        )["input_ids"]
        n_tokens = len(tokens_sin_truncar)
        truncado_512 = bool(n_tokens > MAX_LEN_DEFAULT)

        # Primeros 300 caracteres de la última acción
        ultima_accion = extraer_ultima_accion(row, texto_formateado)
        ult_accion_300 = ultima_accion[:300]

        # Severidad S1
        sev_real = int(row["risk_severity"])
        probs_s1 = np.asarray(rec["probs_s1"], dtype=np.float64)
        sev_pred = int(_predecir_severidad_con_tau(probs_s1, tau_s1))
        err_signo = sev_pred - sev_real
        err_abs = abs(err_signo)
        severo_subestimado = bool((sev_real >= 4) and (sev_pred <= 2))
        prob_s1_bajo = round(float(probs_s1[:3].sum()), 4)

        # S2a: dominios
        dom_real = list(row["risk_domains"])
        probs_s2a_vec = np.array([float(rec["probs_s2a"][str(c)]) for c in classes_s2a], dtype=np.float64)
        dom_pred = list(seleccionar_etiquetas(probs_s2a_vec, classes_s2a, umbral=umbral_s2a))
        jacc_s2a = round(float(jaccard(dom_real, dom_pred)), 4)

        # S2b: factores / orígenes
        fac_real = list(row["risk_drivers"])
        probs_s2b_vec = np.array([float(rec["probs_s2b"][str(c)]) for c in classes_s2b], dtype=np.float64)
        fac_pred = list(seleccionar_etiquetas(probs_s2b_vec, classes_s2b, umbral=umbral_s2b))
        jacc_s2b = round(float(jaccard(fac_real, fac_pred)), 4)

        filas.append({
            "example_id": ex_id,
            "sev_real": sev_real,
            "sev_pred": sev_pred,
            "error_con_signo": err_signo,
            "error_absoluto": err_abs,
            "severo_subestimado": severo_subestimado,
            "prob_s1_bajo": prob_s1_bajo,
            "s2a_real": " | ".join(dom_real),
            "s2a_pred": " | ".join(dom_pred),
            "s2a_jaccard": jacc_s2a,
            "s2b_real": " | ".join(fac_real),
            "s2b_pred": " | ".join(fac_pred),
            "s2b_jaccard": jacc_s2b,
            "longitud_tokens": n_tokens,
            "truncado_512": truncado_512,
            "ult_accion_300_chars": ult_accion_300,
        })

    df_errores = pd.DataFrame(filas)

    # Selección de casos y columnas de clasificación manual
    motivos = seleccionar_casos(df_errores, N_POR_REGLA)
    df_errores["seleccionado"] = df_errores["example_id"].isin(motivos)
    df_errores["motivo_seleccion"] = df_errores["example_id"].map(lambda ex: "; ".join(motivos.get(ex, [])))

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    df_errores.to_csv(ERRORES_CSV_PATH, index=False, encoding="utf-8")

    # Tabla de seleccionados sin texto del dataset; conserva las causas ya escritas a mano
    previas = cargar_causas_previas(SELECCIONADOS_CSV_PATH)
    sel = df_errores[df_errores["seleccionado"]].drop(columns=["ult_accion_300_chars", "seleccionado", "prob_s1_bajo"]).copy()
    sel["causa"] = sel["example_id"].map(lambda ex: previas.get(ex, {}).get("causa", ""))
    sel["comentario"] = sel["example_id"].map(lambda ex: previas.get(ex, {}).get("comentario", ""))
    sel = sel.sort_values("example_id")
    sel.to_csv(SELECCIONADOS_CSV_PATH, index=False, encoding="utf-8")

    print(f"\nArchivo generado: {ERRORES_CSV_PATH} ({len(df_errores)} filas, con texto: no se sube al repositorio)")
    print(f"Archivo generado: {SELECCIONADOS_CSV_PATH} ({len(sel)} casos seleccionados, sin texto del dataset)")
    print(f"MAE S1: {df_errores['error_absoluto'].mean():.4f} | Sesgo medio (pred-real): {df_errores['error_con_signo'].mean():+.4f}")
    print(f"Severos subestimados (real >= 4 y pred <= 2): {int(df_errores['severo_subestimado'].sum())} / {int((df_errores['sev_real'] >= 4).sum())}")
    print(f"Ejemplos truncados a {MAX_LEN_DEFAULT} tokens: {int(df_errores['truncado_512'].sum())} / {len(df_errores)}")

    verificar_contra_referencia(df_errores)

    imprimir_matriz_confusion_s1(
        df_errores["sev_real"].to_numpy(),
        df_errores["sev_pred"].to_numpy(),
    )
    reportar_confusiones_s2(df_errores, "s2a_real", "s2a_pred", top_k=8)
    reportar_confusiones_s2(df_errores, "s2b_real", "s2b_pred", top_k=8)
    reportar_por_longitud(df_errores)
    reportar_por_subgrupo(df_errores)

    print(f"\n=== Casos seleccionados ({len(motivos)} únicos; regla: {N_POR_REGLA} por criterio, desempate por example_id) ===")
    for ex in sorted(motivos):
        r = df_errores[df_errores["example_id"] == ex].iloc[0]
        print(
            f"  {ex} | S1 real={r['sev_real']} pred={r['sev_pred']} | "
            f"J2a={r['s2a_jaccard']:.2f} J2b={r['s2b_jaccard']:.2f} | "
            f"tokens={r['longitud_tokens']}{' (trunc.)' if r['truncado_512'] else ''} | {'; '.join(motivos[ex])}"
        )
    if len(motivos) < MINIMO_CASOS:
        print(f"  AVISO: hay menos de {MINIMO_CASOS} casos únicos; ampliar N_POR_REGLA.")

    causas_invalidas = sorted({c for c in sel["causa"] if c and c not in CATEGORIAS_CAUSA})
    if causas_invalidas:
        print(f"\nAVISO: causas fuera de la taxonomía: {causas_invalidas}")
    print("\nCategorías de causa permitidas:", ", ".join(CATEGORIAS_CAUSA))
    print(f"Escribe la causa de cada caso en {SELECCIONADOS_CSV_PATH} (columnas `causa` y `comentario`).")

    imprimir_frecuencias_train()


if __name__ == "__main__":
    main()
