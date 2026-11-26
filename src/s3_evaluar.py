"""
S3: evalúa el monitor (M1 con los umbrales elegidos + política) contra la intervención de referencia.

La referencia sale de aplicar cada política a las etiquetas reales de dev. El monitor usa las predicciones de M1
(semilla 42, umbrales elegidos) que deja `src/analisis_errores.py` en `experiments/errores_dev.csv`; hay que
ejecutar ese script antes. Políticas en `src/s3_politicas.py` y `docs/politicas_s3.md`.
"""
import json
import os
import sys
from collections import Counter

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.s3_politicas import INTERVENCIONES, POLITICAS, aplicar_politica

DEV_PATH = os.path.join(REPO_ROOT, "data", "raw", "data", "dev.jsonl")
ERRORES_CSV_PATH = os.path.join(REPO_ROOT, "experiments", "errores_dev.csv")
REFERENCIA_CSV_PATH = os.path.join(REPO_ROOT, "experiments", "s3_referencia_dev.csv")
RESULTADOS_PATH = os.path.join(REPO_ROOT, "experiments", "s3_monitor_resultados.json")


def separar_etiquetas(texto) -> list[str]:
    """Convierte un texto 'a | b' en la lista de etiquetas ['a', 'b']."""
    return [x.strip() for x in str(texto).split(" | ") if x.strip()]


def metricas_por_clase(y_real: list[str], y_pred: list[str]) -> dict:
    """Precisión, recall y F1 por intervención, F1 macro, exactitud y matriz de confusión."""
    por_clase = {}
    for c in INTERVENCIONES:
        tp = sum(1 for r, p in zip(y_real, y_pred) if r == c and p == c)
        fp = sum(1 for r, p in zip(y_real, y_pred) if r != c and p == c)
        fn = sum(1 for r, p in zip(y_real, y_pred) if r == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        por_clase[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "soporte": tp + fn}
    matriz = {r: {p: sum(1 for a, b in zip(y_real, y_pred) if a == r and b == p) for p in INTERVENCIONES} for r in INTERVENCIONES}
    return {
        "f1_macro": round(sum(v["f1"] for v in por_clase.values()) / len(INTERVENCIONES), 4),
        "exactitud": round(sum(1 for r, p in zip(y_real, y_pred) if r == p) / len(y_real), 4),
        "por_clase": por_clase,
        "matriz_confusion_real_x_predicha": matriz,
    }


def main():
    """Compara la intervención del monitor (M1 con los umbrales elegidos) con la de referencia, para las políticas A y B."""
    dev = {}
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                r = json.loads(linea)
                dev[str(r["example_id"])] = r

    if not os.path.exists(ERRORES_CSV_PATH):
        raise FileNotFoundError(f"No se encontró {ERRORES_CSV_PATH}. Ejecuta primero `python src/analisis_errores.py`.")
    preds = pd.read_csv(ERRORES_CSV_PATH, dtype=str, keep_default_na=False)
    preds = preds[preds["example_id"].isin(dev)]

    filas = []
    for _, r in preds.iterrows():
        ex = r["example_id"]
        real = dev[ex]
        fila = {"example_id": ex}
        for pol in sorted(POLITICAS):
            fila[f"referencia_{pol}"] = aplicar_politica(pol, real["risk_severity"], real["risk_domains"], real["risk_drivers"])
            fila[f"monitor_{pol}"] = aplicar_politica(pol, int(r["sev_pred"]), separar_etiquetas(r["s2a_pred"]), separar_etiquetas(r["s2b_pred"]))
        filas.append(fila)
    tabla = pd.DataFrame(filas).sort_values("example_id")
    tabla.to_csv(REFERENCIA_CSV_PATH, index=False, encoding="utf-8")

    resultados = {
        "n_casos": int(len(tabla)),
        "modelo": "M1 (semilla 42, umbrales elegidos con los folds de train)",
        "nota": "La referencia sale de aplicar la política a las etiquetas reales; no es un juicio humano independiente.",
        "politicas": {},
    }
    print(f"S3 | monitor M1 + política contra la referencia | {len(tabla)} casos de dev")
    for pol in sorted(POLITICAS):
        real = tabla[f"referencia_{pol}"].tolist()
        pred = tabla[f"monitor_{pol}"].tolist()
        m = metricas_por_clase(real, pred)
        mayoritaria = Counter(real).most_common(1)[0][0]
        m["referencia_clase_mayoritaria"] = {"clase": mayoritaria, **{k: v for k, v in metricas_por_clase(real, [mayoritaria] * len(real)).items() if k in ("f1_macro", "exactitud")}}
        resultados["politicas"][pol] = m
        print(f"\n=== Política {pol}: {POLITICAS[pol]}")
        print(f"F1 macro del monitor: {m['f1_macro']:.4f} | exactitud: {m['exactitud']:.4f} | siempre '{mayoritaria}': F1 macro {m['referencia_clase_mayoritaria']['f1_macro']:.4f}")
        for c in INTERVENCIONES:
            v = m["por_clase"][c]
            print(f"  {c:<10} precisión={v['precision']:.3f} recall={v['recall']:.3f} F1={v['f1']:.3f} (soporte {v['soporte']})")
        print("  Matriz de confusión (filas = referencia, columnas = monitor):")
        print("  " + " " * 10 + "".join(f"{c:>11}" for c in INTERVENCIONES))
        for r_ in INTERVENCIONES:
            print("  " + f"{r_:<10}" + "".join(f"{m['matriz_confusion_real_x_predicha'][r_][p_]:>11}" for p_ in INTERVENCIONES))

    with open(RESULTADOS_PATH, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nArchivos generados: {REFERENCIA_CSV_PATH} y {RESULTADOS_PATH}")


if __name__ == "__main__":
    main()
