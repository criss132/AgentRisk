"""Comprueba que M1, guardado y recargado sin conexión, reproduce las predicciones de la corrida de entrenamiento.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.prototipo import (
    DEFAULT_CKPT_DIR,
    _predecir_severidad_con_tau,
    cargar_m1,
    cargar_umbrales,
    predecir,
)
from src.train_m1 import UMBRAL_SEGUNDA_ETIQUETA, seleccionar_etiquetas

DEV_PATH = os.path.join(REPO_ROOT, "data", "raw", "data", "dev.jsonl")
PREDS_REF_PATH = os.path.join(REPO_ROOT, "experiments", "m1_dev_predicciones.jsonl")

TOL_EXACTA = 1e-4
TOL_ACEPTABLE = 1e-2


def cargar_referencia_por_id(ruta_jsonl: str) -> dict[str, dict]:
    """Lee un JSONL y devuelve sus registros indexados por example_id."""
    ref = {}
    with open(ruta_jsonl, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            rec = json.loads(linea)
            ref[str(rec["example_id"])] = rec
    return ref


def decisiones_desde_referencia(ref: dict, umbrales: dict, classes_s2a: np.ndarray, classes_s2b: np.ndarray) -> dict:
    """Reconstruye las decisiones finales (severidad, S2a, S2b) a partir de las probabilidades de referencia."""
    probs_s1_ref = np.array(ref["probs_s1"], dtype=np.float64)
    tau_s1 = umbrales.get("s1_tau", None)
    sev_ref = int(_predecir_severidad_con_tau(probs_s1_ref, tau_s1))

    umbral_s2a = umbrales.get("s2a_second_label_threshold", UMBRAL_SEGUNDA_ETIQUETA)
    umbral_s2b = umbrales.get("s2b_second_label_threshold", UMBRAL_SEGUNDA_ETIQUETA)

    probs_s2a_vec = np.array([float(ref["probs_s2a"][str(cls)]) for cls in classes_s2a], dtype=np.float64)
    probs_s2b_vec = np.array([float(ref["probs_s2b"][str(cls)]) for cls in classes_s2b], dtype=np.float64)

    dom_ref = list(seleccionar_etiquetas(probs_s2a_vec, classes_s2a, umbral=umbral_s2a))
    fac_ref = list(seleccionar_etiquetas(probs_s2b_vec, classes_s2b, umbral=umbral_s2b))

    return {
        "severidad": sev_ref,
        "dominios": dom_ref,
        "factores": fac_ref,
    }


def main():
    """Recarga M1, repite las predicciones y compara con las guardadas."""
    if not os.path.exists(PREDS_REF_PATH):
        raise FileNotFoundError(f"No se encontró {PREDS_REF_PATH} para comparar.")

    dev_df = pd.read_json(DEV_PATH, lines=True).iloc[:5].reset_index(drop=True)
    ref_map = cargar_referencia_por_id(PREDS_REF_PATH)

    bundle = cargar_m1(ckpt_dir=DEFAULT_CKPT_DIR)
    umbrales = cargar_umbrales()
    classes_s2a = bundle["classes_s2a"]
    classes_s2b = bundle["classes_s2b"]

    print(f"Modelo recargado en {bundle['load_time_sec']}s ({bundle['device']})")
    print(
        f"Umbrales activos -> tau_S1={umbrales.get('s1_tau')}, "
        f"umbral_S2a={umbrales.get('s2a_second_label_threshold')}, "
        f"umbral_S2b={umbrales.get('s2b_second_label_threshold')}"
    )
    print(f"Tolerancias -> EXACTA <= {TOL_EXACTA:.0e} | ACEPTABLE <= {TOL_ACEPTABLE:.0e} (sin cambio de decisión)\n")

    max_diff_s1 = 0.0
    max_diff_s2a = 0.0
    max_diff_s2b = 0.0
    todos_ok = True
    niveles_ejemplo = []

    for _, row in dev_df.iterrows():
        ex_id = str(row["example_id"])
        if ex_id not in ref_map:
            print(f"[FALLO] example_id={ex_id} no encontrado en {PREDS_REF_PATH}")
            todos_ok = False
            continue

        ref = ref_map[ex_id]
        pred = predecir(row.to_dict(), bundle=bundle)
        dec_ref = decisiones_desde_referencia(ref, umbrales, classes_s2a, classes_s2b)

        # 1. Diferencia máxima en probabilidades
        p_s1_pred = np.array([pred["probabilidades"]["s1"][str(i + 1)] for i in range(5)], dtype=np.float64)
        p_s1_ref = np.array(ref["probs_s1"], dtype=np.float64)
        diff_s1 = float(np.max(np.abs(p_s1_pred - p_s1_ref)))

        diff_s2a = max(
            abs(float(pred["probabilidades"]["s2a"][str(cls)]) - float(ref["probs_s2a"][str(cls)]))
            for cls in classes_s2a
        )
        diff_s2b = max(
            abs(float(pred["probabilidades"]["s2b"][str(cls)]) - float(ref["probs_s2b"][str(cls)]))
            for cls in classes_s2b
        )

        diff_global = max(diff_s1, diff_s2a, diff_s2b)
        max_diff_s1 = max(max_diff_s1, diff_s1)
        max_diff_s2a = max(max_diff_s2a, diff_s2a)
        max_diff_s2b = max(max_diff_s2b, diff_s2b)

        # 2. Comparación de decisiones finales (severidad, S2a y S2b)
        sev_iguales = int(pred["severidad"]) == int(dec_ref["severidad"])
        dom_iguales = list(pred["dominios"]) == list(dec_ref["dominios"])
        fac_iguales = list(pred["factores"]) == list(dec_ref["factores"])
        decisiones_ok = sev_iguales and dom_iguales and fac_iguales

        if not decisiones_ok:
            nivel = "FALLO_DECISION"
            ok_ejemplo = False
        elif diff_global <= TOL_EXACTA:
            nivel = "EXACTA"
            ok_ejemplo = True
        elif diff_global <= TOL_ACEPTABLE:
            nivel = "ACEPTABLE"
            ok_ejemplo = True
        else:
            nivel = "FALLO_PROB"
            ok_ejemplo = False

        todos_ok = todos_ok and ok_ejemplo
        niveles_ejemplo.append(nivel)

        print(
            f"example_id={ex_id} | estado={nivel} | "
            f"max_diff (S1={diff_s1:.2e}, S2a={diff_s2a:.2e}, S2b={diff_s2b:.2e}) | "
            f"sev_ok={sev_iguales}, dom_ok={dom_iguales}, fac_ok={fac_iguales}"
        )
        if not decisiones_ok:
            print(
                f"   -> Decisión predicha: sev={pred['severidad']}, dom={pred['dominios']}, fac={pred['factores']}\n"
                f"   -> Decisión ref     : sev={dec_ref['severidad']}, dom={dec_ref['dominios']}, fac={dec_ref['factores']}"
            )

    print(
        f"\nResumen 5 ejemplos -> max_diff S1={max_diff_s1:.2e}, "
        f"S2a={max_diff_s2a:.2e}, S2b={max_diff_s2b:.2e} | estados={niveles_ejemplo}"
    )

    if not todos_ok:
        raise AssertionError(
            "La prueba de recarga falló: hay cambio en decisiones finales o diferencia de probabilidades fuera de TOL_ACEPTABLE."
        )

    print("PRUEBA DE RECARGA SUPERADA: decisiones y diferencias de probabilidad dentro de los márgenes definidos.")


if __name__ == "__main__":
    main()