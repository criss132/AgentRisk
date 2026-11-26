"""Línea base L0: mediana de severidad (S1) y etiqueta más frecuente (S2) aprendidas en train.

Se evalúa en dev y los resultados se guardan en experiments/l0_resultados.json.
"""
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Constantes del proyecto
SEMILLA = 42
TRAIN_PATH = "data/raw/data/train.jsonl"
DEV_PATH = "data/raw/data/dev.jsonl"
OUTPUT_DIR = "experiments"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "l0_resultados.json")

def jaccard(real, pred):
    """
    Calcula el índice de Jaccard entre dos listas de etiquetas convirtiéndolas a conjuntos.
    Fórmula: |Intersección| / |Unión|
    """
    set_real = set(real)
    set_pred = set(pred)
    
    if len(set_real) == 0 and len(set_pred) == 0:
        return 1.0
    
    intersec = len(set_real.intersection(set_pred))
    union = len(set_real.union(set_pred))
    
    if union == 0:
        return 0.0
    
    return intersec / union

def main():
    """Calcula L0 con train, lo evalúa en dev y guarda los resultados."""
    print("=== EJECUCIÓN DEL MODELO BASE L0 ===")
    
    # --- 1. CARGAR DATOS ---
    # Nota: Ejecutar siempre desde la raíz del repositorio
    train = pd.read_json(TRAIN_PATH, lines=True)
    dev = pd.read_json(DEV_PATH, lines=True)
    
    # --- 2. ENTRENAR (CALCULAR L0) USANDO SOLO TRAIN ---
    # S1: Mediana de risk_severity
    l0_s1 = train['risk_severity'].median()
    
    # S2a: Etiqueta más frecuente en risk_domains (usando explode)
    l0_s2a = train['risk_domains'].explode().value_counts().index[0]
    
    # S2b: Etiqueta más frecuente en risk_drivers (usando explode)
    l0_s2b = train['risk_drivers'].explode().value_counts().index[0]
    
    print(f"\n[L0 Aprendido de Train]")
    print(f" - S1 (Mediana severidad): {l0_s1}")
    print(f" - S2a (Dominio más frecuente): {l0_s2a}")
    print(f" - S2b (Driver más frecuente): {l0_s2b}")
    
    # --- PRUEBA MANUAL DE LA FUNCIÓN JACCARD ---
    print("\n--- Verificando función Jaccard ---")
    test_1 = jaccard(["A"], ["A"])
    test_2 = jaccard(["A", "B"], ["A"])
    print(f"Prueba 1 (['A'] vs ['A']) -> Resultado: {test_1} (Esperado: 1.0)")
    print(f"Prueba 2 (['A', 'B'] vs ['A']) -> Resultado: {test_2} (Esperado: 0.5)")
    assert test_1 == 1.0 and test_2 == 0.5, "¡Error en la implementación de Jaccard!"
    print("¡Función Jaccard verificada con éxito!\n")
    
    # --- 3. EVALUAR EN DEV ---
    # S1 (MAE): |mediana - real| promediado
    dev['pred_s1'] = l0_s1
    mae_s1 = (dev['risk_severity'] - dev['pred_s1']).abs().mean()
    
    # S2a y S2b (Jaccard medio): comparar cada lista real contra la predicción constante [l0_s2...]
    pred_list_s2a = [l0_s2a]
    pred_list_s2b = [l0_s2b]
    
    jaccard_s2a = dev['risk_domains'].apply(lambda x: jaccard(x, pred_list_s2a)).mean()
    jaccard_s2b = dev['risk_drivers'].apply(lambda x: jaccard(x, pred_list_s2b)).mean()
    
    # Subestimación severa: Entre los ejemplos con severidad real >= 4, fracción con predicción <= 2
    severe_mask = dev['risk_severity'] >= 4
    if severe_mask.sum() > 0:
        underestimation_rate = (dev.loc[severe_mask, 'pred_s1'] <= 2).mean() * 100
    else:
        underestimation_rate = 0.0

    # --- 4. GUARDAR RESULTADOS ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    resultados = {
        "model": "L0",
        "seed": SEMILLA,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "s1_mae": round(float(mae_s1), 4),
            "s2a_jaccard": round(float(jaccard_s2a), 4),
            "s2b_jaccard": round(float(jaccard_s2b), 4),
            "severe_underestimation_pct": round(float(underestimation_rate), 2)
        }
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
        
    print(f"Resultados guardados correctamente en: {OUTPUT_FILE}")
    
    # --- IMPRIMIR TABLA CORTA EN PANTALLA ---
    print("\n" + "="*50)
    print(" RESUMEN DE EVALUACIÓN - MODELO L0 (DEV) ")
    print("="*50)
    print(f"{'Métrica':<35} | {'Valor':<10}")
    print("-"*50)
    print(f"{'S1 - MAE (Severidad)':<35} | {mae_s1:.3f}")
    print(f"{'S2a - Jaccard Medio (Dominios)':<35} | {jaccard_s2a:.3f}")
    print(f"{'S2b - Jaccard Medio (Drivers)':<35} | {jaccard_s2b:.3f}")
    print(f"{'Subestimación Severa (Real >= 4)':<35} | {underestimation_rate:.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()