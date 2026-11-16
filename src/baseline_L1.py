import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Constantes y rutas
SEMILLA = 42
TRAIN_PATH = "data/raw/data/train.jsonl"
DEV_PATH = "data/raw/data/dev.jsonl"
OUTPUT_DIR = "experiments"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "l1_s1_resultados.json")

def calcular_subestimacion(y_real, y_pred):
    """Calcula la fracción de ejemplos con severidad real >= 4 que tienen predicción <= 2."""
    severe_mask = y_real >= 4
    if severe_mask.sum() > 0:
        return float((y_pred[severe_mask] <= 2).mean() * 100)
    return 0.0

def main():
    print("=== EJECUCIÓN DEL MODELO BASE L1 (S1: Severidad) ===")
    
    # 2. Cargar y separar X, y
    print("Cargando conjuntos train y dev...")
    train = pd.read_json(TRAIN_PATH, lines=True)
    dev = pd.read_json(DEV_PATH, lines=True)
    
    X_train = train['formatted_interaction']
    y_train = train['risk_severity']
    
    X_dev = dev['formatted_interaction']
    y_dev = dev['risk_severity']
    
    # Definición de hiperparámetros base
    params_vectorizer = {
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "max_features": 10000,
        "min_df": 2
    }
    
    # Probaremos dos configuraciones para la regresión logística
    configuraciones = [
        {"name": "L1_standard", "class_weight": None},
        {"name": "L1_balanced", "class_weight": "balanced"}
    ]
    
    resultados_experimentos = {}
    
    for config in configuraciones:
        nombre_config = config["name"]
        print(f"\n--- Entrenando modelo: {nombre_config} ---")
        
        # 3. Armar el pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**params_vectorizer)),
            ('clf', LogisticRegression(
                max_iter=1000, 
                random_state=SEMILLA, 
                class_weight=config["class_weight"]
            ))
        ])
        
        # Ajustar exclusivamente con train (evita data leakage)
        pipeline.fit(X_train, y_train)
        
        # 4. Predecir y medir en dev
        y_pred = pipeline.predict(X_dev)
        
        # Métricas
        mae_s1 = float((y_dev - y_pred).abs().mean())
        subestimacion_pct = calcular_subestimacion(y_dev.values, y_pred)
        
        # Distribución de predicciones (cuántas veces predice cada clase)
        val_counts = pd.Series(y_pred).value_counts().sort_index().to_dict()
        # Asegurar que todas las clases de 1 a 5 aparezcan en el registro
        distribucion_pred = {int(k): int(v) for k, v in val_counts.items()}
        
        print(f"  > MAE S1: {mae_s1:.4f}")
        print(f"  > Subestimación Severa (Real >= 4, Pred <= 2): {subestimacion_pct:.2f}%")
        print(f"  > Distribución de predicciones: {distribucion_pred}")
        
        # Guardar en diccionario temporal
        resultados_experimentos[nombre_config] = {
            "model": "L1",
            "task": "S1_severity",
            "seed": SEMILLA,
            "hyperparameters": {
                "vectorizer": params_vectorizer,
                "classifier": {
                    "max_iter": 1000,
                    "class_weight": config["class_weight"]
                }
            },
            "metrics": {
                "s1_mae": round(mae_s1, 4),
                "severe_underestimation_pct": round(subestimacion_pct, 2)
            },
            "prediction_distribution": distribucion_pred
        }
        
    # 6. Guardar todo en experiments/l1_s1_resultados.json
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": resultados_experimentos
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"\nResultados guardados exitosamente en: {OUTPUT_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()