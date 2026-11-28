"""Crea los 5 folds de validación cruzada, agrupados por scenario_group y estratificados por severidad.

Resultado: data/splits/train_folds.csv (example_id y fold).
"""
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Constantes
SEMILLA = 42
N_FOLDS = 5

def main():
    # 1. Cargar train y el manifest, y unir el scenario_group por example_id
    """Genera los folds, los guarda y muestra comprobaciones de calidad (sin fuga entre folds)."""
    print("Cargando datos...")
    train = pd.read_json("data/raw/data/train.jsonl", lines=True)
    manifest = pd.read_json("data/raw/metadata/split_manifest.jsonl", lines=True)
    
    # Unir el scenario_group desde el manifest
    train_with_groups = train.merge(
        manifest[['example_id', 'scenario_group']], 
        on='example_id', 
        how='inner'
    )
    
    print(f"Train cargado con {len(train_with_groups)} ejemplos.")

    # 2. Asignar cada ejemplo a uno de 5 folds usando StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    
    train_with_groups['fold'] = -1
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(
        X=train_with_groups, 
        y=train_with_groups['risk_severity'], 
        groups=train_with_groups['scenario_group']
    )):
        train_with_groups.loc[val_idx, 'fold'] = fold_idx

    # 3. Guardar el resultado en data/splits/ con example_id y fold
    os.makedirs("data/splits", exist_ok=True)
    
    splits_df = train_with_groups[['example_id', 'fold']]
    output_path = "data/splits/train_folds.csv"
    splits_df.to_csv(output_path, index=False)
    print(f"Folds guardados exitosamente en: {output_path}")

    # 4. Imprimir al final el tamaño de cada fold
    print("\n--- TAMAÑO DE CADA FOLD ---")
    print(splits_df['fold'].value_counts().sort_index())

    # --- COMPROBACIONES DE CALIDAD ---
    print("\n" + "="*40)
    print("COMPROBACIONES DE CALIDAD DEL SPLIT")
    print("="*40)

    # Comprobación 1: ¿Algún scenario_group aparece en más de un fold? (Debe dar 0)
    group_fold_counts = train_with_groups.groupby('scenario_group')['fold'].nunique()
    groups_in_multiple_folds = (group_fold_counts > 1).sum()
    print(f"1. Grupos en más de un fold (debe ser 0): {groups_in_multiple_folds}")

    # Comprobación 2: Tamaños de los folds
    print("\n2. Tamaños aproximados (ideal cerca de 114 por fold):")
    print(splits_df['fold'].value_counts().sort_index().to_dict())

    # Comprobación 3: Distribución de severidad entre folds
    print("\n3. Distribución porcentual de risk_severity por fold:")
    severity_distribution = pd.crosstab(
        train_with_groups['fold'], 
        train_with_groups['risk_severity'], 
        normalize='index'
    ) * 100
    print(severity_distribution.round(2))

if __name__ == "__main__":
    main()