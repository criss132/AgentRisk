"""Reproduce de principio a fin lo que no necesita GPU ni los pesos de M1.

Descarga los datos (si faltan), crea las particiones, corre las líneas base L0 y L1,
la validación cruzada y la selección de umbrales. Usa las predicciones de M1 ya
guardadas en experiments/. Se ejecuta desde la raíz: python src/reproducir.py
"""
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data" / "raw" / "data" / "train.jsonl"

PASOS = [
    "src/make_splits.py",
    "src/baseline_L0.py",
    "src/baseline_L1.py",
    "src/baseline_L1_s2.py",
    "src/baselines_cv.py",
    "src/seleccionar_umbrales.py",
]


def main():
    """Ejecuta cada paso en orden y se detiene si alguno falla."""
    if not DATOS.exists():
        PASOS.insert(0, "data/download_data.py")
    for paso in PASOS:
        print(f"\n=== {paso} ===", flush=True)
        subprocess.run([sys.executable, paso], cwd=RAIZ, check=True)
    print("\nListo: los resultados quedaron en experiments/.")


if __name__ == "__main__":
    main()
