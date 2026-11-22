"""
Mide cuánto tarda el prototipo: carga de M1 y predicción por caso, en CPU y, si hay, en GPU.

Se ejecuta desde la raíz del repositorio, con los pesos de M1 en experiments/checkpoints/m1_dev_seed42/:

    python src/medir_tiempos.py            # 30 casos de dev, 3 de calentamiento
    python src/medir_tiempos.py --n 50

Escribe experiments/prototipo_tiempos.json. Cada caso se predice de a uno (como en la demo),
sin agrupar en lotes, y la primera predicción se descarta como calentamiento.
"""
import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.prototipo import DEFAULT_CKPT_DIR, cargar_m1, predecir

DEV_PATH = os.path.join(REPO_ROOT, "data", "raw", "data", "dev.jsonl")
SALIDA_PATH = os.path.join(REPO_ROOT, "experiments", "prototipo_tiempos.json")


def leer_textos_dev(n: int) -> list[str]:
    textos = []
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                textos.append(str(json.loads(linea)["formatted_interaction"]))
            if len(textos) == n:
                break
    return textos


def sincronizar(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def medir_dispositivo(device: torch.device, calentar: list[str], textos: list[str], ckpt_dir: str) -> dict:
    bundle = cargar_m1(ckpt_dir=ckpt_dir, device=device)
    for t in calentar:
        predecir(t, bundle=bundle)
    sincronizar(device)

    latencias = []
    for t in textos:
        t0 = time.perf_counter()
        predecir(t, bundle=bundle)
        sincronizar(device)
        latencias.append(time.perf_counter() - t0)

    latencias_ordenadas = sorted(latencias)
    p95 = latencias_ordenadas[min(len(latencias) - 1, int(round(0.95 * (len(latencias) - 1))))]
    resultado = {
        "dispositivo": str(device),
        "carga_sec": bundle["load_time_sec"],
        "casos_medidos": len(latencias),
        "prediccion_mediana_sec": round(statistics.median(latencias), 4),
        "prediccion_media_sec": round(statistics.mean(latencias), 4),
        "prediccion_p95_sec": round(p95, 4),
        "prediccion_max_sec": round(max(latencias), 4),
    }
    del bundle
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return resultado


def main():
    ap = argparse.ArgumentParser(description="Tiempos de carga y predicción del prototipo (M1).")
    ap.add_argument("--n", type=int, default=30, help="Casos de dev a predecir en cada dispositivo.")
    ap.add_argument("--calentamiento", type=int, default=3, help="Predicciones previas que no se cuentan.")
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    args = ap.parse_args()

    textos = leer_textos_dev(args.n + args.calentamiento)
    medir = textos[args.calentamiento:]
    calentar = textos[:args.calentamiento]
    print(f"Casos: {len(medir)} medidos + {len(calentar)} de calentamiento (dev, entrada truncada a 512 tokens)")

    equipo = {
        "sistema": platform.platform(),
        "procesador": platform.processor(),
        "hilos_torch_cpu": torch.get_num_threads(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    resultados = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "equipo": equipo, "mediciones": []}

    dispositivos = [torch.device("cpu")]
    if torch.cuda.is_available():
        dispositivos.append(torch.device("cuda"))
    for dev in dispositivos:
        print(f"\nMidiendo en {dev}...")
        r = medir_dispositivo(dev, calentar, medir, args.ckpt_dir)
        resultados["mediciones"].append(r)
        print(f"  carga: {r['carga_sec']} s | predicción por caso: mediana {r['prediccion_mediana_sec']} s, "
              f"p95 {r['prediccion_p95_sec']} s, máx {r['prediccion_max_sec']} s")

    os.makedirs(os.path.dirname(SALIDA_PATH), exist_ok=True)
    with open(SALIDA_PATH, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nArchivo generado: {SALIDA_PATH}")


if __name__ == "__main__":
    main()
