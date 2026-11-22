import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoConfig, AutoTokenizer

# Asegurar que el directorio raíz o src sea importable al ejecutar el script directamente
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.train_m1 import (
    DebertaMultiTask,
    MAX_LEN_DEFAULT,
    SEED_DEFAULT,
    TRAIN_PATH,
    UMBRAL_SEGUNDA_ETIQUETA,
    seleccionar_etiquetas,
)
from src.s3_politicas import POLITICAS, aplicar_politica

EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
DEFAULT_CKPT_DIR = os.path.join(EXPERIMENTS_DIR, "checkpoints", "m1_dev_seed42")
UMBRALES_FILE = os.path.join(EXPERIMENTS_DIR, "umbrales_resultados.json")

PARAMS_VECTORIZER = {
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "max_features": 10000,
    "min_df": 2,
}

PARAMS_ESTIMATOR = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": SEED_DEFAULT,
}


def formatear_texto_caso(texto_o_registro: str | dict) -> str:
    """
    Construye el texto de entrada reutilizando la lógica/estructura del proyecto
    sin duplicar reglas de formato.
    """
    if isinstance(texto_o_registro, dict):
        if "formatted_interaction" in texto_o_registro and texto_o_registro["formatted_interaction"]:
            return str(texto_o_registro["formatted_interaction"])
        # Si se pasa un diccionario con campos sueltos, serializar de forma consistente
        partes = []
        for clave in ("context", "history", "user", "assistant", "proposed_action", "action", "prompt", "text"):
            if clave in texto_o_registro and texto_o_registro[clave]:
                partes.append(f"{clave}: {texto_o_registro[clave]}")
        if partes:
            return "\n".join(partes)
        return json.dumps(texto_o_registro, ensure_ascii=False)
    return str(texto_o_registro)


def cargar_umbrales(ruta_umbrales: str = UMBRALES_FILE, modelo: str = "M1") -> dict:
    """Lee los umbrales elegidos para `modelo` ("M1" o "L1") de umbrales_resultados.json o usa valores por defecto."""
    umbrales = {
        "s2a_second_label_threshold": UMBRAL_SEGUNDA_ETIQUETA,
        "s2b_second_label_threshold": UMBRAL_SEGUNDA_ETIQUETA,
        "s1_tau": None,
    }
    if not os.path.exists(ruta_umbrales):
        return umbrales

    with open(ruta_umbrales, "r", encoding="utf-8") as f:
        data = json.load(f)

    elegidos = data.get("models", {}).get(modelo, {}).get("chosen_thresholds", {})

    if "s2a_second_label_threshold" in elegidos:
        umbrales["s2a_second_label_threshold"] = elegidos["s2a_second_label_threshold"]
    if "s2b_second_label_threshold" in elegidos:
        umbrales["s2b_second_label_threshold"] = elegidos["s2b_second_label_threshold"]
    if "s1_tau" in elegidos:
        umbrales["s1_tau"] = elegidos["s1_tau"]

    return umbrales


def cargar_m1(ckpt_dir: str = DEFAULT_CKPT_DIR, device: torch.device | None = None) -> dict:
    """
    Carga encoder_config, tokenizer, label_classes.json y model_weights.pt sin conexión.
    Deja el modelo en modo eval(). Mide y registra el tiempo de carga.
    """
    t0 = time.perf_counter()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_dir = os.path.join(ckpt_dir, "encoder_config")
    tokenizer_dir = os.path.join(ckpt_dir, "tokenizer")
    label_meta_path = os.path.join(ckpt_dir, "label_classes.json")
    weights_path = os.path.join(ckpt_dir, "model_weights.pt")

    for p in (config_dir, tokenizer_dir, label_meta_path, weights_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Falta artefacto necesario para cargar M1: {p}")

    with open(label_meta_path, "r", encoding="utf-8") as f:
        label_meta = json.load(f)

    classes_s2a = np.array(label_meta["s2a_domains"])
    classes_s2b = np.array(label_meta["s2b_drivers"])
    num_s1 = int(label_meta.get("num_s1", 5))

    config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        truncation_side="left",
    )
    tokenizer.truncation_side = "left"

    # Instanciar arquitectura con config local y cargar state_dict guardado
    model = DebertaMultiTask(
        model_name=getattr(config, "_name_or_path", "microsoft/deberta-v3-base"),
        num_s1=num_s1,
        num_s2a=len(classes_s2a),
        num_s2b=len(classes_s2b),
        config=config,
        cargar_preentrenado=False,
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    load_time_sec = round(time.perf_counter() - t0, 4)
    umbrales = cargar_umbrales(modelo="M1")

    return {
        "backend": "M1",
        "device": str(device),
        "load_time_sec": load_time_sec,
        "model": model,
        "tokenizer": tokenizer,
        "classes_s2a": classes_s2a,
        "classes_s2b": classes_s2b,
        "umbrales": umbrales,
    }


def construir_respaldo_l1(train_path: str = TRAIN_PATH) -> dict:
    """
    Respaldo en L1: entrena TF-IDF + LogisticRegression / OneVsRest sobre train.jsonl
    cuando falla la carga del modelo M1.
    """
    t0 = time.perf_counter()
    ruta_train = train_path if os.path.isabs(train_path) else os.path.join(REPO_ROOT, train_path)
    train_df = pd.read_json(ruta_train, lines=True)

    X_train = train_df["formatted_interaction"].astype(str)
    y_s1 = train_df["risk_severity"].to_numpy()

    pipe_s1 = Pipeline([
        ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
        ("clf", LogisticRegression(**PARAMS_ESTIMATOR)),
    ])
    pipe_s1.fit(X_train, y_s1)

    mlb_s2a = MultiLabelBinarizer()
    Y_s2a = mlb_s2a.fit_transform(train_df["risk_domains"])
    pipe_s2a = Pipeline([
        ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
        ("clf", OneVsRestClassifier(LogisticRegression(**PARAMS_ESTIMATOR))),
    ])
    pipe_s2a.fit(X_train, Y_s2a)

    mlb_s2b = MultiLabelBinarizer()
    Y_s2b = mlb_s2b.fit_transform(train_df["risk_drivers"])
    pipe_s2b = Pipeline([
        ("tfidf", TfidfVectorizer(**PARAMS_VECTORIZER)),
        ("clf", OneVsRestClassifier(LogisticRegression(**PARAMS_ESTIMATOR))),
    ])
    pipe_s2b.fit(X_train, Y_s2b)

    load_time_sec = round(time.perf_counter() - t0, 4)
    umbrales = cargar_umbrales(modelo="L1")

    return {
        "backend": "L1_fallback",
        "device": "cpu",
        "load_time_sec": load_time_sec,
        "pipe_s1": pipe_s1,
        "pipe_s2a": pipe_s2a,
        "pipe_s2b": pipe_s2b,
        "classes_s1": list(pipe_s1.classes_),
        "classes_s2a": mlb_s2a.classes_,
        "classes_s2b": mlb_s2b.classes_,
        "umbrales": umbrales,
    }


def cargar_con_respaldo(ckpt_dir: str = DEFAULT_CKPT_DIR, device: torch.device | None = None) -> dict:
    """Intenta cargar M1; si ocurre cualquier error, activa el respaldo en L1."""
    try:
        return cargar_m1(ckpt_dir=ckpt_dir, device=device)
    except Exception as exc:
        print(f"[Aviso] Falló cargar_m1 ({exc}). Activando respaldo en L1...")
        return construir_respaldo_l1()


def _predecir_severidad_con_tau(probs_s1: np.ndarray, tau: float | None) -> int:
    if tau is not None and len(probs_s1) >= 5:
        p_alta = float(probs_s1[3] + probs_s1[4])
        if p_alta > float(tau):
            return 4 if probs_s1[3] >= probs_s1[4] else 5
        return int(np.argmax(probs_s1[:3]) + 1)
    return int(np.argmax(probs_s1) + 1)


def predecir(
    texto: str | dict,
    bundle: dict,
    max_len: int = MAX_LEN_DEFAULT,
) -> dict:
    """
    Tokeniza con truncamiento izquierdo y max_len=512 (en M1), evalúa sin gradiente
    y aplica los umbrales de umbrales_resultados.json.
    Devuelve severidad, dominios, factores y probabilidades.
    """
    texto_formateado = formatear_texto_caso(texto)
    umbrales = bundle.get("umbrales", {})
    umbral_s2a = umbrales.get("s2a_second_label_threshold", UMBRAL_SEGUNDA_ETIQUETA)
    umbral_s2b = umbrales.get("s2b_second_label_threshold", UMBRAL_SEGUNDA_ETIQUETA)
    tau_s1 = umbrales.get("s1_tau", None)

    if bundle.get("backend") == "M1":
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
        device = torch.device(bundle["device"])
        classes_s2a = bundle["classes_s2a"]
        classes_s2b = bundle["classes_s2b"]

        tokenizer.truncation_side = "left"
        encoded = tokenizer(
            texto_formateado,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        model.eval()
        with torch.no_grad():
            logits_s1, logits_s2a, logits_s2b = model(input_ids, attention_mask)
            probs_s1 = torch.softmax(logits_s1.float(), dim=-1).squeeze(0).cpu().numpy()
            probs_s2a = torch.sigmoid(logits_s2a.float()).squeeze(0).cpu().numpy()
            probs_s2b = torch.sigmoid(logits_s2b.float()).squeeze(0).cpu().numpy()

        severidad = _predecir_severidad_con_tau(probs_s1, tau_s1)
        dominios = seleccionar_etiquetas(probs_s2a, classes_s2a, umbral=umbral_s2a)
        factores = seleccionar_etiquetas(probs_s2b, classes_s2b, umbral=umbral_s2b)

        return {
            "backend": "M1",
            "severidad": int(severidad),
            "dominios": list(dominios),
            "factores": list(factores),
            "probabilidades": {
                "s1": {str(i + 1): round(float(probs_s1[i]), 6) for i in range(len(probs_s1))},
                "s2a": {str(cls): round(float(probs_s2a[j]), 6) for j, cls in enumerate(classes_s2a)},
                "s2b": {str(cls): round(float(probs_s2b[j]), 6) for j, cls in enumerate(classes_s2b)},
            },
        }

    # Respaldo L1
    pipe_s1 = bundle["pipe_s1"]
    pipe_s2a = bundle["pipe_s2a"]
    pipe_s2b = bundle["pipe_s2b"]
    classes_s1 = bundle["classes_s1"]
    classes_s2a = bundle["classes_s2a"]
    classes_s2b = bundle["classes_s2b"]

    raw_s1 = pipe_s1.predict_proba([texto_formateado])[0]
    probs_s1 = np.zeros(5, dtype=np.float64)
    for idx_c, c_val in enumerate(classes_s1):
        if int(c_val) in (1, 2, 3, 4, 5):
            probs_s1[int(c_val) - 1] = float(raw_s1[idx_c])
    if probs_s1.sum() == 0:
        probs_s1 = np.full(5, 0.2, dtype=np.float64)

    probs_s2a = pipe_s2a.predict_proba([texto_formateado])[0]
    probs_s2b = pipe_s2b.predict_proba([texto_formateado])[0]

    severidad = _predecir_severidad_con_tau(probs_s1, tau_s1)
    dominios = seleccionar_etiquetas(probs_s2a, classes_s2a, umbral=umbral_s2a)
    factores = seleccionar_etiquetas(probs_s2b, classes_s2b, umbral=umbral_s2b)

    return {
        "backend": "L1_fallback",
        "severidad": int(severidad),
        "dominios": list(dominios),
        "factores": list(factores),
        "probabilidades": {
            "s1": {str(i + 1): round(float(probs_s1[i]), 6) for i in range(5)},
            "s2a": {str(cls): round(float(probs_s2a[j]), 6) for j, cls in enumerate(classes_s2a)},
            "s2b": {str(cls): round(float(probs_s2b[j]), 6) for j, cls in enumerate(classes_s2b)},
        },
    }


def imprimir_resultado_legible(resultado: dict, titulo: str | None = None, politica: str | None = None):
    if titulo:
        print(f"\n--- {titulo} ---")
    print(f"Modelo activo : {resultado['backend']}")
    print(f"Severidad (1-5): {resultado['severidad']}")
    print(f"Dominios (S2a) : {', '.join(resultado['dominios'])}")
    print(f"Factores (S2b) : {', '.join(resultado['factores'])}")
    top_s2a = sorted(resultado["probabilidades"]["s2a"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_s2b = sorted(resultado["probabilidades"]["s2b"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    print(f"Top-3 prob S2a : {top_s2a}")
    print(f"Top-3 prob S2b : {top_s2b}")
    print(f"Probs S1       : {resultado['probabilidades']['s1']}")
    if politica:
        intervencion = aplicar_politica(politica, resultado["severidad"], resultado["dominios"], resultado["factores"])
        print(f"Intervención (política {politica}): {intervencion.upper()}")


def leer_casos_desde_archivo(ruta_archivo: str) -> list[tuple[str, str | dict]]:
    """Lee casos desde .jsonl, .json o texto plano."""
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {ruta_archivo}")

    if ruta_archivo.endswith(".jsonl"):
        casos = []
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            for i, linea in enumerate(f):
                linea = linea.strip()
                if not linea:
                    continue
                obj = json.loads(linea)
                eid = str(obj.get("example_id", f"caso_{i+1}"))
                casos.append((eid, obj))
        return casos

    if ruta_archivo.endswith(".json"):
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [(str(item.get("example_id", f"caso_{i+1}")), item) for i, item in enumerate(data)]
        if isinstance(data, dict):
            return [(str(data.get("example_id", "caso_1")), data)]

    with open(ruta_archivo, "r", encoding="utf-8") as f:
        contenido = f.read().strip()
    return [("archivo_texto", contenido)]


def medir_tiempos_carga(ckpt_dir: str = DEFAULT_CKPT_DIR) -> dict:
    """Mide el tiempo de carga de M1 en CPU y, si hay GPU disponible, también en CUDA."""
    tiempos = {}
    bundle_cpu = cargar_m1(ckpt_dir=ckpt_dir, device=torch.device("cpu"))
    tiempos["cpu_sec"] = bundle_cpu["load_time_sec"]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        bundle_gpu = cargar_m1(ckpt_dir=ckpt_dir, device=torch.device("cuda"))
        torch.cuda.synchronize()
        tiempos["cuda_sec"] = bundle_gpu["load_time_sec"]
    else:
        tiempos["cuda_sec"] = None

    return tiempos


def parse_args():
    parser = argparse.ArgumentParser(description="Prototipo de inferencia offline para M1 (con respaldo L1)")
    parser.add_argument("--archivo", type=str, default=None, help="Ruta a archivo (.jsonl, .json o .txt) con casos")
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR, help="Directorio del checkpoint de M1")
    parser.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT, help="Longitud máxima de tokens (512)")
    parser.add_argument("--politica", choices=sorted(POLITICAS), default=None, help="Política de S3 (A o B): además del riesgo, imprime la intervención (permitir, confirmar o bloquear)")
    parser.add_argument("--medir-tiempos", action="store_true", help="Mide tiempo de carga en CPU y GPU")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.medir_tiempos:
        tiempos = medir_tiempos_carga(ckpt_dir=args.ckpt_dir)
        print(f"Tiempo de carga M1 -> CPU: {tiempos['cpu_sec']}s | GPU: {tiempos['cuda_sec']}s")
        return

    bundle = cargar_con_respaldo(ckpt_dir=args.ckpt_dir)
    print(f"Cargado [{bundle['backend']}] en {bundle['device']} en {bundle['load_time_sec']}s")

    if args.archivo:
        casos = leer_casos_desde_archivo(args.archivo)
        for eid, caso in casos:
            res = predecir(caso, bundle=bundle, max_len=args.max_len)
            imprimir_resultado_legible(res, titulo=f"Caso {eid}", politica=args.politica)
        return

    # Modo interactivo por consola
    print("\nModo interactivo (escribe 'salir' o deja una línea en blanco para terminar):")
    idx = 1
    while True:
        try:
            entrada = input(f"\nInteracción #{idx}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break
        if not entrada or entrada.lower() in {"salir", "exit", "quit"}:
            break
        res = predecir(entrada, bundle=bundle, max_len=args.max_len)
        imprimir_resultado_legible(res, titulo=f"Resultado #{idx}", politica=args.politica)
        idx += 1


if __name__ == "__main__":
    main()