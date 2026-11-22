import os

# 1. Restringir a una sola GPU antes de importar torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import math
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    __version__ as transformers_version,
)

# Constantes y rutas
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN_DEFAULT = 512
SEED_DEFAULT = 42
BATCH_SIZE_DEFAULT = 4
GRAD_ACCUM_DEFAULT = 4
LR_DEFAULT = 2e-5
EPOCHS_DEFAULT = 9
WARMUP_RATIO = 0.10
UMBRAL_SEGUNDA_ETIQUETA = 0.3

TRAIN_PATH = "data/raw/data/train.jsonl"
DEV_PATH = "data/raw/data/dev.jsonl"
OUTPUT_DIR = "experiments"
METRICS_FILE = os.path.join(OUTPUT_DIR, "m1_resultados.json")
PREDS_FILE = os.path.join(OUTPUT_DIR, "m1_dev_predicciones.jsonl")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jaccard(real, pred):
    set_real = set(real)
    set_pred = set(pred)
    if len(set_real) == 0 and len(set_pred) == 0:
        return 1.0
    union = len(set_real.union(set_pred))
    if union == 0:
        return 0.0
    return len(set_real.intersection(set_pred)) / union


def seleccionar_etiquetas(probs_fila, classes, umbral=UMBRAL_SEGUNDA_ETIQUETA):
    indices_ordenados = np.argsort(probs_fila)[::-1]
    elegidos_idx = [indices_ordenados[0]]
    if len(indices_ordenados) > 1:
        segundo_idx = indices_ordenados[1]
        if probs_fila[segundo_idx] >= umbral:
            elegidos_idx.append(segundo_idx)
    return [classes[idx] for idx in elegidos_idx]


def verificar_truncamiento_izquierda(texts, tokenizer, max_len):
    """Toma el texto más largo y verifica que tras truncar conserva el final."""
    textos_lista = list(texts)
    texto_largo = max(textos_lista, key=lambda t: len(str(t)))
    ids_truncados = tokenizer(
        texto_largo,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )["input_ids"]
    decodificado = tokenizer.decode(ids_truncados, skip_special_tokens=True)

    print("\n--- Verificación de truncamiento por la izquierda ---")
    print(f"Longitud original (chars): {len(texto_largo)} | max_len: {max_len}")
    print(f"Final del texto original (-180 chars): {repr(texto_largo[-180:])}")
    print(f"Final decodificado tras truncar (-180 chars): {repr(decodificado[-180:])}")
    print(f"Inicio decodificado tras truncar (+120 chars): {repr(decodificado[:120])}\n")


class MultiTaskDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, y_s1=None, y_s2a=None, y_s2b=None):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.y_s1 = y_s1
        self.y_s2a = y_s2a
        self.y_s2b = y_s2b

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.y_s1 is not None:
            item["label_s1"] = torch.tensor(self.y_s1[idx], dtype=torch.long)
            item["label_s2a"] = torch.tensor(self.y_s2a[idx], dtype=torch.float32)
            item["label_s2b"] = torch.tensor(self.y_s2b[idx], dtype=torch.float32)
        return item


class DebertaMultiTask(nn.Module):
    def __init__(self, model_name, num_s1=5, num_s2a=11, num_s2b=7, dropout_prob=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout_prob)
        self.head_s1 = nn.Linear(hidden_size, num_s1)
        self.head_s2a = nn.Linear(hidden_size, num_s2a)
        self.head_s2b = nn.Linear(hidden_size, num_s2b)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits_s1 = self.head_s1(cls_repr)
        logits_s2a = self.head_s2a(cls_repr)
        logits_s2b = self.head_s2b(cls_repr)
        return logits_s1, logits_s2a, logits_s2b


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento multi-tarea M1 con DeBERTa-v3-base")
    parser.add_argument("--rapido", action="store_true", help="Modo rápido: 64 ejemplos, 1 época, 128 tokens")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_DEFAULT)
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    return parser.parse_args()


def evaluate_dev(model, loader, device, classes_s2a, classes_s2b, y_dev_s1_orig, y_dev_s2a_listas, y_dev_s2b_listas):
    model.eval()
    probs_s1_list, probs_s2a_list, probs_s2b_list = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits_s1, logits_s2a, logits_s2b = model(input_ids, attention_mask)
            p_s1 = torch.softmax(logits_s1.float(), dim=-1).cpu().numpy()
            p_s2a = torch.sigmoid(logits_s2a.float()).cpu().numpy()
            p_s2b = torch.sigmoid(logits_s2b.float()).cpu().numpy()

            probs_s1_list.append(p_s1)
            probs_s2a_list.append(p_s2a)
            probs_s2b_list.append(p_s2b)

    probs_s1 = np.vstack(probs_s1_list)
    probs_s2a = np.vstack(probs_s2a_list)
    probs_s2b = np.vstack(probs_s2b_list)

    pred_s1 = np.argmax(probs_s1, axis=-1) + 1
    mae_s1 = float(np.mean(np.abs(y_dev_s1_orig - pred_s1)))

    severe_mask = y_dev_s1_orig >= 4
    if np.sum(severe_mask) > 0:
        underestimation_pct = float(np.mean(pred_s1[severe_mask] <= 2) * 100.0)
    else:
        underestimation_pct = 0.0

    preds_s2a = [seleccionar_etiquetas(row, classes_s2a, UMBRAL_SEGUNDA_ETIQUETA) for row in probs_s2a]
    preds_s2b = [seleccionar_etiquetas(row, classes_s2b, UMBRAL_SEGUNDA_ETIQUETA) for row in probs_s2b]

    jaccard_s2a = float(np.mean([jaccard(r, p) for r, p in zip(y_dev_s2a_listas, preds_s2a)]))
    jaccard_s2b = float(np.mean([jaccard(r, p) for r, p in zip(y_dev_s2b_listas, preds_s2b)]))

    metrics = {
        "s1_mae": round(mae_s1, 4),
        "severe_underestimation_pct": round(underestimation_pct, 2),
        "s2a_jaccard": round(jaccard_s2a, 4),
        "s2b_jaccard": round(jaccard_s2b, 4),
    }
    return metrics, probs_s1, probs_s2a, probs_s2b


def main():
    args = parse_args()

    max_len = 128 if args.rapido else args.max_len
    epochs = 1 if args.rapido else args.epochs
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    lr = args.lr
    seed = args.seed

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp_initial = device.type == "cuda"

    print(f"=== M1 Training | device={device} | rapido={args.rapido} | epochs={epochs} | max_len={max_len} ===")

    train_df = pd.read_json(TRAIN_PATH, lines=True)
    dev_df = pd.read_json(DEV_PATH, lines=True)

    if args.rapido:
        train_df, _ = train_test_split(
            train_df,
            train_size=min(64, len(train_df)),
            stratify=train_df["risk_severity"],
            random_state=seed,
        )
        dev_df = dev_df.iloc[: min(64, len(dev_df))].reset_index(drop=True)

    y_train_s1_orig = train_df["risk_severity"].to_numpy()
    y_train_s1 = y_train_s1_orig - 1
    y_dev_s1_orig = dev_df["risk_severity"].to_numpy()

    mlb_s2a = MultiLabelBinarizer()
    y_train_s2a = mlb_s2a.fit_transform(train_df["risk_domains"])
    classes_s2a = mlb_s2a.classes_

    mlb_s2b = MultiLabelBinarizer()
    y_train_s2b = mlb_s2b.fit_transform(train_df["risk_drivers"])
    classes_s2b = mlb_s2b.classes_

    clases_presentes = np.unique(y_train_s1)
    pesos_calculados = compute_class_weight(class_weight="balanced", classes=clases_presentes, y=y_train_s1)
    weights_s1 = np.ones(5, dtype=np.float32)
    for c, w in zip(clases_presentes, pesos_calculados):
        weights_s1[int(c)] = float(w)
    weights_s1_tensor = torch.tensor(weights_s1, dtype=torch.float32, device=device)

    # 1. Truncamiento por la izquierda directo en from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncation_side="left")

    # 2. Comprobar que corta por la izquierda y conserva el final
    verificar_truncamiento_izquierda(train_df["formatted_interaction"], tokenizer, max_len)

    train_dataset = MultiTaskDataset(
        texts=train_df["formatted_interaction"],
        tokenizer=tokenizer,
        max_len=max_len,
        y_s1=y_train_s1,
        y_s2a=y_train_s2a,
        y_s2b=y_train_s2b,
    )
    dev_dataset = MultiTaskDataset(
        texts=dev_df["formatted_interaction"],
        tokenizer=tokenizer,
        max_len=max_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size * 2, shuffle=False)

    model = DebertaMultiTask(
        model_name=MODEL_NAME,
        num_s1=5,
        num_s2a=len(classes_s2a),
        num_s2b=len(classes_s2b),
    ).to(device)

    loss_s1_fn = nn.CrossEntropyLoss(weight=weights_s1_tensor)
    loss_s2a_fn = nn.BCEWithLogitsLoss()
    loss_s2b_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 3. Redondeo hacia arriba en los pasos por época para cubrir el último batch parcial
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    fp16_enabled = use_amp_initial
    scaler = torch.amp.GradScaler("cuda", enabled=fp16_enabled)

    start_time = time.time()
    last_epoch_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lbl_s1 = batch["label_s1"].to(device)
            lbl_s2a = batch["label_s2a"].to(device)
            lbl_s2b = batch["label_s2b"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16_enabled):
                logits_s1, logits_s2a, logits_s2b = model(input_ids, attention_mask)
                loss_1 = loss_s1_fn(logits_s1, lbl_s1)
                loss_2a = loss_s2a_fn(logits_s2a, lbl_s2a)
                loss_2b = loss_s2b_fn(logits_s2b, lbl_s2b)
                loss = loss_1 + loss_2a + loss_2b

            if torch.isnan(loss) or torch.isinf(loss):
                if fp16_enabled:
                    print(f"[Aviso] NaN/Inf detectado en época {epoch}, paso {step}. Apagando fp16...")
                    fp16_enabled = False
                    scaler = torch.amp.GradScaler("cuda", enabled=False)
                optimizer.zero_grad()
                continue

            loss_scaled = loss / grad_accum
            scaler.scale(loss_scaled).backward()
            running_loss += float(loss.item())

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        last_epoch_loss = running_loss / max(1, len(train_loader))
        print(f"Época {epoch}/{epochs} — Pérdida media train: {last_epoch_loss:.4f} — fp16: {fp16_enabled}")

    elapsed_seconds = round(time.time() - start_time, 2)

    metrics, probs_s1, probs_s2a, probs_s2b = evaluate_dev(
        model=model,
        loader=dev_loader,
        device=device,
        classes_s2a=classes_s2a,
        classes_s2b=classes_s2b,
        y_dev_s1_orig=y_dev_s1_orig,
        y_dev_s2a_listas=dev_df["risk_domains"].tolist(),
        y_dev_s2b_listas=dev_df["risk_drivers"].tolist(),
    )

    print("\n=== Resultados finales en DEV (última época) ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resultados_json = {
        "model": "M1",
        "backbone": MODEL_NAME,
        "mode_rapido": args.rapido,
        "seed": seed,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_sec": elapsed_seconds,
        "final_fp16_enabled": fp16_enabled,
        "last_epoch_train_loss": round(float(last_epoch_loss), 4),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers_version,
        },
        "hyperparameters": {
            "max_len": max_len,
            "truncation_side": "left",
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "learning_rate": lr,
            "epochs": epochs,
            "warmup_ratio": WARMUP_RATIO,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "threshold_second_label": UMBRAL_SEGUNDA_ETIQUETA,
            "s1_class_weights": [round(float(w), 4) for w in weights_s1],
        },
        "classes": {
            "s2a_domains": list(classes_s2a),
            "s2b_drivers": list(classes_s2b),
        },
        "metrics": metrics,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(resultados_json, f, indent=4, ensure_ascii=False)

    with open(PREDS_FILE, "w", encoding="utf-8") as f:
        for i, ex_id in enumerate(dev_df["example_id"]):
            record = {
                "example_id": ex_id,
                "probs_s1": [round(float(x), 6) for x in probs_s1[i]],
                "probs_s2a": {cls: round(float(probs_s2a[i, j]), 6) for j, cls in enumerate(classes_s2a)},
                "probs_s2b": {cls: round(float(probs_s2b[i, j]), 6) for j, cls in enumerate(classes_s2b)},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Métricas guardadas en: {METRICS_FILE}")
    print(f"Predicciones por ejemplo guardadas en: {PREDS_FILE}")


if __name__ == "__main__":
    main()