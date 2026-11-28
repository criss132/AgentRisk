"""Interfaz web (Gradio) del monitor AgentRisk.

Pega la trayectoria de un agente y devuelve la severidad (S1), los dominios (S2a), los factores (S2b)
y, si se elige una política, la intervención de S3 (permitir, confirmar o bloquear).
Reutiliza M1 y los umbrales de src/prototipo.py. Se ejecuta desde la raíz: python src/app_gradio.py
y se abre en http://127.0.0.1:7860
"""
import json
import os
import sys

import gradio as gr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.prototipo import cargar_con_respaldo, predecir
from src.s3_politicas import aplicar_politica

DEV_PATH = os.path.join(REPO_ROOT, "data", "raw", "data", "dev.jsonl")
CASOS_DEMO = ("AGENTRISK-064", "AGENTRISK-051")
SIN_POLITICA = "ninguna"
OPCIONES_POLITICA = [
    ("Sin política", SIN_POLITICA),
    ("Política A (equilibrada)", "A"),
    ("Política B (estricta en dominios sensibles)", "B"),
]

_bundle = None


def obtener_modelo() -> dict:
    """Carga M1 la primera vez (con respaldo en L1) y la reutiliza en las siguientes consultas."""
    global _bundle
    if _bundle is None:
        _bundle = cargar_con_respaldo()
    return _bundle


def cargar_ejemplos() -> list:
    """Lee de dev los casos de la demo; si los datos no se han descargado, no hay ejemplos."""
    if not os.path.exists(DEV_PATH):
        return []
    ejemplos = []
    with open(DEV_PATH, encoding="utf-8") as archivo:
        for linea in archivo:
            registro = json.loads(linea)
            if registro["example_id"] in CASOS_DEMO:
                ejemplos.append([registro["formatted_interaction"], SIN_POLITICA])
    return ejemplos


def analizar(trayectoria: str, politica: str):
    """Predice con M1 y devuelve (modelo, severidad, dominios, factores, intervención, probabilidades de S1)."""
    if not trayectoria or not trayectoria.strip():
        raise gr.Error("Pega una trayectoria para analizar.")
    resultado = predecir(trayectoria, bundle=obtener_modelo())
    intervencion = "—"
    if politica != SIN_POLITICA:
        intervencion = aplicar_politica(
            politica, resultado["severidad"], resultado["dominios"], resultado["factores"]
        ).upper()
    probabilidades_s1 = {f"Severidad {k}": v for k, v in resultado["probabilidades"]["s1"].items()}
    return (
        resultado["backend"],
        str(resultado["severidad"]),
        ", ".join(resultado["dominios"]),
        ", ".join(resultado["factores"]),
        intervencion,
        probabilidades_s1,
    )


def construir_interfaz() -> gr.Blocks:
    """Arma la página: entrada, política, botón y las salidas del monitor."""
    with gr.Blocks(title="AgentRisk") as demo:
        gr.Markdown(
            "# AgentRisk: monitor de riesgo para agentes\n"
            "Pega la trayectoria del agente (con la acción propuesta al final) y pulsa **Analizar**."
        )
        trayectoria = gr.Textbox(label="Trayectoria del agente", lines=14)
        politica = gr.Radio(OPCIONES_POLITICA, value=SIN_POLITICA, label="Política de intervención (S3)")
        boton = gr.Button("Analizar", variant="primary")
        with gr.Row():
            modelo = gr.Textbox(label="Modelo activo")
            severidad = gr.Textbox(label="Severidad (1 a 5)")
            intervencion = gr.Textbox(label="Intervención (S3)")
        dominios = gr.Textbox(label="Dominios de daño (S2a)")
        factores = gr.Textbox(label="Factores de riesgo (S2b)")
        probabilidades = gr.Label(label="Probabilidad de cada severidad (S1)", num_top_classes=5)
        boton.click(
            analizar,
            inputs=[trayectoria, politica],
            outputs=[modelo, severidad, dominios, factores, intervencion, probabilidades],
        )
        ejemplos = cargar_ejemplos()
        if ejemplos:
            gr.Examples(examples=ejemplos, inputs=[trayectoria, politica], label="Casos de la demostración (dev)")
    return demo


if __name__ == "__main__":
    modelo_activo = obtener_modelo()["backend"]
    if modelo_activo != "M1":
        print(f"[Aviso] Modelo activo: {modelo_activo}. Revisa que los pesos de M1 estén en experiments/checkpoints/.")
    construir_interfaz().launch()
