"""Streamlit MVP: review held-out sender windows with trained model outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


@st.cache_data
def load_artifacts():
    with (ARTIFACTS / "test_cases.json").open(encoding="utf-8") as file:
        cases = json.load(file)
    with (ARTIFACTS / "results.json").open(encoding="utf-8") as file:
        results = json.load(file)
    return cases, results


def explain(case: dict, threshold: float) -> str:
    attention = np.asarray(case["attention"])
    errors = np.asarray(case["reconstruction_error"])
    # Both maps are descriptive signals, not causal explanations.
    top = sorted(set(np.argsort(attention)[-2:].tolist() +
                     np.argsort(errors)[-2:].tolist()))
    details = []
    for index in top:
        tx = case["transactions"][index]
        details.append(f"#{index + 1} ({tx['timestamp']}, {tx['amount']:,.2f} "
                       f"{tx['currency']}, {tx['format']})")
    status = "supera" if case["alert"] else "no supera"
    return (f"La ventana del remitente {case['sender']} {status} el umbral de "
            f"alerta ({threshold:.3f}). La etapa A produjo un error de reconstrucción "
            f"de {case['anomaly_score']:.3f} y la etapa B una probabilidad calibrada "
            f"de {case['stage_b_probability']:.1%}. Las transacciones con mayor atención "
            f"o error fueron {', '.join(details)}. Esto orienta una revisión humana; "
            "la atención por sí sola no demuestra causalidad ni delito.")


st.set_page_config(page_title="AML | revisión de remesas", layout="wide")
st.title("Revisión de patrones transaccionales")
st.caption("Proyecto académico con datos sintéticos IBM AML. Ventanas retenidas del conjunto de prueba.")

if not (ARTIFACTS / "test_cases.json").exists():
    st.error("Faltan artefactos del entrenamiento. Ejecute `python -m src.train` antes de iniciar el MVP.")
    st.stop()

cases, results = load_artifacts()
sender_ids = sorted({case["sender"] for case in cases})
left, right = st.columns([2, 3])
with left:
    selected_sender = st.selectbox("Seleccionar remitente de prueba", sender_ids)
with right:
    entered = st.text_input("O ingresar ID de remitente del conjunto de prueba", placeholder="Banco:Cuenta")
sender = entered.strip() or selected_sender
matches = [case for case in cases if case["sender"] == sender]
if not matches:
    st.warning("Ese remitente no está en las ventanas de prueba publicadas con el MVP.")
    st.stop()

case = st.selectbox("Ventana cronológica", matches,
                    format_func=lambda item: f"Ventana {item['window']} · {len(item['transactions'])} transacciones")
threshold = results["test"]["combined"]["threshold"]
col_a, col_b, col_c = st.columns(3)
col_a.metric("Etapa A · error de reconstrucción", f"{case['anomaly_score']:.3f}")
col_b.metric("Etapa B · probabilidad calibrada", f"{case['stage_b_probability']:.1%}")
col_c.metric("Decisión combinada", "Alerta" if case["alert"] else "Sin alerta",
             delta=f"score {case['combined_score']:.3f} / umbral {threshold:.3f}", delta_color="off")

data = pd.DataFrame(case["transactions"])
data.insert(0, "Nº", np.arange(1, len(data) + 1))
data["Atención"] = np.asarray(case["attention"]).round(4)
data["Error de reconstrucción"] = np.asarray(case["reconstruction_error"]).round(4)
data = data.drop(columns=["transaction_label"])
st.subheader("Secuencia de transacciones")
st.dataframe(data, use_container_width=True, hide_index=True)

fig, ax = plt.subplots(figsize=(max(8, len(data) * 0.45), 2.4))
heat = np.vstack([case["attention"], case["reconstruction_error"]]).astype(float)
heat = heat / np.maximum(heat.max(axis=1, keepdims=True), 1e-9)
plot = ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_yticks([0, 1], ["Atención", "Error A"])
ax.set_xticks(range(len(data)), range(1, len(data) + 1))
ax.set_xlabel("Transacción en orden temporal")
fig.colorbar(plot, ax=ax, label="Intensidad relativa")
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

st.subheader("Explicación de la alerta")
st.write(explain(case, threshold))
with st.expander("Información para evaluación académica"):
    st.write(f"Etiqueta real de esta ventana: {'lavado sintético' if case['label'] else 'normal'}")
    st.write("El conjunto IBM incluye transferencias bancarias y otros pagos. No representa remesas Guatemala–EE. UU. ni permite inferir culpabilidad.")
