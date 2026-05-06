import unicodedata
import re
import gradio as gr
import json
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

UMBRAL_CONFIANZA = 0.18

modelo        = joblib.load("modelo_nlp.pkl")     if os.path.exists("modelo_nlp.pkl")     else None
vectorizer    = joblib.load("vectorizer.pkl")     if os.path.exists("vectorizer.pkl")     else None
label_encoder = joblib.load("label_encoder.pkl") if os.path.exists("label_encoder.pkl") else None

if os.path.exists("dataset_respuestas.json"):
    with open("dataset_respuestas.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
else:
    dataset = []

def limpiar_texto(texto):
    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def predecir_respuesta(pregunta_usuario):
    if not pregunta_usuario or not pregunta_usuario.strip():
        return None
    pregunta_limpia = limpiar_texto(pregunta_usuario)
    if modelo is None or vectorizer is None or label_encoder is None:
        return "⚠️ El modelo aún no está disponible. Verifica que los archivos .pkl estén en el repositorio."
    try:
        X             = vectorizer.transform([pregunta_limpia])
        proba         = modelo.predict_proba(X)[0]
        confianza     = float(np.max(proba))
        categoria_enc = int(np.argmax(proba))
        categoria     = label_encoder.inverse_transform([categoria_enc])[0]
        if confianza < UMBRAL_CONFIANZA:
            return (
                "🤔 No estoy completamente seguro de tu consulta.\n"
                "¿Podrías reformularla o dar más detalles?\n\n"
                "Ejemplos:\n"
                "- ¿Cuánto debo pagar?\n"
                "- ¿Cómo pago mi impuesto?\n"
                "- ¿Puedo fraccionar mi deuda?"
            )
        for item in dataset:
            if item.get("categoria") == categoria:
                return item.get("respuesta", "No tengo información sobre eso aún.")
        return "No encontré información para esa consulta. Le recomendamos acercarse a la Subgerencia de Rentas."
    except Exception as e:
        return f"Ocurrió un error al procesar su consulta: {str(e)}"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #f3f6f4 !important;
    font-family: 'Inter', sans-serif !important;
}

/* CONTENEDOR CENTRAL */
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}

/* HEADER ESTILO APP */
.chat-header {
    background: linear-gradient(135deg, #0d4a2a, #1a6b3c);
    padding: 20px 25px;
    border-radius: 0 0 20px 20px;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

.chip {
    background: #fff;
    border: 1px solid #d0e2d3;
    border-radius: 20px;
    padding: 7px 14px;
    font-size: 12px;
    color: #145c32;
    cursor: pointer;
    transition: 0.2s;
}

.chip:hover {
    background: #145c32;
    color: #fff;
}

/* CHAT */
.gradio-container [data-testid="chatbot"] {
    background: #ffffff !important;
    margin: 10px 20px !important;
    border-radius: 16px !important;
    height: 420px !important;
    border: 1px solid #e2e8e3 !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    padding: 15px;
}

/* 🔥 BURBUJAS (SIMULADAS) */
.gradio-container [data-testid="chatbot"] .message {
    padding: 10px 14px !important;
    border-radius: 14px !important;
    margin-bottom: 8px !important;
    max-width: 75%;
}

/* USUARIO (DERECHA) */
.gradio-container [data-testid="chatbot"] .message.user {
    background: #145c32 !important;
    color: white !important;
    margin-left: auto !important;
}

/* BOT (IZQUIERDA) */
.gradio-container [data-testid="chatbot"] .message.bot {
    background: #f1f5f2 !important;
    color: #1a2e1e !important;
}

/* INPUT COMO APP */
.gradio-row {
    margin: 10px 20px;
}

/* INPUT */
.gradio-container textarea {
    border-radius: 20px !important;
    padding: 12px 16px !important;
    border: 1px solid #d0e2d3 !important;
}

/* BOTÓN REDONDO */
#btn-enviar {
    border-radius: 20px !important;
    background: #145c32 !important;
    color: white !important;
    font-weight: 600;
}

#btn-enviar:hover {
    background: #0d4a2a !important;
}

/* DISCLAIMER */
.disclaimer {
    text-align: center;
    font-size: 11px;
    color: #7a9c86;
    padding: 10px;
}
"""

def generar_grafico_red():
    if modelo is None or vectorizer is None or label_encoder is None:
        return None

    entradas = len(vectorizer.get_feature_names_out())
    capas_ocultas = modelo.hidden_layer_sizes

    if isinstance(capas_ocultas, int):
        capas_ocultas = (capas_ocultas,)

    salidas = len(label_encoder.classes_)

    capas = [entradas] + list(capas_ocultas) + [salidas]

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, n_neuronas in enumerate(capas):
        y_positions = np.linspace(0, 1, n_neuronas)
        x = i / (len(capas) - 1)

        for y in y_positions:
            ax.scatter(x, y)

        if i > 0:
            prev_y = np.linspace(0, 1, capas[i - 1])
            for y1 in prev_y:
                for y2 in y_positions:
                    ax.plot(
                        [(i-1)/(len(capas)-1), x],
                        [y1, y2],
                        linewidth=0.2
                    )

    # 🔥 Etiquetas (esto sí va aquí)
    ax.text(0, 1.05, f"Entrada\n({entradas})", ha='center')
    ax.text(0.5, 1.05, f"Ocultas\n{capas_ocultas}", ha='center')
    ax.text(1, 1.05, f"Salida\n({salidas})", ha='center')

    ax.set_title("Arquitectura del Modelo NLP (MLP)")
    ax.axis('off')

    return fig
    

def responder(mensaje, historial):
    if not mensaje or not mensaje.strip():
        return historial, ""
    respuesta = predecir_respuesta(mensaje)
    if respuesta is None:
        return historial, ""
    historial = historial or []
    historial.append((mensaje, respuesta))
    return historial, ""

chips_html = (
    '<div class="sugerencias-wrap">'
    '<div class="sugerencias-label">Preguntas frecuentes</div>'
    '<div class="sugerencias-grid">'
)
chips_html += '</div></div><div class="divider"></div>'

with gr.Blocks(css=CSS, title="Asistente Tributario — Pangoa") as demo:

    gr.HTML("""
    <div class="chat-header">
        <div class="header-badge"><span class="header-dot"></span>En línea</div>
        <div class="header-title">🏛️ Asistente Tributario Virtual</div>
        <div class="header-sub">Municipalidad Distrital de Pangoa · Subgerencia de Rentas y Orientación Tributaria</div>
    </div>
    """)

    gr.HTML(chips_html)

    chatbot = gr.Chatbot(label="", height=420, show_label=False, bubble_full_width=False)

    # 🔥 AQUÍ VA EL GRÁFICO (BIEN UBICADO)
    with gr.Row():
        grafico = gr.Plot(label="Arquitectura del modelo de IA")

    # INPUT
    with gr.Row():
        txt = gr.Textbox(
            placeholder="Escribe tu consulta tributaria aquí...",
            show_label=False, lines=1, scale=8, container=False,
        )
        btn = gr.Button("Enviar →", scale=2, elem_id="btn-enviar")

    gr.HTML("""
    <div class="disclaimer">
        🧠 Red Neuronal MLP · Procesamiento de Lenguaje Natural ·
        Modelo entrenado con datos reales de Pangoa
    </div>
    """)

    # 🔥 CARGA DEL GRÁFICO
    demo.load(
        fn=generar_grafico_red,
        outputs=grafico
    )

    # MENSAJE INICIAL
    demo.load(
        fn=lambda: [(None, "👋 ¡Hola! Soy el Asistente Virtual Tributario de la Municipalidad Distrital de Pangoa. Puedo ayudarte con consultas sobre pagos, deudas, fraccionamientos, plazos y más. ¿En qué te puedo ayudar hoy?")],
        outputs=[chatbot],
    )

    btn.click(fn=responder, inputs=[txt, chatbot], outputs=[chatbot, txt])
    txt.submit(fn=responder, inputs=[txt, chatbot], outputs=[chatbot, txt])
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
