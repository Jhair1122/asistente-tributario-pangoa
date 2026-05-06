import unicodedata
import re
import gradio as gr
import json
import os
import joblib
import numpy as np
import time

UMBRAL_CONFIANZA = 0.25

modelo        = joblib.load("modelo_nlp.pkl")     if os.path.exists("modelo_nlp.pkl")     else None
vectorizer    = joblib.load("vectorizer.pkl")     if os.path.exists("vectorizer.pkl")     else None
label_encoder = joblib.load("label_encoder.pkl") if os.path.exists("label_encoder.pkl") else None

if os.path.exists("dataset_respuestas.json"):
    with open("dataset_respuestas.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
else:
    dataset = []

SUGERENCIAS = [
    "¿Cómo pago mi impuesto predial?",
    "¿Cuánto debo pagar?",
    "¿Puedo fraccionar mi deuda?",
    "¿Cuáles son los horarios de atención?",
    "¿Qué pasa si no pago a tiempo?",
    "¿Dónde queda el área de rentas?",
]

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

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
}

/* 🌞 MODO CLARO */
body.light {
    background: #f5f7f6;
}

/* 🌙 MODO OSCURO */
body.dark {
    background: #0f172a;
}

/* CONTENEDOR */
.gradio-container {
    max-width: 900px !important;
    margin: 30px auto !important;
    border-radius: 18px;
    overflow: hidden;
}

/* HEADER */
.chat-header {
    padding: 20px;
    background: linear-gradient(135deg, #145c32, #1f7a45);
    color: white;
}

/* CHAT */
[data-testid="chatbot"] {
    height: 360px !important;
    margin: 15px !important;
    border-radius: 12px;
    padding: 10px;
    overflow-y: auto;
}

/* 💬 BURBUJAS CLARAS */
.message.user {
    background: #e8f5ec !important;
    color: #145c32 !important;
    border-radius: 12px;
    padding: 8px 12px;
}

.message.bot {
    background: #f1f5f2 !important;
    color: #1a2e1e !important;
    border-radius: 12px;
    padding: 8px 12px;
}

/* INPUT */
textarea {
    border-radius: 12px !important;
}

/* BOTÓN */
#btn-enviar {
    border-radius: 12px !important;
    background: #145c32 !important;
    color: white !important;
}

/* DARK MODE OVERRIDE */
body.dark .gradio-container {
    background: #111827 !important;
}

body.dark [data-testid="chatbot"] {
    background: #1f2937 !important;
}

body.dark .message.bot {
    background: #374151 !important;
    color: #e5e7eb !important;
}

body.dark .message.user {
    background: #064e3b !important;
    color: #d1fae5 !important;
}

"""

def responder(mensaje, historial):
    if not mensaje or not mensaje.strip():
        return historial, ""

    historial = historial or []

    # 👤 usuario escribe
    historial.append((mensaje, "✍️ Escribiendo..."))

    yield historial, ""

    # 🧠 delay realista
    time.sleep(1.2)

    respuesta = predecir_respuesta(mensaje)

    # reemplazar mensaje temporal
    historial[-1] = (mensaje, respuesta)

    yield historial, ""

chips_html = (
    '<div class="sugerencias-wrap">'
    '<div class="sugerencias-label">Preguntas frecuentes</div>'
    '<div class="sugerencias-grid">'
)
for s in SUGERENCIAS:
    chips_html += (
        f'<button class="chip" onclick="(function(){{'
        f'var ta=document.querySelector(\'textarea\');'
        f'if(ta){{ta.value=\'{s}\';ta.dispatchEvent(new Event(\'input\',{{bubbles:true}}));}}'
        f'}})()\">{s}</button>'
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

    # Mensaje de bienvenida
    demo.load(
        fn=lambda: [(None, "👋 ¡Hola! Soy el Asistente Virtual Tributario de la Municipalidad Distrital de Pangoa. Puedo ayudarte con consultas sobre pagos, deudas, fraccionamientos, plazos y más. ¿En qué te puedo ayudar hoy?")],
        outputs=[chatbot],
    )

    btn.click(fn=responder, inputs=[txt, chatbot], outputs=[chatbot, txt], queue=True)
    txt.submit(fn=responder, inputs=[txt, chatbot], outputs=[chatbot, txt], queue=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
