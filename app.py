import unicodedata
import re
import gradio as gr
import json
import os
import joblib
import numpy as np

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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: linear-gradient(135deg, #eef5ef 0%, #e6f2e8 100%) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* CONTENEDOR CENTRADO (NO FULL SCREEN EXAGERADO) */
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    padding-bottom: 20px;
}

/* HEADER MÁS ELEGANTE */
.chat-header {
    background: linear-gradient(135deg, #0d4a2a, #1a6b3c);
    padding: 30px;
    border-radius: 0 0 25px 25px;
    color: white;
    box-shadow: 0 6px 25px rgba(0,0,0,0.15);
}

/* SUGERENCIAS */
.sugerencias-wrap {
    padding: 20px 30px;
}

.sugerencias-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.chip {
    background: #ffffff;
    border: 1px solid #c5dcc9;
    border-radius: 20px;
    padding: 8px 15px;
    font-size: 13px;
    color: #145c32;
    cursor: pointer;
    transition: all 0.2s ease;
}

.chip:hover {
    background: #145c32;
    color: #fff;
    transform: translateY(-2px);
}

/* CHAT COMO TARJETA */
.gradio-container [data-testid="chatbot"] {
    background: #ffffff !important;
    margin: 10px 30px !important;
    border-radius: 18px !important;
    height: 420px !important;   /* 🔥 más pequeño */
    overflow-y: auto !important;
    border: 1px solid #dbe8dc !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    padding: 10px;
}

/* INPUT ZONA */
.gradio-row {
    margin: 10px 30px;
}

/* INPUT */
.gradio-container textarea {
    background: #ffffff !important;
    color: #1a2e1e !important;
    border: 1px solid #c5dcc9 !important;
    border-radius: 14px !important;
    padding: 12px !important;
}

/* BOTÓN */
#btn-enviar {
    background: linear-gradient(135deg, #145c32, #1a6b3c) !important;
    color: white !important;
    border-radius: 14px !important;
    font-weight: 600;
    transition: 0.2s;
}

#btn-enviar:hover {
    transform: scale(1.05);
    background: #0d4a2a !important;
}

/* TEXTO */
.gradio-container * {
    color: #1a2e1e;
}

/* DISCLAIMER */
.disclaimer {
    text-align: center;
    font-size: 12px;
    color: #6c8f74;
    padding: 10px;
}

/* SCROLL */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #145c32;
    border-radius: 10px;
}
"""

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

    chatbot = gr.Chatbot(label="", height=420, show_label=False)

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

    btn.click(fn=responder, inputs=[txt, chatbot], outputs=[chatbot, txt])
    txt.submit(fn=responder, inputs=[txt, chatbot], outputs=[chatbot, txt])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
