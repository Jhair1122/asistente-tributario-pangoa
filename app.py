import unicodedata
import re
import gradio as gr
import json
import os
import joblib
import numpy as np

UMBRAL_CONFIANZA = 0.1

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
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body, .gradio-container {
    background: #f0f4f0 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    min-height: 100vh;
}
.gradio-container { max-width: 780px !important; margin: 0 auto !important; padding: 0 !important; }
.chat-header {
    background: linear-gradient(135deg, #0d4a2a 0%, #1a6b3c 60%, #145c32 100%);
    padding: 28px 32px 22px; border-radius: 0 0 24px 24px;
    box-shadow: 0 4px 24px rgba(13,74,42,0.18); position: relative; overflow: hidden;
}
.chat-header::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px; background: rgba(255,255,255,0.04); border-radius: 50%;
}
.header-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px; padding: 4px 12px; font-size: 11px; font-weight: 600;
    color: #a8dbbe; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 10px;
}
.header-dot {
    width: 6px; height: 6px; background: #4ade80;
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}
.header-title { font-size: 22px; font-weight: 800; color: #ffffff; margin-bottom: 4px; }
.header-sub   { font-size: 13px; color: #a8dbbe; font-weight: 500; }
.sugerencias-wrap { padding: 16px 20px 4px; }
.sugerencias-label {
    font-size: 11px; font-weight: 700; color: #4a7c5a;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px;
}
.sugerencias-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    background: #ffffff; border: 1.5px solid #c5dcc9; border-radius: 20px;
    padding: 7px 14px; font-size: 13px; font-weight: 500; color: #145c32;
    cursor: pointer; transition: all 0.18s ease; font-family: 'Plus Jakarta Sans', sans-serif;
}
.chip:hover {
    background: #145c32; color: #ffffff; border-color: #145c32;
    transform: translateY(-1px); box-shadow: 0 4px 12px rgba(20,92,50,0.2);
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #c5dcc9, transparent);
    margin: 12px 20px;
}
.gradio-container [data-testid="chatbot"] {
    background: #ffffff !important; border: 1.5px solid #c5dcc9 !important;
    border-radius: 16px !important; margin: 0 16px !important;
    box-shadow: 0 2px 12px rgba(20,92,50,0.06) !important;
}
.gradio-container textarea,
.gradio-container input[type=text] {
    background: #ffffff !important; color: #1a2e1e !important;
    border: 1.5px solid #c5dcc9 !important; border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important; padding: 10px 14px !important;
    caret-color: #145c32 !important;
}
.gradio-container textarea::placeholder,
.gradio-container input[type=text]::placeholder { color: #9ab5a0 !important; }
.gradio-container textarea:focus,
.gradio-container input[type=text]:focus {
    border-color: #145c32 !important;
    box-shadow: 0 0 0 3px rgba(20,92,50,0.08) !important; outline: none !important;
}
#btn-enviar { background: #145c32 !important; border: none !important;
    border-radius: 12px !important; color: #ffffff !important;
    font-weight: 700 !important; font-size: 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important; }
#btn-enviar:hover { background: #0d4a2a !important; }
.disclaimer {
    text-align: center; font-size: 11px; color: #7aad8a;
    padding: 8px 20px 16px; font-weight: 500;
}
footer { display: none !important; }
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
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    app = demo.app
    uvicorn.run(app, host="0.0.0.0", port=port)
