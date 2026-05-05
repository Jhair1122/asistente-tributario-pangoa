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
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;700&family=Inter:wght@400;600;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* Fondo general estilo Terminal/IDE Moderno expandido al 100% */
body, .gradio-container {
    background: #0d1117 !important; /* Color de fondo tipo GitHub Dark */
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh !important;
    max-width: 100% !important; /* Aquí forzamos que use toda la pantalla */
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden;
}

/* Cabecera Principal Animada */
.chat-header {
    background: linear-gradient(90deg, #052e16 0%, #065f46 50%, #052e16 100%);
    padding: 25px 40px;
    border-bottom: 2px solid #10b981;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    animation: glowHeader 4s infinite alternate;
}

@keyframes glowHeader {
    from { box-shadow: 0 4px 15px rgba(16, 185, 129, 0.1); }
    to { box-shadow: 0 4px 30px rgba(16, 185, 129, 0.4); }
}

.header-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(16, 185, 129, 0.1); 
    border: 1px solid #10b981;
    border-radius: 4px; padding: 4px 12px; 
    font-family: 'Fira Code', monospace;
    font-size: 12px; color: #34d399; 
    text-transform: uppercase; margin-bottom: 10px;
}

.header-dot {
    width: 8px; height: 8px; background: #10b981;
    border-radius: 50%; animation: blink 1.5s infinite;
}

@keyframes blink { 
    0%, 100% { opacity: 1; box-shadow: 0 0 10px #10b981; } 
    50% { opacity: 0.3; box-shadow: 0 0 0 transparent; } 
}

.header-title { 
    font-size: 28px; font-weight: 800; color: #ecfdf5;
    letter-spacing: 1px;
}

.header-sub { 
    font-size: 14px; color: #6ee7b7; font-family: 'Fira Code', monospace;
    margin-top: 5px;
}

/* Botones de Sugerencias (Chips) con estilo de comandos */
.sugerencias-wrap { 
    padding: 20px 5vw; 
    background: #0f172a;
    border-bottom: 1px solid #1e293b;
}
.sugerencias-label {
    font-family: 'Fira Code', monospace;
    font-size: 12px; color: #10b981;
    text-transform: uppercase; margin-bottom: 15px;
    display: flex; align-items: center; gap: 10px;
}
.sugerencias-label::before { content: '>'; color: #34d399; font-weight: bold; }
.sugerencias-grid { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; }

.chip {
    background: rgba(16, 185, 129, 0.05); border: 1px solid #059669; 
    border-radius: 4px; padding: 10px 20px; 
    font-size: 13px; font-family: 'Fira Code', monospace; color: #6ee7b7;
    cursor: pointer; transition: all 0.3s ease;
    position: relative; overflow: hidden;
}
.chip:hover {
    background: #10b981; color: #022c22; 
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    transform: translateY(-3px) scale(1.02);
}

/* Contenedor del Chat expandido */
.gradio-container [data-testid="chatbot"] {
    background: #0d1117 !important; 
    border: none !important;
    margin: 0 !important;
    padding: 20px 5vw !important;
    height: calc(100vh - 400px) !important; /* Ajusta la altura dinámicamente */
    min-height: 400px !important;
}

/* Área de Texto y Botón de Envío */
.gradio-container > div > div > div > div.wrap, 
.gradio-container > div > div > div > div > div {
    padding: 0 5vw 20px 5vw !important;
    background: transparent !important;
    border: none !important;
}

.gradio-container textarea,
.gradio-container input[type=text] {
    background: #1e293b !important; color: #f8fafc !important;
    border: 1px solid #334155 !important; border-radius: 8px !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 14px !important; padding: 18px !important;
    transition: all 0.3s ease !important;
}
.gradio-container textarea:focus,
.gradio-container input[type=text]:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.2) !important; 
    outline: none !important;
}

#btn-enviar { 
    background: #10b981 !important; 
    border: none !important; border-radius: 8px !important; 
    color: #022c22 !important; font-weight: 800 !important; 
    font-family: 'Fira Code', monospace !important;
    font-size: 15px !important; text-transform: uppercase !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    min-height: 58px !important;
}
#btn-enviar:hover { 
    background: #34d399 !important;
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.5) !important;
    transform: scale(1.05) !important;
}

/* Pie de página */
.disclaimer {
    text-align: center; font-size: 12px; color: #475569;
    font-family: 'Fira Code', monospace;
    padding: 10px 0 20px 0; background: transparent;
}
footer { display: none !important; }
.divider { display: none; }
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
