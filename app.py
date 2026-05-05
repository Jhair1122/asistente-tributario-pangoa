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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: linear-gradient(135deg, #e8f0eb 0%, #f4f7f6 100%) !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
}

.gradio-container { 
    max-width: 800px !important; 
    margin: 0 auto !important; 
    padding: 20px 10px !important; 
}

/* Cabecera Principal */
.chat-header {
    background: linear-gradient(135deg, #053b20 0%, #0d6b3a 100%);
    padding: 32px 40px; 
    border-radius: 24px;
    box-shadow: 0 10px 30px rgba(13, 107, 58, 0.2); 
    position: relative; 
    overflow: hidden;
    margin-bottom: 24px;
    color: white;
}

.chat-header::after {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 300px; height: 300px; 
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%); 
    border-radius: 50%;
}

.header-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.15); 
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px; padding: 6px 14px; 
    font-size: 12px; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase; 
    margin-bottom: 12px;
    backdrop-filter: blur(5px);
}

.header-dot {
    width: 8px; height: 8px; background: #4ade80;
    border-radius: 50%; animation: pulse 2s infinite;
    box-shadow: 0 0 10px rgba(74, 222, 128, 0.8);
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(74, 222, 128, 0); }
    100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
}

.header-title { 
    font-size: 26px; font-weight: 800; 
    margin-bottom: 6px; 
    text-shadow: 0 2px 4px rgba(0,0,0,0.2); 
}

.header-sub { 
    font-size: 14px; color: #d1fae5; 
    font-weight: 500; opacity: 0.9; 
}

/* Botones de Sugerencias (Chips) */
.sugerencias-wrap { margin-bottom: 20px; padding: 0 10px; }
.sugerencias-label {
    font-size: 12px; font-weight: 700; color: #166534;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 14px;
    display: flex; align-items: center; gap: 10px;
}
.sugerencias-label::after { 
    content: ''; flex-grow: 1; height: 2px; 
    background: #dcfce7; border-radius: 2px;
}
.sugerencias-grid { display: flex; flex-wrap: wrap; gap: 10px; }

.chip {
    background: #ffffff; border: 2px solid #bbf7d0; border-radius: 24px;
    padding: 10px 18px; font-size: 14px; font-weight: 600; color: #166534;
    cursor: pointer; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 5px rgba(22, 101, 52, 0.05);
}
.chip:hover {
    background: #166534; color: #ffffff; border-color: #166534;
    transform: translateY(-3px); box-shadow: 0 6px 15px rgba(22, 101, 52, 0.2);
}
.divider { display: none; } /* Ocultamos el divider viejo para un look más limpio */

/* Contenedor del Chat */
.gradio-container [data-testid="chatbot"] {
    background: #ffffff !important; 
    border: 2px solid #e2e8f0 !important;
    border-radius: 24px !important; 
    box-shadow: 0 10px 40px rgba(0,0,0,0.04) !important;
    padding: 10px !important;
    margin-bottom: 20px !important;
}

/* Área de Texto y Botón de Envío */
.gradio-container textarea,
.gradio-container input[type=text] {
    background: #ffffff !important; color: #064e3b !important;
    border: 2px solid #e2e8f0 !important; border-radius: 20px !important;
    font-size: 15px !important; padding: 16px 20px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
}
.gradio-container textarea:focus,
.gradio-container input[type=text]:focus {
    border-color: #15803d !important;
    box-shadow: 0 0 0 4px rgba(21, 128, 61, 0.1) !important; 
    outline: none !important;
}

#btn-enviar { 
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important; 
    border: none !important; border-radius: 20px !important; 
    color: #ffffff !important; font-weight: 700 !important; 
    font-size: 15px !important; letter-spacing: 0.03em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(21, 128, 61, 0.2) !important;
    height: 100% !important; min-height: 55px !important;
}
#btn-enviar:hover { 
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(21, 128, 61, 0.3) !important;
}

/* Pie de página */
.disclaimer {
    text-align: center; font-size: 12px; color: #64748b;
    padding: 20px 0; font-weight: 500;
}
footer { display: none !important; }

/* =========================================
   SOPORTE PARA MODO OSCURO (DARK MODE)
   ========================================= */
@media (prefers-color-scheme: dark) {
    body, .gradio-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    }
    .chat-header {
        background: linear-gradient(135deg, #022c22 0%, #064e3b 100%);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .sugerencias-label { color: #6ee7b7; }
    .sugerencias-label::after { background: #334155; }
    
    .chip {
        background: #1e293b; border-color: #334155; color: #a7f3d0;
    }
    .chip:hover {
        background: #059669; color: white; border-color: #059669;
    }
    
    .gradio-container [data-testid="chatbot"] {
        background: #1e293b !important;
        border-color: #334155 !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3) !important;
    }
    
    .gradio-container textarea,
    .gradio-container input[type=text] {
        background: #1e293b !important; color: #f8fafc !important;
        border-color: #334155 !important;
    }
    .gradio-container textarea:focus,
    .gradio-container input[type=text]:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.1) !important; 
    }
    
    .disclaimer { color: #94a3b8; }
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
