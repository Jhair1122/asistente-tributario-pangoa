import unicodedata
import re
import gradio as gr
import json
import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

UMBRAL_CONFIANZA = 0.18

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

── Gráfico de arquitectura (sin dibujar neuronas individuales) ──────────────

def generar_grafico_red():
try:
if modelo is None or vectorizer is None or label_encoder is None:
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.text(0.5, 0.5, "⚠️ Modelo no disponible",
ha="center", va="center", fontsize=13, color="#888")
ax.axis("off")
return fig

n_entrada  = len(vectorizer.get_feature_names_out())  
    capas_ocultas = modelo.hidden_layer_sizes  
    if isinstance(capas_ocultas, int):  
        capas_ocultas = (capas_ocultas,)  
    n_salida = len(label_encoder.classes_)  

    # Representamos cada capa como un rectángulo con altura proporcional  
    capas = [("Entrada\nTF-IDF", n_entrada, "#1a6b3c"),  
             *[(f"Oculta {i+1}\nReLU", n, "#2e86ab") for i, n in enumerate(capas_ocultas)],  
             ("Salida\nSoftmax", n_salida, "#c45c2e")]  

    fig, ax = plt.subplots(figsize=(9, 4))  
    fig.patch.set_facecolor("#f8fdf9")  
    ax.set_facecolor("#f8fdf9")  

    max_n = max(c[1] for c in capas)  
    x_positions = np.linspace(0.1, 0.9, len(capas))  
    bar_w = 0.10  

    for i, (nombre, n, color) in enumerate(capas):  
        altura = 0.15 + 0.75 * (n / max_n)  
        y0 = (1 - altura) / 2  
        rect = mpatches.FancyBboxPatch(  
            (x_positions[i] - bar_w/2, y0), bar_w, altura,  
            boxstyle="round,pad=0.01",  
            linewidth=1.5, edgecolor="white",  
            facecolor=color, alpha=0.9  
        )  
        ax.add_patch(rect)  
        # Etiqueta arriba  
        ax.text(x_positions[i], y0 + altura + 0.06, nombre,  
                ha="center", va="bottom", fontsize=8.5,  
                fontweight="bold", color="#2d2d2d")  
        # Número de neuronas  
        ax.text(x_positions[i], y0 + altura/2, f"{n:,}",  
                ha="center", va="center", fontsize=9,  
                fontweight="bold", color="white")  
        # Flechas entre capas  
        if i < len(capas) - 1:  
            ax.annotate("",   
                xy=(x_positions[i+1] - bar_w/2 - 0.01, 0.5),  
                xytext=(x_positions[i] + bar_w/2 + 0.01, 0.5),  
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.5))  

    ax.set_xlim(0, 1)  
    ax.set_ylim(0, 1.3)  
    ax.axis("off")  
    ax.set_title("Arquitectura del Modelo NLP — Red Neuronal MLP",  
                 fontsize=11, fontweight="bold", color="#1a3a24", pad=10)  
    plt.tight_layout()  
    return fig  

except Exception as e:  
    fig, ax = plt.subplots(figsize=(8, 3))  
    ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", color="red")  
    ax.axis("off")  
    return fig

── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
background: #f0f5f1 !important;
font-family: 'Inter', sans-serif !important;
}
.gradio-container {
max-width: 860px !important;
margin: 0 auto !important;
padding: 0 !important;
}

/* ── Header ── */
.chat-header {
background: linear-gradient(135deg, #0a3d22 0%, #145c32 55%, #1a7a42 100%);
padding: 24px 32px 20px;
border-radius: 0 0 28px 28px;
box-shadow: 0 6px 28px rgba(10,61,34,0.22);
position: relative;
overflow: hidden;
}
.chat-header::after {
content: '';
position: absolute;
bottom: -50px; right: -50px;
width: 200px; height: 200px;
background: rgba(255,255,255,0.04);
border-radius: 50%;
pointer-events: none;
}
.header-top {
display: flex;
align-items: center;
gap: 14px;
margin-bottom: 10px;
}
.header-icon {
width: 46px; height: 46px;
background: rgba(255,255,255,0.15);
border-radius: 12px;
display: flex; align-items: center; justify-content: center;
font-size: 22px;
flex-shrink: 0;
border: 1px solid rgba(255,255,255,0.2);
}
.header-texts { flex: 1; }
.header-title {
font-size: 18px;
font-weight: 700;
color: #ffffff;
line-height: 1.2;
}
.header-sub {
font-size: 12px;
color: rgba(255,255,255,0.7);
margin-top: 2px;
}
.header-badge {
display: inline-flex;
align-items: center;
gap: 6px;
background: rgba(74,222,128,0.15);
border: 1px solid rgba(74,222,128,0.3);
border-radius: 20px;
padding: 4px 12px;
font-size: 11px;
font-weight: 600;
color: #86efac;
letter-spacing: 0.05em;
}
.header-dot {
width: 7px; height: 7px;
background: #4ade80;
border-radius: 50%;
animation: pulse 2s infinite;
}
@keyframes pulse {
0%, 100% { opacity: 1; transform: scale(1); }
50% { opacity: 0.4; transform: scale(0.75); }
}

/* ── Sugerencias ── */
.sugerencias-wrap {
padding: 14px 20px 6px;
background: #f0f5f1;
}
.sugerencias-label {
font-size: 10px;
font-weight: 700;
color: #5a8a6a;
letter-spacing: 0.1em;
text-transform: uppercase;
margin-bottom: 8px;
}
.sugerencias-grid {
display: flex;
flex-wrap: wrap;
gap: 7px;
}
.chip {
background: #ffffff;
border: 1px solid #c8ddd0;
border-radius: 20px;
padding: 6px 13px;
font-size: 12px;
font-weight: 500;
color: #145c32;
cursor: pointer;
transition: all 0.15s ease;
font-family: 'Inter', sans-serif;
white-space: nowrap;
box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.chip:hover {
background: #145c32;
color: #ffffff;
border-color: #145c32;
transform: translateY(-1px);
box-shadow: 0 3px 10px rgba(20,92,50,0.25);
}
.divider {
height: 1px;
background: linear-gradient(90deg, transparent, #c8ddd0, transparent);
margin: 10px 20px 0;
}

/* ── Chatbot ── */
.gradio-container [data-testid="chatbot"] {
background: #ffffff !important;
border: 1px solid #d8e8de !important;
border-radius: 20px !important;
margin: 14px 16px !important;
box-shadow: 0 4px 20px rgba(20,92,50,0.07) !important;
}

/* ── Input y botón ── */
.gradio-container textarea,
.gradio-container input[type=text] {
background: #ffffff !important;
color: #1a2e1e !important;
border: 1.5px solid #c8ddd0 !important;
border-radius: 14px !important;
font-family: 'Inter', sans-serif !important;
font-size: 14px !important;
padding: 11px 16px !important;
caret-color: #145c32 !important;
transition: border-color 0.2s, box-shadow 0.2s !important;
}
.gradio-container textarea::placeholder { color: #9cb8a8 !important; }
.gradio-container textarea:focus,
.gradio-container input[type=text]:focus {
border-color: #145c32 !important;
box-shadow: 0 0 0 3px rgba(20,92,50,0.1) !important;
outline: none !important;
}
#btn-enviar button,
.gradio-container button.primary {
background: linear-gradient(135deg, #145c32, #1a7a42) !important;
border: none !important;
border-radius: 14px !important;
color: #ffffff !important;
font-weight: 700 !important;
font-size: 14px !important;
font-family: 'Inter', sans-serif !important;
box-shadow: 0 3px 12px rgba(20,92,50,0.3) !important;
transition: all 0.2s !important;
}
#btn-enviar button:hover {
background: linear-gradient(135deg, #0d4a2a, #145c32) !important;
transform: translateY(-1px) !important;
box-shadow: 0 5px 16px rgba(20,92,50,0.4) !important;
}

/* ── Gráfico ── */
.grafico-wrap {
margin: 0 16px 4px;
background: #ffffff;
border: 1px solid #d8e8de;
border-radius: 16px;
padding: 16px;
box-shadow: 0 2px 10px rgba(20,92,50,0.05);
}
.grafico-titulo {
font-size: 11px;
font-weight: 700;
color: #5a8a6a;
letter-spacing: 0.08em;
text-transform: uppercase;
margin-bottom: 10px;
}

/* ── Disclaimer ── */
.disclaimer {
text-align: center;
font-size: 11px;
color: #7aaa8a;
padding: 6px 20px 18px;
font-weight: 500;
}

footer { display: none !important; }
"""

── Lógica ───────────────────────────────────────────────────────────────────

def responder(mensaje, historial):
if not mensaje or not mensaje.strip():
return historial, ""
respuesta = predecir_respuesta(mensaje)
if respuesta is None:
return historial, ""
historial = historial or []
historial.append((mensaje, respuesta))
return historial, ""

def bienvenida():
return [(None, "👋 ¡Hola! Soy el Asistente Virtual Tributario de la Municipalidad Distrital de Pangoa. Puedo ayudarte con consultas sobre pagos, deudas, fraccionamientos, plazos y más. ¿En qué te puedo ayudar hoy?")]

── Chips ─────────────────────────────────────────────────────────────────────

chips_html = (
'<div class="sugerencias-wrap">'
'<div class="sugerencias-label">💬 Preguntas frecuentes</div>'
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

── Interfaz ──────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title="Asistente Tributario — Pangoa") as demo:

gr.HTML("""  
<div class="chat-header">  
    <div class="header-top">  
        <div class="header-icon">🏛️</div>  
        <div class="header-texts">  
            <div class="header-title">Asistente Tributario Virtual</div>  
            <div class="header-sub">Municipalidad Distrital de Pangoa · Subgerencia de Rentas</div>  
        </div>  
        <div class="header-badge">  
            <span class="header-dot"></span>En línea  
        </div>  
    </div>  
</div>  
""")  

gr.HTML(chips_html)  

chatbot = gr.Chatbot(label="", height=400, show_label=False)  


with gr.Row():  
    txt = gr.Textbox(  
        placeholder="Escribe tu consulta tributaria aquí...",  
        show_label=False, lines=1, scale=8, container=False,  
    )  
    btn = gr.Button("Enviar →", scale=2, variant="primary", elem_id="btn-enviar")  
    # Gráfico de arquitectura  
gr.HTML('<div class="grafico-wrap"><div class="grafico-titulo">🧠 Arquitectura de la Red Neuronal MLP</div>')  
grafico = gr.Plot(label="", show_label=False)  
gr.HTML('</div>')

gr.HTML("""  
<div class="disclaimer">  
    🧠 Red Neuronal MLP · TF-IDF · Modelo entrenado con datos reales de Pangoa  
</div>  
""")  

btn.click(fn=responder, inputs=[txt, chatbot], outputs=[chatbot, txt])  
txt.submit(fn=responder, inputs=[txt, chatbot], outputs=[chatbot, txt])  
demo.load(fn=bienvenida, outputs=[chatbot])  
demo.load(fn=generar_grafico_red, outputs=[grafico])

if name == "main":
port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)
