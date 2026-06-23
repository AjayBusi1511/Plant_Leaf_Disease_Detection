import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import streamlit as st
from PIL import Image
from groq import Groq

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

st.markdown("""
<style>
    /* ── Background ── */
    .stApp { background-color: #f0f7f0; }

    /* ── Top banner ── */
    .banner {
        background: linear-gradient(135deg, #1b5e20, #43a047, #a5d6a7);
        border-radius: 18px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .banner h1 { color: white !important; font-size: 3.5rem !important; font-weight: 900 !important; margin: 0 !important; }
    .banner p  { color: #e8f5e9 !important; font-size: 1.3rem !important; margin-top: 0.5rem !important; }

    /* ── Hide default title ── */
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* ── Section subheaders ── */
    h3 { color: #1b5e20 !important; font-size: 1.6rem !important; font-weight: 700 !important; border-bottom: 3px solid #43a047; padding-bottom: 0.3rem; }

    /* ── Body text ── */
    .stMarkdown p, .stMarkdown li { font-size: 1.15rem !important; line-height: 1.7 !important; color: #2e2e2e; }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        border-radius: 14px;
        padding: 1.2rem 1.5rem !important;
        margin-bottom: 1rem !important;
        border: none !important;
    }
    div[data-testid="stMetric"]:nth-child(1) { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border-left: 6px solid #2e7d32 !important; }
    div[data-testid="stMetric"]:nth-child(2) { background: linear-gradient(135deg, #e3f2fd, #bbdefb); border-left: 6px solid #1565c0 !important; }
    div[data-testid="stMetric"]:nth-child(3) { background: linear-gradient(135deg, #fff8e1, #ffecb3); border-left: 6px solid #f57f17 !important; }
    div[data-testid="stMetricLabel"] p { font-size: 1rem !important; color: #555 !important; font-weight: 600 !important; }
    div[data-testid="stMetricValue"]  { font-size: 2rem !important; font-weight: 800 !important; color: #1a1a1a !important; }

    /* ── File uploader ── */
    section[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #f1f8e9, #dcedc8) !important;
        border: 2.5px dashed #558b2f !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        font-size: 1.1rem !important;
    }
    section[data-testid="stFileUploaderDropzone"] button {
        background-color: #2e7d32 !important;
        color: white !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        padding: 0.5rem 1.2rem !important;
    }

    /* ── Progress bars ── */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #43a047, #a5d6a7) !important;
        border-radius: 8px;
        height: 1.2rem !important;
    }
    div[data-testid="stProgress"] p { font-size: 1rem !important; color: #2e2e2e; }

    /* ── Expanders ── */
    div[data-testid="stExpander"] {
        border-radius: 12px !important;
        border: 1.5px solid #c8e6c9 !important;
        background: white !important;
        margin-bottom: 0.8rem !important;
    }
    summary { background: linear-gradient(90deg, #e8f5e9, #f1f8e9) !important; border-radius: 10px !important; padding: 0.8rem 1rem !important; }
    summary p { font-size: 1.2rem !important; font-weight: 700 !important; color: #1b5e20 !important; }
    div[data-testid="stExpander"] .stMarkdown p { font-size: 1.1rem !important; color: #333 !important; }

    /* ── Info box ── */
    div[data-testid="stAlert"] {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9) !important;
        border-left: 6px solid #43a047 !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
    }

    /* ── Image captions ── */
    [data-testid="stImageCaption"] { font-size: 1rem !important; color: #555; text-align: center; }

    /* ── Image containers ── */
    div[data-testid="stImage"] {
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 2px solid #c8e6c9 !important;
        box-shadow: 0 4px 16px rgba(46,125,50,0.12) !important;
    }

    /* ── Divider ── */
    hr { border-color: #c8e6c9 !important; margin: 1.5rem 0 !important; }

    /* ── Page container ── */
    .block-container {
        max-width: 1600px !important;
        padding: 1.5rem 3rem 3rem 3rem !important;
    }
    div[data-testid="column"] { padding: 0 1rem !important; }

    /* ── Top-3 section label ── */
    .top3-label {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin: 1rem 0 0.5rem 0;
    }

    /* ── Spinner ── */
    div[data-testid="stSpinner"] p { font-size: 1.1rem !important; color: #2e7d32 !important; }
</style>
""", unsafe_allow_html=True)

# ── Banner ────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
    <h1>🌿 Plant Disease Detection</h1>
    <p>Upload a leaf image to detect disease and get AI-powered treatment recommendations.</p>
</div>
""", unsafe_allow_html=True)

# ── Groq client ──────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── Load model & assets ──────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

@st.cache_data
def load_assets():
    return np.load("class_names.npy", allow_pickle=True).item()

model       = load_model()
class_names = load_assets()


# ── Prediction ───────────────────────────────────────────────
def predict(img: Image.Image):
    img_r = img.resize((224, 224))
    arr   = np.array(img_r) / 255.0
    arr   = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top3  = np.argsort(preds)[::-1][:3]
    return arr, preds, top3


# ── Grad-CAM ─────────────────────────────────────────────────
def get_gradcam(img_array, model, last_conv_layer_name="Conv_1"):
    base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base = layer
            break
    if base is not None:
        conv_layer      = base.get_layer(last_conv_layer_name)
        base_grad_model = tf.keras.models.Model(
            inputs=base.input, outputs=[conv_layer.output, base.output]
        )
        with tf.GradientTape() as tape:
            conv_out, base_out = base_grad_model(img_array)
            x = base_out
            for layer in model.layers:
                if layer is base or isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                x = layer(x)
            preds       = x
            top_class   = tf.argmax(preds[0])
            class_score = preds[:, top_class]
        grads = tape.gradient(class_score, conv_out)
    else:
        conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layer_name = layer.name
                break
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            top_class       = tf.argmax(preds[0])
            class_score     = preds[:, top_class]
        grads = tape.gradient(class_score, conv_out)

    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap  = conv_out @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)
    heatmap  = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(original_arr, heatmap):
    heatmap_colored = cm.jet(np.uint8(255 * heatmap))[:, :, :3]
    heatmap_resized = tf.image.resize(heatmap_colored, (224, 224)).numpy()
    return np.clip(heatmap_resized * 0.4 + original_arr[0], 0, 1)


# ── Groq recommendation ──────────────────────────────────────
def get_recommendation(predicted_label: str) -> dict:
    if not groq_client:
        return {
            "common_name": predicted_label.replace("___", " — ").replace("_", " "),
            "cause":    "Set GROQ_API_KEY to enable AI recommendations.",
            "symptoms": "—", "severity": "Unknown",
            "prevention": ["Set GROQ_API_KEY for AI-powered tips."],
            "treatment":  ["Set GROQ_API_KEY for AI-powered advice."]
        }
    prompt = f"""You are an agricultural plant disease expert.
The plant disease classifier predicted: "{predicted_label}"
Respond ONLY with a JSON object (no markdown) with keys:
common_name, cause, symptoms, severity (High/Medium/Low/None),
prevention (array of 3-4 tips), treatment (array of 3-4 steps)."""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=512,
        )
        text = resp.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {
            "common_name": predicted_label.replace("___", " — ").replace("_", " "),
            "cause": f"LLM error: {e}", "symptoms": "—", "severity": "Unknown",
            "prevention": ["Consult a local agronomist."],
            "treatment":  ["Contact your agricultural extension service."]
        }


# ── UI ───────────────────────────────────────────────────────
uploaded = st.file_uploader("📁  Upload a leaf image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    arr, preds, top3 = predict(img)
    top_label = class_names[top3[0]]
    top_conf  = preds[top3[0]]

    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🖼️ Input image")
        st.image(img, use_column_width=True)
        st.subheader("🔥 Grad-CAM heatmap")
        heatmap = get_gradcam(arr, model)
        overlay = overlay_heatmap(arr, heatmap)
        st.image(overlay, caption="Highlighted regions that influenced the prediction",
                 use_column_width=True)

    with col2:
        severity_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "None": "🟢", "Unknown": "⚪"}
        with st.spinner("🤖 Asking Groq AI for recommendations..."):
            rec = get_recommendation(top_label)

        st.subheader("📊 Prediction result")
        st.metric("🌿 Detected disease", rec["common_name"])
        st.metric("🎯 Confidence",       f"{top_conf * 100:.1f}%")
        st.metric("⚠️ Severity",         f"{severity_icon.get(rec['severity'], '⚪')} {rec['severity']}")

        st.markdown("---")
        st.markdown('<p class="top3-label">🏆 Top 3 predictions</p>', unsafe_allow_html=True)
        for idx in top3:
            lbl  = class_names[idx].replace("___", " — ").replace("_", " ")
            conf = preds[idx] * 100
            st.progress(int(conf), text=f"{lbl} — {conf:.1f}%")

        st.markdown("---")
        with st.expander("📋 Disease info", expanded=True):
            st.markdown(f"**Cause:** {rec['cause']}")
            st.markdown(f"**Symptoms:** {rec['symptoms']}")
        with st.expander("🛡️ Prevention tips"):
            for tip in rec["prevention"]:
                st.markdown(f"- {tip}")
        with st.expander("💊 Treatment suggestions"):
            for tip in rec["treatment"]:
                st.markdown(f"- {tip}")

else:
    st.markdown("---")
    st.info("👆 Upload a leaf image above to get started.")
    st.markdown("""
    **Supported crops:** Tomato, Potato, Apple, Corn, Grape, Pepper, Strawberry, and more.

    Model trained on the **PlantVillage dataset** with **38 disease classes**.

    Recommendations powered by **Groq LLaMA 3.3 70B**.
    """)
