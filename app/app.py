"""PPE detector demo.  Run from the project root:  streamlit run app/app.py"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `import src...` works when launched from anywhere

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.inference import ONNX_PATH, PT_PATH, load_model, predict

st.set_page_config(page_title="PPE Detector", page_icon="🦺", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


st.title("🦺 PPE / Safety-Gear Detector")
st.caption(
    "YOLO11n trained on SH17 (17 classes) at 960 px. "
    "Runs on CPU, so expect about 0.1 s per image."
)

if not (ONNX_PATH.exists() or PT_PATH.exists()):
    st.error(f"No model found. Put final_yolo11n_960.onnx (or .pt) in {ONNX_PATH.parent}.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
    st.markdown(
        "**Tip:** small, distant objects (helmets, vests) often score low. "
        "Lower the threshold to see them, at the cost of more false positives."
    )
    st.markdown("---")
    st.markdown(
        "Known weak spots: safety-vest, ear-mufs, foot, tool and rare suit classes. "
        "See the README failure analysis."
    )

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Upload a photo of workers to start.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
image_np = np.array(image)

model = get_model()
with st.spinner("Detecting..."):
    annotated, counts, detections = predict(model, image_np, conf=conf)

left, right = st.columns(2)
left.subheader("Input")
left.image(image, use_container_width=True)
right.subheader("Detections")
right.image(annotated, use_container_width=True)

st.subheader(f"{len(detections)} objects found at confidence >= {conf:.2f}")
if detections:
    c1, c2 = st.columns([1, 2])
    c1.dataframe(
        pd.DataFrame(counts.most_common(), columns=["class", "count"]),
        hide_index=True, use_container_width=True,
    )
    c2.dataframe(pd.DataFrame(detections), hide_index=True, use_container_width=True)

    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button("Download annotated image", buf.getvalue(),
                       file_name="detections.png", mime="image/png")
else:
    st.warning("Nothing detected. Try lowering the confidence threshold.")