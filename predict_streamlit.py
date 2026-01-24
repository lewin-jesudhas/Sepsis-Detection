import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model

# ---------------- Load model & threshold ----------------
@st.cache_resource
def load_assets():
    model = load_model("sepsis_bigru_model_lime.h5")
    with open("best_threshold.json", "r") as f:
        threshold = json.load(f)["threshold"]
    return model, threshold

model, best_threshold = load_assets()

st.title("🧪 Early Sepsis Prediction (Hourly Input)")
st.markdown("Enter **one hour of patient data at a time** (41 features).")

# ---------------- Session state ----------------
if "sequence" not in st.session_state:
    st.session_state.sequence = []

# ---------------- Feature input ----------------
st.subheader("Hourly Feature Input")

input_text = st.text_area(
    "Input:",
    height=120,
    placeholder="e.g. 80,98,36.7,120,..."
)

# ---------------- Add hour ----------------
if st.button("➕ Add Hour Data"):
    try:
        values = [float(x.strip()) for x in input_text.split(",")]
        if len(values) != 41:
            st.error("❌ Exactly 41 values required")
        else:
            st.session_state.sequence.append(values)
            st.success(f"✅ Hour {len(st.session_state.sequence)} added")
    except:
        st.error("❌ Invalid input format")

# ---------------- Show progress ----------------
st.write(f"🕒 Hours entered: **{len(st.session_state.sequence)} / 12**")

# ---------------- Prediction ----------------
# ---------------- Prediction ----------------
if len(st.session_state.sequence) > 0:
    seq = np.array(st.session_state.sequence, dtype=np.float32)

    T_expected = model.input_shape[1]   # e.g. 12
    F_expected = model.input_shape[2]   # e.g. 42

    # -------- FIX FEATURE COUNT FIRST --------
    F_actual = seq.shape[1]

    if F_actual < F_expected:
        pad_feat = np.zeros((seq.shape[0], F_expected - F_actual), dtype=np.float32)
        seq = np.hstack([seq, pad_feat])

    elif F_actual > F_expected:
        seq = seq[:, :F_expected]

    # -------- FIX TIMESTEPS NEXT --------
    if T_expected is not None:
        if seq.shape[0] < T_expected:
            pad_time = np.zeros((T_expected - seq.shape[0], F_expected), dtype=np.float32)
            seq = np.vstack([pad_time, seq])
        elif seq.shape[0] > T_expected:
            seq = seq[-T_expected:, :]

    # Final shape: (1, T, F)
    seq = seq[np.newaxis, :, :]

    # -------- Predict --------
    y_prob = model.predict(seq, verbose=0)[0][0]
    y_pred = int(y_prob >= best_threshold)

    st.subheader("📊 Current Prediction")
    st.metric("Sepsis Probability", f"{y_prob:.4f}")

    if y_pred == 1:
        st.error("🚨 Sepsis Detected – Early Warning")
    else:
        st.success("✅ No Sepsis Detected")


# ---------------- Reset ----------------
if st.button("🔄 Reset Patient"):
    st.session_state.sequence = []
    st.experimental_rerun()
