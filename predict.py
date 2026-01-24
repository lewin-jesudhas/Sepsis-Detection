from tensorflow.keras.models import load_model
import numpy as np
import json

# --- Load model & threshold ---
model = load_model("sepsis_bigru_model_lime.h5")
with open("best_threshold.json", "r") as f:
    best_threshold = json.load(f)["threshold"]
print("Using threshold:", best_threshold)

# --- Your sequence here (12 x 41) ---
new_patient_sequence = [
    [80, 98, 36.7, 120, 80, 93, 18, 35, 1, 24, 0.21, 7.40, 40, 98, 25, 1, 10, 12, 0.5, 2, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [82, 97, 36.8, 118, 78, 92, 19, 36, 0, 23, 0.22, 7.39, 42, 97, 24, 1, 11, 13, 0.5, 2.1, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [81, 97, 36.7, 119, 79, 92, 18, 35, 0, 23, 0.21, 7.40, 41, 97, 24, 1, 10, 12, 0.5, 2, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [79, 96, 36.6, 117, 77, 91, 17, 34, 0, 23, 0.21, 7.39, 39, 96, 23, 1, 9, 12, 0.5, 2, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [78, 97, 36.7, 118, 78, 92, 18, 35, 0, 23, 0.21, 7.39, 40, 97, 24, 1, 10, 12, 0.5, 2, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [80, 98, 36.8, 119, 79, 93, 18, 35, 1, 24, 0.21, 7.40, 41, 98, 25, 1, 10, 12, 0.5, 2, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [81, 97, 36.7, 118, 78, 92, 18, 34, 0, 23, 0.21, 7.39, 42, 97, 24, 1, 11, 13, 0.5, 2.1, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [82, 96, 36.6, 117, 77, 91, 17, 35, 0, 23, 0.21, 7.39, 41, 96, 23, 1, 10, 12, 0.5, 2, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [80, 97, 36.7, 118, 78, 92, 18, 34, 0, 23, 0.21, 7.40, 40, 97, 24, 1, 10, 12, 0.5, 2, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [79, 96, 36.6, 117, 77, 91, 17, 35, 0, 23, 0.21, 7.39, 39, 96, 23, 1, 9, 12, 0.5, 2, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [80, 97, 36.7, 118, 78, 92, 18, 34, 0, 23, 0.21, 7.40, 40, 97, 24, 1, 10, 12, 0.5, 2, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [81, 98, 36.8, 119, 79, 93, 18, 35, 1, 24, 0.21, 7.40, 41, 98, 25, 1, 10, 12, 0.5, 2, 0.1, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# --- Make numpy array, float32 ---
seq = np.array(new_patient_sequence, dtype=np.float32)

# --- Inspect expected input shape ---
# model.input_shape is typically (None, timesteps, features)
inp_shape = model.input_shape
T_expected = inp_shape[1]  # can be None (variable length) or a fixed int
F_expected = inp_shape[2]

print(f"Model expects: timesteps={T_expected}, features={F_expected}")
print(f"Given: timesteps={seq.shape[0]}, features={seq.shape[1]}")

# --- Fix timesteps if model was built with a fixed length ---
if T_expected is not None:
    if seq.shape[0] > T_expected:
        # keep the most recent T_expected hours
        seq = seq[-T_expected:, :]
    elif seq.shape[0] < T_expected:
        # pad at the front with zeros (or your training-time pad value)
        pad = np.zeros((T_expected - seq.shape[0], seq.shape[1]), dtype=np.float32)
        seq = np.vstack([pad, seq])

# --- Fix feature count to match the model ---
F_actual = seq.shape[1]
if F_actual > F_expected:
    # Trim extra columns (ideally, map using the exact training feature order)
    seq = seq[:, :F_expected]
elif F_actual < F_expected:
    # Pad missing columns with zeros (or training-time defaults)
    padf = np.zeros((seq.shape[0], F_expected - F_actual), dtype=np.float32)
    seq = np.hstack([seq, padf])

# Final batch shape: (1, T, F)
new_patient = seq[np.newaxis, :, :]

# --- Predict ---
y_prob = model.predict(new_patient)
y_pred = (y_prob >= best_threshold).astype(int)

status = "✅ Sepsis Detected" if y_pred[0][0] == 1 else "❌ No Sepsis Detected"
print("Sepsis Prediction:", status)
print("Predicted probability of sepsis:", float(y_prob[0][0]))
