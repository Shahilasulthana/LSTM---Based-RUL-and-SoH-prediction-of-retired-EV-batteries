
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Load Model and Scaler
MODEL_PATH = 'battery_lstm_model.keras'
SCALER_PATH = 'scaler.pkl'

print("Loading model and scaler...")
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    print("Please ensure you have run 'battery_rul_soh_prediction.py' first.")
    exit()

# 2. Define Features
features = ['V_avg', 'V_min', 'V_max', 'I_avg', 'T_avg', 'T_min', 'T_max']

print("\n--- Manual Battery RUL & SoH Prediction ---")
print("Please enter the following feature values for a single cycle:")

# 3. Get User Input
input_data = {}
try:
    # Providing default example values in prompt
    print("\n(Example values: V_avg=3.5, V_min=2.0, V_max=4.2, I_avg=-2.0, T_avg=32.0, T_min=22.0, T_max=40.0)")
    
    for feature in features:
        val = float(input(f"Enter {feature}: "))
        input_data[feature] = [val]
        
except ValueError:
    print("Invalid input. Please enter numeric values.")
    exit()

# 4. Preprocess Input
input_df = pd.DataFrame(input_data)

# Normalize
norm_features = scaler.transform(input_df)

# Reshape for LSTM (Samples, Time_Steps, Features)
# The model expects a sequence history. Since we only have a single point, 
# we repeat it to simulate a "steady state" sequence.
SEQUENCE_LENGTH = 20
input_sequence = np.repeat(norm_features, SEQUENCE_LENGTH, axis=0) # Shape (20, 7)
input_sequence = input_sequence.reshape(1, SEQUENCE_LENGTH, len(features)) # Shape (1, 20, 7)

# 5. Predict
print("\nPredicting...")
predictions = model.predict(input_sequence)
pred_rul = predictions[0][0][0]
pred_soh = predictions[1][0][0]

# Clip SoH to be physically meaningful (0 to 1)
pred_soh = max(0.0, min(1.0, pred_soh))

# Convert RUL to months (Approximation: 1 cycle/day -> 30 cycles/month)
rul_months = pred_rul / 30.0

print("\n-----------------------------")
print(f"Predicted State of Health (SoH): {pred_soh:.4f} ({pred_soh*100:.2f}%)")
print(f"Predicted Remaining Useful Life (RUL): {pred_rul:.1f} cycles (~{rul_months:.1f} months)")
print("-----------------------------")

# Interpretation
if pred_soh < 0.80:
    print("Status: Unhealthy / Replace Battery")
else:
    print("Status: Healthy")
