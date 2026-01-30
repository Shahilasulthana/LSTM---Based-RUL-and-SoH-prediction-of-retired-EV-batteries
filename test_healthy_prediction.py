
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel(logging.ERROR)

def predict_healthy_test():
    try:
        model = load_model('battery_lstm_model.keras')
        scaler = joblib.load('scaler.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Healthy Input (Cycle 1, B0005)
    # V_avg: 3.5298, V_min: 2.6125, V_max: 4.1915, I_avg: -1.8187, T_avg: 32.5723, T_min: 24.3260, T_max: 38.9822
    
    features = ['V_avg', 'V_min', 'V_max', 'I_avg', 'T_avg', 'T_min', 'T_max']
    input_data = {
        'V_avg': [3.5298],
        'V_min': [2.6125],
        'V_max': [4.1915],
        'I_avg': [-1.8187],
        'T_avg': [32.5723],
        'T_min': [24.3260],
        'T_max': [38.9822]
    }
    
    # 4. Preprocess Input
    input_df = pd.DataFrame(input_data)
    norm_features = scaler.transform(input_df)
    
    # Reshape for LSTM (Samples, Time_Steps, Features)
    # The model expects a sequence history. Since we only have a single point, 
    # we repeat it to simulate a "steady state" sequence.
    SEQUENCE_LENGTH = 20
    input_sequence = np.repeat(norm_features, SEQUENCE_LENGTH, axis=0) # Shape (20, 7)
    input_sequence = input_sequence.reshape(1, SEQUENCE_LENGTH, norm_features.shape[1]) # Shape (1, 20, 7) 

    # 5. Predict
    predictions = model.predict(input_sequence, verbose=0)
    pred_rul = predictions[0][0][0]
    pred_soh = predictions[1][0][0]

    # Clip SoH (0 to 1)
    pred_soh = max(0.0, min(1.0, pred_soh))

    # Convert RUL to months
    rul_months = pred_rul / 30.0

    print("\n--- Final Validation Test (Healthy Battery Input) ---")
    print(f"Predicted SoH: {pred_soh:.4f} ({pred_soh*100:.2f}%)")
    print(f"Predicted RUL: {pred_rul:.1f} cycles (~{rul_months:.1f} months)")
    
    # 6. Apply Capability Band Logic
    if pred_soh >= 0.70:
        band = 'Band A'
        recs = 'Home backup, Small solar storage, Off-grid residential loads'
    elif pred_soh >= 0.55:
        band = 'Band B'
        recs = 'Backup power systems, Telecom auxiliary storage, Emergency or contingency power'
    elif pred_soh >= 0.40:
        band = 'Band C'
        recs = 'Low-power DC systems, Small electronics backup, Educational or experimental setups'
    else:
        band = 'Band D'
        recs = 'Not suitable for second-life deployment, Requires controlled recycling'

    print(f"Capability Band: {band}")
    print(f"Recommended Applications: {recs}")
    
    print("-----------------------------------------------------")

if __name__ == "__main__":
    predict_healthy_test()
