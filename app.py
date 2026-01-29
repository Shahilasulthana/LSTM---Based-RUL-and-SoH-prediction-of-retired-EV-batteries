
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
import os

# Set Page Configuration
st.set_page_config(
    page_title="Battery Health Monitor",
    page_icon="🔋",
    layout="wide"
)

# Load Model and Scaler (Cached for Performance)
@st.cache_resource
def load_resources():
    try:
        model_path = 'battery_lstm_model.keras'
        scaler_path = 'scaler.pkl'
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}. Please train the model first.")
            return None, None
            
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

model, scaler = load_resources()

# ==========================================
# Sidebar: Input Features
# ==========================================
st.sidebar.header("🔋 Input Parameters")
st.sidebar.markdown("Adjust the battery measurements to predict health.")

def user_input_features():
    # Default values based on a healthy battery (approximate)
    V_avg = st.sidebar.number_input("Average Voltage (V)", min_value=0.0, max_value=5.0, value=3.5298, format="%.4f")
    V_min = st.sidebar.number_input("Min Voltage (V)", min_value=0.0, max_value=5.0, value=2.6125, format="%.4f")
    V_max = st.sidebar.number_input("Max Voltage (V)", min_value=0.0, max_value=5.0, value=4.1915, format="%.4f")
    I_avg = st.sidebar.number_input("Average Current (A)", min_value=-10.0, max_value=10.0, value=-1.8187, format="%.4f")
    T_avg = st.sidebar.number_input("Average Temp (°C)", min_value=0.0, max_value=100.0, value=32.5723, format="%.4f")
    T_min = st.sidebar.number_input("Min Temp (°C)", min_value=0.0, max_value=100.0, value=24.3260, format="%.4f")
    T_max = st.sidebar.number_input("Max Temp (°C)", min_value=0.0, max_value=100.0, value=38.9822, format="%.4f")

    data = {
        'V_avg': [V_avg],
        'V_min': [V_min],
        'V_max': [V_max],
        'I_avg': [I_avg],
        'T_avg': [T_avg],
        'T_min': [T_min],
        'T_max': [T_max]
    }
    return pd.DataFrame(data)

input_df = user_input_features()

# ==========================================
# Main Dashboard
# ==========================================
st.title("🔋 Battery Health & RUL Predictor")
st.markdown("Real-time AI prediction of **State of Health (SoH)** and **Remaining Useful Life (RUL)**.")

if st.button("Analyze Battery Health", type="primary"):
    if model is not None and scaler is not None:
        # Preprocess
        norm_features = scaler.transform(input_df)
        
        # Reshape for LSTM (Samples, Time_Steps, Features)
        # Create sequence by repeating the single time step
        SEQUENCE_LENGTH = 20
        input_sequence = np.repeat(norm_features, SEQUENCE_LENGTH, axis=0) # Shape (20, 7)
        input_sequence = input_sequence.reshape(1, SEQUENCE_LENGTH, norm_features.shape[1]) # Shape (1, 20, 7)
        
        # Predict
        predictions = model.predict(input_sequence, verbose=0)
        pred_rul = float(predictions[0][0][0])
        pred_soh = float(predictions[1][0][0])

        # Post-process
        pred_soh = max(0.0, min(1.0, pred_soh)) # Clip SoH
        rul_months = pred_rul / 30.0
        soh_percentage = pred_soh * 100

        # Determine Status
        if pred_soh >= 0.80:
            status = "Healthy"
            color = "green"
        elif pred_soh >= 0.60:
            status = "Warning"
            color = "orange"
        else:
            status = "Critical (Replace)"
            color = "red"

        # --- Visualization Layout ---
        col1, col2, col3 = st.columns(3)

        # 1. SoH Gauge Chart
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = soh_percentage,
                title = {'text': "State of Health (SoH)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 60], 'color': "#ffcccc"},
                        {'range': [60, 80], 'color': "#ffeebb"},
                        {'range': [80, 100], 'color': "#ccffcc"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': soh_percentage
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # 2. RUL Metric Card
        with col2:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; margin-top: 50px;">
                <h3 style="color: #555;">Remaining Useful Life</h3>
                <h1 style="color: #333; font-size: 60px;">{pred_rul:.0f}</h1>
                <p style="font-size: 20px; color: #777;">Cycles (~{rul_months:.1f} Months)</p>
            </div>
            """, unsafe_allow_html=True)

        # 3. Status Card
        with col3:
            st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; margin-top: 50px; color: white;">
                <h3>Battery Status</h3>
                <h1 style="font-size: 50px;">{status}</h1>
                <p style="font-size: 18px;">Recommendation: { "Normal Operation" if status == "Healthy" else "Schedule Maintenance" }</p>
            </div>
            """, unsafe_allow_html=True)

        # --- Detailed Data View ---
        st.subheader("📝 Input Summary")
        st.dataframe(input_df, hide_index=True)

    else:
        st.warning("Model resources failed to load. Check file paths.")
else:
    st.info("Adjust parameters in the sidebar and click 'Analyze Battery Health' to see results.")
