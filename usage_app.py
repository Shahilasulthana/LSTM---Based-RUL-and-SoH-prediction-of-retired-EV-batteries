import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import plotly.graph_objects as go

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="Smart Battery Usage Advisor",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
        .main { background-color: #f4f6f9; }
        .stButton>button {
            width: 100%;
            background-color: #2b313e;
            color: white;
            padding: 12px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover { background-color: #4a5568; }
        
        /* Metric Card Styling */
        .metric-container {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center;
        }
        
        /* Usage Tip Card Styling */
        .tip-card {
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            border-left: 6px solid #ccc;
            background-color: white;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        .tip-header { font-size: 22px; font-weight: bold; margin-bottom: 10px; }
        .tip-msg { font-size: 16px; font-style: italic; margin-bottom: 15px; color: #555; }
        .tip-list { font-size: 16px; color: #333; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Helper Logic (Usage Guidance)
# ==========================================
def get_usage_guidance(credit_score):
    """Returns guidance dict based on credit score."""
    if credit_score >= 80:
        return {
            "band": "Very Healthy",
            "color": "#4CAF50",  # Green
            "bg_color": "#E8F5E9",
            "icon": "🟢",
            "message": "This battery shows low degradation and stable behavior.",
            "practices": [
                "Prefer slow charge/discharge cycles.",
                "Avoid continuous full-depth discharges.",
                "Maintain moderate operating temperatures."
            ]
        }
    elif credit_score >= 65:
        return {
            "band": "Healthy",
            "color": "#FFC107",  # Amber
            "bg_color": "#FFF8E1",
            "icon": "🟡",
            "message": "This battery shows moderate aging; operate under controlled conditions.",
            "practices": [
                "Limit depth of discharge to ~60–70%.",
                "Avoid frequent high-current operations.",
                "Ensure adequate ventilation during use."
            ]
        }
    elif credit_score >= 45:
        return {
            "band": "Degraded",
            "color": "#FF9800",  # Orange
            "bg_color": "#FFF3E0",
            "icon": "🔵",
            "message": "This battery shows noticeable degradation and reduced efficiency.",
            "practices": [
                "Use strictly for low-power or intermittent loads.",
                "Avoid fast charging entirely.",
                "Keep discharge durations short."
            ]
        }
    else:
        return {
            "band": "Critical",
            "color": "#F44336",  # Red
            "bg_color": "#FFEBEE",
            "icon": "🔴",
            "message": "This battery exhibits severe degradation and elevated risk.",
            "practices": [
                "Continued reuse is NOT recommended and may vary likely fail.",
                "Professional evaluation or recycling is advised."
            ]
        }

# ==========================================
# 3. Load Resources
# ==========================================
@st.cache_resource
def load_resources():
    try:
        # Load Scaler
        scaler = joblib.load('scaler.pkl')
        
        # Load LSTM (RUL/SoH)
        lstm_model = load_model('battery_lstm_model.keras')
        
        # Load LightGBM (Credit Score)
        # Note: In a real deployment, ensure this file exists. 
        # The user just created it in the previous turn.
        credit_model = joblib.load('battery_credit_model_lightGBM.pkl')
        
        return scaler, lstm_model, credit_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

scaler, lstm_model, credit_model = load_resources()

# ==========================================
# 4. Sidebar Inputs
# ==========================================
with st.sidebar:
    st.title("🔋 Battery Inputs")
    st.markdown("Confugure the latest cycle parameters:")
    
    # Default inputs based on a standard test case
    V_avg = st.number_input("Avg Voltage (V)", 0.0, 5.0, 3.5298, format="%.4f")
    V_min = st.number_input("Min Voltage (V)", 0.0, 5.0, 2.6125, format="%.4f")
    V_max = st.number_input("Max Voltage (V)", 0.0, 5.0, 4.1915, format="%.4f")
    I_avg = st.number_input("Avg Current (A)", -10.0, 10.0, -1.8187, format="%.4f")
    T_avg = st.number_input("Avg Temp (°C)", 0.0, 100.0, 32.5723, format="%.4f")
    T_min = st.number_input("Min Temp (°C)", 0.0, 100.0, 24.3260, format="%.4f")
    T_max = st.number_input("Max Temp (°C)", 0.0, 100.0, 38.9822, format="%.4f")
    
    # We also need 'cycle' for the Credit Score model
    cycle = st.number_input("Cycle Count", 0, 2000, 100)

    # Prepare DataFrame
    input_dict = {
        'V_avg': [V_avg], 'V_min': [V_min], 'V_max': [V_max],
        'I_avg': [I_avg],
        'T_avg': [T_avg], 'T_min': [T_min], 'T_max': [T_max],
        'cycle': [cycle] # Used for Credit Score
    }
    input_df = pd.DataFrame(input_dict)
    
    st.write("---")
    analyze = st.button("🔍 Generate Report")

# ==========================================
# 5. Main Dashboard Logic
# ==========================================
st.title("💡 Smart Battery Usage Advisor")
st.markdown("### Diagnosis & Life-Extension Recommendations")
st.markdown("This system analyzes battery telemetry to generate a **Health Credit Score** and actionable **Usage Guidelines**.")

if analyze and scaler and lstm_model and credit_model:
    with st.spinner("Processing telemetry data..."):
        # --- A. Predict RUL & SoH (LSTM) ---
        # LSTM Features: ['V_avg', 'V_min', 'V_max', 'I_avg', 'T_avg', 'T_min', 'T_max']
        lstm_features = ['V_avg', 'V_min', 'V_max', 'I_avg', 'T_avg', 'T_min', 'T_max']
        
        # Scale
        input_norm = scaler.transform(input_df[lstm_features])
        
        # Reshape for LSTM: (1, 20, 7) - Repeat single step 20 times for steady-state shim
        seq_len = 20
        input_seq = np.repeat(input_norm, seq_len, axis=0).reshape(1, seq_len, len(lstm_features))
        
        lstm_preds = lstm_model.predict(input_seq, verbose=0)
        pred_rul = float(lstm_preds[0][0][0])
        pred_soh = float(lstm_preds[1][0][0]) * 100  # Convert to %
        
        # --- B. Predict Credit Score (LightGBM) ---
        # LightGBM Features: ['V_avg', 'V_min', 'V_max', 'I_avg', 'T_avg', 'T_min', 'T_max', 'cycle']
        credit_score = float(credit_model.predict(input_df)[0])
        
        # Clamp Score
        credit_score = max(0, min(100, credit_score))
        
        # Get Guidance
        guidance = get_usage_guidance(credit_score)

    # --- C. Display Results ---
    
    # 1. Scorecard Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color:#777;">Health Credit Score</h4>
            <h1 style="color:{guidance['color']}; font-size: 56px;">{credit_score:.1f}</h1>
            <p>out of 100</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color:#777;">State of Health (SoH)</h4>
            <h1 style="color:#333; font-size: 56px;">{pred_soh:.1f}%</h1>
            <p>Estimated Capacity</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color:#777;">Remaining Useful Life</h4>
            <h1 style="color:#333; font-size: 56px;">{pred_rul:.0f}</h1>
            <p>Cycles Remaining</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")

    # 2. Main Guidance Section
    st.subheader("📋 Usage Recommendations")
    
    guidance_html = f"""
    <div class="tip-card" style="border-left-color: {guidance['color']}; background-color: {guidance['bg_color']};">
        <div class="tip-header" style="color: {guidance['color']};">
            {guidance['icon']} Status: {guidance['band']}
        </div>
        <div class="tip-msg">
            "{guidance['message']}"
        </div>
        <hr style="border-top: 1px dashed {guidance['color']}; opacity: 0.5;">
        <div class="tip-list">
            <b>Recommended Practices:</b>
            <ul>
    """
    for tip in guidance['practices']:
        guidance_html += f"<li>{tip}</li>"
    
    guidance_html += """
            </ul>
        </div>
    </div>
    """
    st.markdown(guidance_html, unsafe_allow_html=True)
    
    # 3. Technical Details Expander
    with st.expander("🔍 View Technical Diagnostics"):
        st.dataframe(input_df)
        st.json({
            "Used Cycle Count": cycle,
            "Predicted RUL": f"{pred_rul:.2f} cycles",
            "Predicted SoH": f"{pred_soh:.2f}%",
            "Credit Score Raw": f"{credit_score:.4f}"
        })

else:
    # Initial State / Instructions
    st.info("👈 Please configure the input parameters in the sidebar and click 'Generate Report'.")
