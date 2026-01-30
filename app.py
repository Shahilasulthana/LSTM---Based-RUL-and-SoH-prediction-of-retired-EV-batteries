
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
    page_title="Battery Health Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancements
st.markdown("""
    <style>
        .main {
            background-color: #FAFAFA;
        }
        .stButton>button {
            width: 100%;
            background-color: #007BFF;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 8px;
            border: None;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            color: white;
        }
        h1 {
            color: #2C3E50;
        }
        h3 {
            color: #34495E;
        }
        .css-1d391kg {
            padding-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

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
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/234/234699.png", width=100) # Placeholder Battery Icon
    st.header("🔋 Battery Parameters")
    st.markdown("Enter the latest cycle measurements to analyze usage capability.")
    st.write("---")

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
st.title("⚡ Battery Health Predictor & Recommender")
st.markdown("### AI-Powered Second-Life Assessment")
st.markdown("""
This tool utilizes a **Deep LSTM Neural Network** to analyze battery sensor data. 
It predicts the **Remaining Useful Life (RUL)** and **State of Health (SoH)** to recommend safe second-life applications.
""")
st.write("---")

col_action_1, col_action_2, col_action_3 = st.columns([1, 2, 1])
with col_action_2:
    analyze_btn = st.button("🔍 Analyze Battery Health & Capabilities", type="primary")

if analyze_btn:
    if model is not None and scaler is not None:
        with st.spinner("Analyzing battery signals..."):
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

        # Determine Capability Band & Recommendations
        if pred_soh >= 0.70:
            band = "Band A"
            recs = "Home backup, Small solar storage, Off-grid residential loads"
            color = "#2E7D32" # Dark Green
            bg_color = "#E8F5E9"
            deployment_class = "Moderate-demand stationary systems"
        elif pred_soh >= 0.55:
            band = "Band B"
            recs = "Backup power systems, Telecom auxiliary storage, Emergency or contingency power"
            color = "#F9A825" # Dark Amber
            bg_color = "#FFFDE7"
            deployment_class = "Intermittent, backup-oriented energy profiles"
        elif pred_soh >= 0.40:
            band = "Band C"
            recs = "Low-power DC systems, Small electronics backup, Educational or experimental setups"
            color = "#EF6C00" # Dark Orange
            bg_color = "#FFF3E0"
            deployment_class = "Low-stress, non-continuous usage"
        else:
            band = "Band D"
            recs = "Not suitable for second-life deployment, Requires controlled recycling"
            color = "#C62828" # Dark Red
            bg_color = "#FFEBEE"
            deployment_class = "Recycle"

        # --- Visualization Layout ---
        st.success("Analysis Complete!")
        
        # Top Row: Gauge and Key Metrics
        col1, col2 = st.columns([1, 1])

        # 1. SoH Gauge Chart
        with col1:
            st.markdown(f"<h4 style='text-align: center;'>State of Health (SoH)</h4>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = soh_percentage,
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 40], 'color': "#EF9A9A"},
                        {'range': [40, 55], 'color': "#FFE0B2"},
                        {'range': [55, 70], 'color': "#FFF59D"},
                        {'range': [70, 100], 'color': "#A5D6A7"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': soh_percentage
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_gauge, width="stretch")

        # 2. Key Metrics Card
        with col2:
            st.markdown(f"<h4 style='text-align: center;'>Prediction Summary</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 25px; border-radius: 15px; border: 2px solid {color}; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align_items: center; margin-bottom: 15px;">
                    <span style="font-size: 18px; color: #555; font-weight: 600;">Remaining Useful Life</span>
                    <span style="font-size: 26px; font-weight: bold; color: #333;">{pred_rul:.0f} Cycles</span>
                </div>
                <div style="font-size: 15px; color: #666; margin-bottom: 20px; text-align: right;">(approx. {rul_months:.1f} Months)</div>
                
                <hr style="margin: 15px 0; border: 0; border-top: 1px dashed {color};">
                
                <div style="display: flex; justify-content: space-between; align_items: center; margin-bottom: 5px;">
                    <span style="font-size: 18px; color: #555; font-weight: 600;">Capability Band</span>
                    <span style="font-size: 26px; font-weight: bold; color: {color};">{band}</span>
                </div>
                <div style="font-size: 14px; color: #555; font-style: italic; margin-bottom: 5px; text-align: right;">
                    {deployment_class}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("---")

        # 3. Recommended Applications (Full Width)
        st.markdown(f"<h3 style='text-align: center; color: {color};'>🎯 Recommended Applications</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 30px; border-radius: 15px; border: 2px solid {color}; text-align: center; margin-top: 10px;">
            <p style="font-size: 22px; color: #333; font-weight: 500; margin: 0;">{recs}</p>
        </div>
        """, unsafe_allow_html=True)

        # --- Detailed Data View ---
        st.write("")
        with st.expander("📝 View Input Signal Summary"):
            st.dataframe(input_df, hide_index=True)

    else:
        st.warning("Model resources failed to load. Check file paths.")
else:
    st.info("👈 Adjust battery parameters in the sidebar and click **Analyze** to generate a recommendation.")
