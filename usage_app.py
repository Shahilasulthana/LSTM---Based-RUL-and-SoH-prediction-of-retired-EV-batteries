import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import plotly.graph_objects as go
import shap
import warnings

# Use plotting backend compatible with Streamlit if needed, though we use Plotly
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

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
        
        /* Explainability Styling */
        .impact-positive { color: #2E7D32; font-weight: bold; }
        .impact-negative { color: #C62828; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Battery Advisor Logic Class
# ==========================================
class BatteryAdvisor:
    """
    All-in-One Battery Advisor Engine.
    Contains: ML Model, Rule Logic, and SHAP Explainer.
    """
    def __init__(self, model_path='battery_credit_model_lightGBM.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        
        self.model = joblib.load(model_path)
        self.explainer = None 
        
    def _init_explainer(self):
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)

    def get_usage_guidance(self, credit_score):
        if credit_score >= 80:
            return {
                "band": "Very Healthy",
                "color": "#4CAF50", "bg_color": "#E8F5E9", "icon": "🟢",
                "message": "This battery shows low degradation and stable behavior.",
                "practices": ["Prefer slow charge/discharge.", "Avoid continuous full-depth cycles.", "Maintain moderate temps."]
            }
        elif credit_score >= 65:
            return {
                "band": "Healthy",
                "color": "#FFC107", "bg_color": "#FFF8E1", "icon": "🟡",
                "message": "Moderate aging; operate under controlled conditions.",
                "practices": ["Limit DoD to ~60–70%.", "Avoid high-current ops.", "Ensure ventilation."]
            }
        elif credit_score >= 45:
            return {
                "band": "Degraded",
                "color": "#FF9800", "bg_color": "#FFF3E0", "icon": "🔵",
                "message": "Noticeable degradation; reduced efficiency.",
                "practices": ["Low-power loads only.", "No fast charging.", "Short discharge durations."]
            }
        else:
            return {
                "band": "Critical",
                "color": "#F44336", "bg_color": "#FFEBEE", "icon": "🔴",
                "message": "Severe degradation; elevated risk.",
                "practices": ["Reuse NOT recommended.", "Recycling advised."]
            }

    def predict(self, input_features):
        if isinstance(input_features, dict):
            input_features = pd.DataFrame([input_features])
        
        raw_score = self.model.predict(input_features)[0]
        credit_score = max(0, min(100, raw_score))
        guidance = self.get_usage_guidance(credit_score)
        
        return {"score": credit_score, "guidance": guidance}

    def explain(self, input_features):
        self._init_explainer()
        if isinstance(input_features, dict):
            input_features = pd.DataFrame([input_features])
            
        shap_values = self.explainer(input_features)
        
        features = input_features.columns
        impacts = shap_values.values[0]
        
        explanations = []
        for feat, impact in sorted(zip(features, impacts), key=lambda x: abs(x[1]), reverse=True):
            if abs(impact) < 0.01: continue 
            direction = "INCREASED" if impact > 0 else "DECREASED"
            val = input_features[feat].values[0]
            explanations.append({
                "feature": feat, "value": val, "impact": impact,
                "direction": direction,
                "text": f"{feat} ({val:.2f}) {direction} score by {abs(impact):.2f}"
            })
        return explanations

# ==========================================
# 3. Load Resources
# ==========================================
@st.cache_resource
def load_resources():
    try:
        # Load Scaler
        scaler = joblib.load('scaler.pkl')
        
        # Load LSTM
        lstm_model = load_model('battery_lstm_model.keras')
        
        # Initialize Advisor (Loads LightGBM internally)
        advisor = BatteryAdvisor('battery_credit_model_lightGBM.pkl')
        
        return scaler, lstm_model, advisor
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None

scaler, lstm_model, advisor = load_resources()

# ==========================================
# 4. Sidebar Inputs
# ==========================================
with st.sidebar:
    st.title("🔋 Battery Inputs")
    st.markdown("Configure parameters:")
    
    # Default inputs based on a standard test case
    V_avg = st.number_input("Avg Voltage (V)", 0.0, 5.0, 3.5298, format="%.4f")
    V_min = st.number_input("Min Voltage (V)", 0.0, 5.0, 2.6125, format="%.4f")
    V_max = st.number_input("Max Voltage (V)", 0.0, 5.0, 4.1915, format="%.4f")
    I_avg = st.number_input("Avg Current (A)", -10.0, 10.0, -1.8187, format="%.4f")
    T_avg = st.number_input("Avg Temp (°C)", 0.0, 100.0, 32.5723, format="%.4f")
    T_min = st.number_input("Min Temp (°C)", 0.0, 100.0, 24.3260, format="%.4f")
    T_max = st.number_input("Max Temp (°C)", 0.0, 100.0, 38.9822, format="%.4f")
    cycle = st.number_input("Cycle Count", 0, 2000, 100)

    input_dict = {
        'V_avg': [V_avg], 'V_min': [V_min], 'V_max': [V_max],
        'I_avg': [I_avg],
        'T_avg': [T_avg], 'T_min': [T_min], 'T_max': [T_max],
        'cycle': [cycle]
    }
    input_df = pd.DataFrame(input_dict)
    
    st.write("---")
    analyze = st.button("🔍 Generate Report", type="primary")

# ==========================================
# 5. Main Dashboard Logic
# ==========================================
st.title("💡 Smart Battery Usage Advisor")
st.markdown("### Diagnosis & Life-Extension Recommendations")

if analyze and advisor:
    with st.spinner("Analyzing telemetry & generating explanation..."):
        # --- A. Predict RUL & SoH (LSTM) ---
        lstm_features = ['V_avg', 'V_min', 'V_max', 'I_avg', 'T_avg', 'T_min', 'T_max']
        input_norm = scaler.transform(input_df[lstm_features])
        input_seq = np.repeat(input_norm, 20, axis=0).reshape(1, 20, 7)
        lstm_preds = lstm_model.predict(input_seq, verbose=0)
        pred_rul = float(lstm_preds[0][0][0])
        pred_soh = float(lstm_preds[1][0][0]) * 100

        # --- B. Advisor Prediction ---
        result = advisor.predict(input_df)
        score = result['score']
        guidance = result['guidance']
        
        # --- C. Explainability ---
        explanations = advisor.explain(input_df)

    # --- ROW 1: Scorecards ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color:#777;">Health Credit Score</h4>
            <h1 style="color:{guidance['color']}; font-size: 56px;">{score:.1f}</h1>
            <p>out of 100</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color:#777;">State of Health (SoH)</h4>
            <h1 style="color:#333; font-size: 56px;">{pred_soh:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color:#777;">RUL Estimate</h4>
            <h1 style="color:#333; font-size: 56px;">{pred_rul:.0f}</h1>
            <p>Cycles</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")

    # --- ROW 2: Recommendations ---
    st.subheader("📋 Usage Recommendations")
    guidance_html = f"""
    <div class="tip-card" style="border-left-color: {guidance['color']}; background-color: {guidance['bg_color']};">
        <div class="tip-header" style="color: {guidance['color']};">
            {guidance['icon']} Status: {guidance['band']}
        </div>
        <div class="tip-msg">"{guidance['message']}"</div>
        <div class="tip-list">
            <b>Recommended Practices:</b><ul>
    """
    for tip in guidance['practices']:
        guidance_html += f"<li>{tip}</li>"
    guidance_html += "</ul></div></div>"
    st.markdown(guidance_html, unsafe_allow_html=True)

    # --- ROW 3: Explainability ---
    st.subheader("🧠 Explainability Analysis")
    st.markdown("Why did the AI assign this score? Here are the top contributing factors:")
    
    # Visualization: Horizontal Bar Chart of Impacts using Plotly
    if explanations:
        features = [x['feature'] for x in explanations[:8]] # Top 8
        impacts = [x['impact'] for x in explanations[:8]]
        colors = ['#2E7D32' if x > 0 else '#C62828' for x in impacts]
        
        fig = go.Figure(go.Bar(
            x=impacts,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f"{x:.2f}" for x in impacts],
            textposition='auto'
        ))
        fig.update_layout(
            title="Feature Impact on Credit Score (SHAP Values)",
            xaxis_title="Impact on Score (Points)",
            yaxis_title="Feature",
            yaxis={'categoryorder':'total ascending'},
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Textual Breakdown
        with st.expander("📝 See Detailed Impact Breakdown"):
            for exp in explanations:
                css_class = "impact-positive" if exp['impact'] > 0 else "impact-negative"
                icon = "⬆️" if exp['impact'] > 0 else "⬇️"
                st.markdown(f"- **{exp['feature']}** ({exp['value']:.2f}): <span class='{css_class}'>{icon} {exp['direction']}</span> score by {abs(exp['impact']):.2f}", unsafe_allow_html=True)

else:
    st.info("👈 Configure inputs to see the AI diagnosis.")
