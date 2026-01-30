# 🔋 AI-Powered Battery Health & Second-Life Predictor

![Battery Health](https://img.shields.io/badge/Battery-Health_Monitor-green?style=for-the-badge&logo=battery)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit)
![TensorFlow](https://img.shields.io/badge/AI-TensorFlow_LSTM-orange?style=for-the-badge&logo=tensorflow)

## 📌 Overview
This project is an advanced **Battery Management System (BMS)** analytics tool designed to assess the health of retired EV batteries. It goes beyond simple metrics by providing a **Battery Credit Score**, calculating **Remaining Useful Life (RUL)**, and offering actionable **Second-Life Usage Recommendations**.

Using a hybrid AI approach (**Deep LSTM** for time-series forecasting + **LightGBM** for scoring), the system helps users decide whether a battery is suitable for *home backup*, *low-power storage*, or *recycling*.

---

## 🌟 Key Features

### 1. 🧠 Hybrid AI Models
- **LSTM Neural Network**: Predicts **State of Health (SoH)** and **Remaining Useful Life (RUL)** based on raw telemetry data (Voltage, Current, Temperature).
- **LightGBM Regressor**: Calculates a comprehensive **Battery Credit Score (0-100)** to grade overall capability.

### 2. 🚦 Smart Usage Recommendations
Instead of generic labels, the system assigns a simplified **Capability Band** with specific advice:
- **🟢 Very Healthy (80-100)**: Suitable for critical home backup & solar storage.
- **🟡 Healthy (65-80)**: Good for controlled, moderate load applications.
- **🔵 Degraded (45-65)**: Restricted to low-power, intermittent use.
- **🔴 Critical (< 45)**: End-of-life; recommended for recycling.

### 3. 🔍 Explainable AI (XAI)
- Powered by **SHAP (SHapley Additive exPlanations)**.
- Tells you *why* a battery got a specific score (e.g., *"High temperature history reduced the score by 12 points"*).
- Provides transparency for decision-making.

### 4. 📊 Interactive Dashboard
- Built with **Streamlit** for real-time interaction.
- Visualizes Score, RUL, and SoH.
- Actionable "Tip Cards" for maximizing battery life.

---

## 🛠️ Tech Stack
- **Languages**: Python
- **ML/DL Frameworks**: TensorFlow (Keras), LightGBM, Scikit-Learn
- **Explainability**: SHAP
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Web App**: Streamlit
- **Data Handling**: Pandas, NumPy

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `usage_app.py` | 🚀 **Main Application**. Run this to launch the Smart Advisor. |
| `battery_rul_soh_prediction.py` | Training script for the **LSTM (RUL/SoH)** model. |
| `usage_tips.py` | Training script for the **Credit Score (LightGBM)** model & SHAP analysis. |
| `package_usage_tips.py` | Helper script to bundle logic into `usage_tips.pkl`. |
| `battery_lstm_model.keras` | Saved LSTM model artifact. |
| `usage_tips.pkl` | Saved "All-in-One" Advisor artifact (Model + Rules + Explainer). |
| `validate_dataset.py` | Utility to check dataset integrity. |

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Shahilasulthana/LSTM---Based-RUL-and-SoH-prediction-of-retired-EV-batteries.git
cd rul_soh_predictor
pip install -r requirements.txt
```
*(Note: Ensure you have `tensorflow`, `streamlit`, `shap`, `lightgbm`, `joblib`, `plotly` installed)*

### 2. Run the Web Application
Launch the Smart Advisor dashboard:
```bash
streamlit run usage_app.py
```

### 3. (Optional) Retrain Models
If you want to retrain the models on new data:
```bash
# 1. Train LSTM Model
python battery_rul_soh_prediction.py

# 2. Train Credit Score Model & Explainability
python usage_tips.py

# 3. Package the Advisor
python package_usage_tips.py
```

---

## 📊 How It Works (The "Credit Score" Logic)

The system evaluates the battery based on telemetry features:
- **Voltage Stability**: `V_avg`, `V_min`, `V_max`
- **Thermal Stress**: `T_avg`, `T_max`
- **Usage History**: `Cycle Count`

It then outputs a **Credit Score** that maps to a guidance band:

| Score | Band | Icon | Recommendation |
|-------|------|------|----------------|
| **80-100** | Very Healthy | 🟢 | **Stable.** Safe for high-demand loads. |
| **65-80** | Healthy | 🟡 | **Good.** Limit full depth of discharge. |
| **45-65** | Degraded | 🔵 | **Fair.** Low power usage only. |
| **< 45** | Critical | 🔴 | **Risk.** Recycle immediately. |

---

## 🔮 Future Improvements
- [ ] Integration with real-time BMS hardware.
- [ ] Cloud deployment for remote battery monitoring.
- [ ] Advanced anomaly detection for safety alerts.

---
*Created by Shahila Sulthana*
