import joblib
import pandas as pd
import numpy as np
import shap
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class BatteryAdvisor:
    """
    All-in-One Battery Advisor Engine.
    Contains:
    1. Trained ML Model (LightGBM) for Credit Scoring.
    2. Rule-Based Logic for Band Assignment & Recommendations.
    3. SHAP Explainer for Prediction Explainability.
    """
    def __init__(self, model_path='battery_credit_model_lightGBM.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Train usage_tips.py first.")
        
        print(f"Loading ML model from {model_path}...")
        self.model = joblib.load(model_path)
        self.explainer = None # Lazy load explainer
        
    def _init_explainer(self):
        if self.explainer is None:
            # TreeExplainer is efficient for LightGBM
            self.explainer = shap.TreeExplainer(self.model)

    def get_usage_guidance(self, credit_score):
        """Internal logic for bands."""
        if credit_score >= 80:
            return {
                "band": "Very Healthy",
                "color": "#4CAF50", "icon": "🟢",
                "message": "This battery shows low degradation and stable behavior.",
                "practices": ["Prefer slow charge/discharge.", "Avoid continuous full-depth cycles.", "Maintain moderate temps."]
            }
        elif credit_score >= 65:
            return {
                "band": "Healthy",
                "color": "#FFC107", "icon": "🟡",
                "message": "Moderate aging; operate under controlled conditions.",
                "practices": ["Limit DoD to ~60–70%.", "Avoid high-current ops.", "Ensure ventilation."]
            }
        elif credit_score >= 45:
            return {
                "band": "Degraded",
                "color": "#FF9800", "icon": "🔵",
                "message": "Noticeable degradation; reduced efficiency.",
                "practices": ["Low-power loads only.", "No fast charging.", "Short discharge durations."]
            }
        else:
            return {
                "band": "Critical",
                "color": "#F44336", "icon": "🔴",
                "message": "Severe degradation; elevated risk.",
                "practices": ["Reuse NOT recommended.", "Recycling advised."]
            }

    def predict(self, input_features):
        """
        Predicts Score and returns full Report (Score, Band, Tips).
        input_features: pandas DataFrame or dict matching model columns.
        """
        # Ensure DataFrame
        if isinstance(input_features, dict):
            input_features = pd.DataFrame([input_features])
        
        # Predict Score
        raw_score = self.model.predict(input_features)[0]
        credit_score = max(0, min(100, raw_score))
        
        # Get Rules
        guidance = self.get_usage_guidance(credit_score)
        
        return {
            "score": credit_score,
            "guidance": guidance
        }

    def explain(self, input_features):
        """
        Returns a list of top feature impacts explaining the score.
        """
        self._init_explainer()
        
        if isinstance(input_features, dict):
            input_features = pd.DataFrame([input_features])
            
        shap_values = self.explainer(input_features)
        
        # Extract impacts
        features = input_features.columns
        impacts = shap_values.values[0]
        
        # Sort by magnitude
        explanations = []
        for feat, impact in sorted(zip(features, impacts), key=lambda x: abs(x[1]), reverse=True):
            if abs(impact) < 0.05: continue # Ignore noise
            direction = "INCREASED" if impact > 0 else "DECREASED"
            val = input_features[feat].values[0]
            explanations.append({
                "feature": feat,
                "value": val,
                "impact": impact,
                "text": f"{feat} ({val:.2f}) {direction} score by {abs(impact):.2f}"
            })
            
        return explanations

def main():
    print("📦 Packaging Battery Advisor (Model + Logic + Explainability)...")
    
    try:
        advisor = BatteryAdvisor()
        
        # Initialize explainer immediately to cache it in the pickle
        advisor._init_explainer()
        
        # Save to pkl
        output_file = 'usage_tips.pkl'
        joblib.dump(advisor, output_file)
        
        print(f"✅ Successfully saved 'BatteryAdvisor' object to {output_file}")
        print(f"   Size: {os.path.getsize(output_file) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ Error packaging model: {e}")

if __name__ == "__main__":
    main()
