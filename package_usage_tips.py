import joblib
import os

class UsagePolicyModel:
    """
    A portable wrapper for the Battery Usage Guidance logic.
    This allows the rule-based logic to be loaded and used just like an ML model.
    """
    def __init__(self):
        self.version = "1.0"
        
    def predict(self, credit_score):
        """
        Input: credit_score (float 0-100)
        Output: Dictionary containing band, message, and recommended practices.
        """
        # Ensure score is scalar if it's an array
        if hasattr(credit_score, 'item'):
            credit_score = credit_score.item()
            
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

def main():
    print("Packaging usage logic into 'usage_tips.pkl'...")
    
    # Instantiate the logic class
    policy_model = UsagePolicyModel()
    
    # Save using joblib
    output_filename = 'usage_tips.pkl'
    joblib.dump(policy_model, output_filename)
    
    print(f"✅ Successfully saved UsagePolicyModel to {output_filename}")
    print("\nExample Usage:")
    print(f"  model = joblib.load('{output_filename}')")
    print("  guidance = model.predict(85.5)")
    print("  print(guidance['band'])")

if __name__ == "__main__":
    main()
