import pandas as pd
import numpy as np
import joblib
import os
import sys

# Import the guidance logic from the training script
try:
    from usage_tips import get_usage_guidance
except ImportError:
    print("Error: usage_tips.py not found or cannot Import 'get_usage_guidance'.")
    exit()

def validate_model_and_tips():
    print("--- 🧪 Validating Battery Usage Tips System ---")

    # 1. Load the Model
    model_path = 'battery_credit_model_lightGBM.pkl'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run 'usage_tips.py' to train the model first.")
        return
    
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Define Test Cases (Synthetic Data representing different states)
    # Features: ['V_avg', 'V_min', 'V_max', 'I_avg', 'T_avg', 'T_min', 'T_max', 'cycle']
    
    test_cases = [
        {
            "Label": "New/Healthy Battery (Early Cycle)",
            "Features": [3.6, 2.7, 4.2, -1.0, 24.0, 22.0, 26.0, 10] 
        },
        {
            "Label": "Used/Moderate Battery (Mid Cycle)",
            "Features": [3.4, 2.5, 4.1, -1.0, 30.0, 25.0, 35.0, 300]
        },
        {
            "Label": "Degraded Battery (High Cycle/Stress)",
            "Features": [3.2, 2.0, 4.0, -1.0, 38.0, 30.0, 45.0, 600]
        },
        {
            "Label": "Critical/End-of-Life (Very High Cycle)",
            "Features": [3.0, 1.8, 3.8, -1.0, 45.0, 35.0, 55.0, 900]
        }
    ]

    print("\n--- Running Prediction Scenarios ---")
    
    feature_names = ['V_avg', 'V_min', 'V_max', 'I_avg', 'T_avg', 'T_min', 'T_max', 'cycle']

    for case in test_cases:
        print(f"\n🔍 Testing Case: {case['Label']}")
        input_data = np.array(case['Features']).reshape(1, -1)
        input_df = pd.DataFrame(input_data, columns=feature_names)
        
        # Predict Score
        try:
            predicted_score = model.predict(input_df)[0]
            
            # Clamp output to 0-100 logic just in case model extrapolates
            predicted_score = max(0, min(100, predicted_score))
            
            # Get Guidance
            guidance = get_usage_guidance(predicted_score)
            
            # Display Result
            print(f"   📊 Predicted Credit Score: {predicted_score:.1f} / 100")
            print(f"   🏷️  Band: {guidance['color']} {guidance['band']}")
            print(f"   💬 System Message: {guidance['message']}")
            print(f"   💡 Recommendations:")
            for tip in guidance['practices']:
                print(f"      - {tip}")
                
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")

    print("\n--- Validation Complete ---")

if __name__ == "__main__":
    validate_model_and_tips()
