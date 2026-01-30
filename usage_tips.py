import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def calculate_accuracy(y_true, y_pred, tolerance=2.5):
    """
    Calculate accuracy based on a tolerance threshold.
    Accuracy = Percentage of predictions within ±tolerance of the true value.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)
    
    # Check which predictions are within the tolerance
    within_tolerance = errors <= tolerance
    
    # Calculate accuracy percentage
    accuracy = np.mean(within_tolerance) * 100
    return accuracy

def get_usage_guidance(credit_score):
    """
    Returns the usage guidance and health state based on the Battery Credit Score.
    """
    if credit_score >= 80:
        band = "Very Healthy"
        system_msg = "This battery currently shows low degradation and stable behavior."
        practices = [
            "Prefer slow charge/discharge",
            "Avoid continuous full-depth cycles",
            "Maintain moderate operating temperatures"
        ]
        color_code = "🟢"
    elif credit_score >= 65:
        band = "Healthy"
        system_msg = "This battery shows moderate aging and should be operated under controlled conditions."
        practices = [
            "Limit depth of discharge to ~60–70%",
            "Avoid frequent high-current operation",
            "Ensure adequate ventilation"
        ]
        color_code = "🟡"
    elif credit_score >= 45:
        band = "Degraded"
        system_msg = "This battery shows noticeable degradation and reduced efficiency."
        practices = [
            "Use for low-power or intermittent loads",
            "Avoid fast charging",
            "Keep discharge durations short"
        ]
        color_code = "🔵"
    else:
        band = "Critical"
        system_msg = "This battery exhibits severe degradation and elevated risk."
        practices = [
            "Continued reuse may accelerate failure",
            "Professional evaluation or recycling is advised"
        ]
        color_code = "🔴"
        
    return {
        "score": credit_score,
        "band": band,
        "color": color_code,
        "message": system_msg,
        "practices": practices
    }

def main():
    print("--- Battery Credit Score & Usage Recommendation Model Training ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv('battery_dataset_with_soh_rul_bands.csv')
        print(f"Dataset loaded: {len(df)} records")
    except FileNotFoundError:
        print("Error: Dataset not found. Please run 'battery_rul_soh_prediction.py' first.")
        return

    # 2. Feature Engineering: Create Battery_Credit_Score
    # We derive it from SoH (0-1) -> Score (0-100)
    # We can add a bit of noise or complex logic if needed, but for now SoH is the direct indicator.
    if 'SoH' in df.columns:
        df['Battery_Credit_Score'] = df['SoH'] * 100
    else:
        print("Error: 'SoH' column missing.")
        return

    # Select Features
    # Numeric features relevant to health
    features = ['V_avg', 'V_min', 'V_max', 'I_avg', 'T_avg', 'T_min', 'T_max', 'cycle']
    target = 'Battery_Credit_Score'
    
    # Drop rows with missing values
    df_aug = df[features + [target]].dropna()
    
    X = df_aug[features]
    y = df_aug[target]

    print(f"Training on {len(X)} samples.")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Define Models
    models = {
        "RandomForest": GridSearchCV(RandomForestRegressor(random_state=42), {'n_estimators': [50, 100], 'max_depth': [5, 10]}, cv=3),
        "XGBoost": GridSearchCV(XGBRegressor(random_state=42), {'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}, cv=3),
        "LightGBM": GridSearchCV(LGBMRegressor(verbose=-1, random_state=42), {'learning_rate': [0.05, 0.1], 'min_child_samples': [10, 20]}, cv=3)
        # Reduced Grid Search params slightly for speed, as user snippet was just example
    }

    print("\nTraining optimized models...")
    best_overall_model = None
    best_rmse = float('inf')
    
    for name, grid in models.items():
        print(f"Training {name}...")
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        acc = calculate_accuracy(y_test, y_pred, tolerance=2.5)
        
        print(f"  -> Best Params: {grid.best_params_}")
        print(f"  -> RMSE: {rmse:.4f}, Accuracy (within ±2.5): {acc:.2f}%")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_overall_model = best_model

        # Save LightGBM as 'ultimate' if specified, or just save the best one
        if name == "LightGBM":
            joblib.dump(best_model, 'battery_credit_model_lightGBM.pkl')
            print("  -> LightGBM Model Saved as 'battery_credit_model_lightGBM.pkl'")

    print("\n--- Model Training Complete ---")
    
    # 5. Demonstration of Usage Tips
    print("\n🧠 Example Usage Recommendation System")
    
    # Take a few samples from test set
    sample_indices = X_test.sample(5).index
    
    for idx in sample_indices:
        sample_input = X.loc[idx]
        true_score = y.loc[idx]
        
        # Predict using the LightGBM model (or best)
        # We assume the saved LightGBM is the one we want to use for the "Ultimate" experience
        try:
            loaded_model = joblib.load('battery_credit_model_lightGBM.pkl')
            model_to_use = loaded_model
        except:
            model_to_use = best_overall_model

        pred_score = model_to_use.predict([sample_input])[0]
        guidance = get_usage_guidance(pred_score)
        
        print(f"\n{guidance['color']} Battery (True: {true_score:.1f}, Pred: {pred_score:.1f}) | {guidance['band']}")
        print(f"   System Message: {guidance['message']}")
        print(f"   Recommended practices:")
        for practice in guidance['practices']:
            print(f"     - {practice}")

    # ---------------------------------------------------------
    # 6. EXPLAINABILITY SECTION (SHAP)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🧠 EXPLAINABILITY ANALYSIS (Why did the model predict this?)")
    print("="*50)
    
    # Use the best model (Likely LightGBM)
    model_for_shap = best_overall_model
    
    # 1. Global Feature Importance (Avg impact)
    try:
        if hasattr(model_for_shap, 'feature_importances_'):
            print("\n1. Generating Global Feature Importance Plot...")
            feature_importances = pd.Series(model_for_shap.feature_importances_, index=X.columns).sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_importances, y=feature_importances.index, hue=feature_importances.index, palette='viridis', legend=False)
            plt.title("Global Feature Importance (What matters most?)")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            plt.show() # In a script this might just pop up a window or do nothing if headless.
            # We also print it for CLI visibility
            print("\nTop Features driving the Credit Score:")
            print(feature_importances.head(5))
    except Exception as e:
        print(f"Skipping feature importance plot: {e}")

    # 2. Local Explanation (Waterfall Plot for a single sample)
    print("\n2. analyzing a specific battery case...")
    try:
        # Create a sample that matches OUR features
        # Features: ['V_avg', 'V_min', 'V_max', 'I_avg', 'T_avg', 'T_min', 'T_max', 'cycle']
        sample_data = pd.DataFrame([{
            'V_avg': 3.4, 'V_min': 2.5, 'V_max': 4.1, 
            'I_avg': -1.5, 
            'T_avg': 34.0, 'T_min': 25.0, 'T_max': 45.0, # Slightly hot
            'cycle': 500  # Mid-life
        }])
        
        # Predict
        score = model_for_shap.predict(sample_data)[0]
        print(f"\n--- Prediction Result for Sample Case ---")
        print(f"Input: Mid-life battery (500 cycles, Max Temp 45°C)")
        print(f"Predicted Credit Score: {score:.2f} / 100")
        
        guidance = get_usage_guidance(score)
        print(f"Band: {guidance['band']}")
        print(f"Recommendation: {guidance['message']}")

        # SHAP Values
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer(sample_data)
        
        # Textual Explanation
        base_value = shap_values.base_values[0]
        print(f"\n--- Textual Explanation of the Score ---")
        print(f"Base Score (Average Battery): {base_value:.2f}")
        print(f"Final Score: {score:.2f}")
        print("Why?")
        
        # Extract impacts
        features_names = sample_data.columns
        impacts = shap_values.values[0]
        feature_impacts = sorted(zip(features_names, impacts), key=lambda x: abs(x[1]), reverse=True)
        
        for feat, impact in feature_impacts:
            if abs(impact) < 0.05: continue # Ignore negligible
            direction = "INCREASED" if impact > 0 else "DECREASED"
            val = sample_data[feat].values[0]
            print(f"  - {feat} (Value: {val}) {direction} the score by {abs(impact):.2f} points")
            
        print("\n(Note: High Cycle count and Temperature typically decrease the score)")

    except Exception as e:
        print(f"Explainability error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
