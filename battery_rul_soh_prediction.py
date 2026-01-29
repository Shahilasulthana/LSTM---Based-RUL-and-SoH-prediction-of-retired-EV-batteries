
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D, Flatten, Bidirectional, GlobalAveragePooling1D, Concatenate, Attention, Reshape

# 1. Environment Setup & Data Loading
# Dataset is locally available
BASE_PATH = os.path.join(os.getcwd(), 'archive (9)', 'cleaned_dataset')

METADATA_PATH = os.path.join(BASE_PATH, 'metadata.csv')
DATA_DIR = os.path.join(BASE_PATH, 'data')

print(f"Metadata Path: {METADATA_PATH}")
print(f"Data Directory: {DATA_DIR}")

if not os.path.exists(METADATA_PATH):
    print(f"Warning: Metadata file not found at {METADATA_PATH}. Checking current directory...")
    # Fallback to check relative to script location if getcwd() is different
    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_PATH = os.path.join(script_dir, 'archive (9)', 'cleaned_dataset')
    METADATA_PATH = os.path.join(BASE_PATH, 'metadata.csv')
    DATA_DIR = os.path.join(BASE_PATH, 'data')
    print(f"New Metadata Path: {METADATA_PATH}")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

# Load Metadata
metadata = pd.read_csv(METADATA_PATH)

# Ensure Capacity is numeric
metadata['Capacity'] = pd.to_numeric(metadata['Capacity'], errors='coerce')

print("Columns:", metadata.columns)
print("Unique Batteries:", metadata['battery_id'].unique())

# 2. Feature Extraction
def extract_cycle_features(meta_row, data_dir):
    """
    Reads the CSV file for a specific cycle and extracts summary features.
    """
    filename = meta_row['filename']
    file_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Features to extract (simple statistics over the cycle time-series)
        features = {
            'battery_id': meta_row['battery_id'],
            'cycle': meta_row['test_id'],  # Using test_id as cycle count
            'type': meta_row['type'],
            'time_start': meta_row['start_time'],
            
            # Voltage features
            'V_avg': df['Voltage_measured'].mean(),
            'V_min': df['Voltage_measured'].min(),
            'V_max': df['Voltage_measured'].max(),
            
            # Current features
            'I_avg': df['Current_measured'].mean(),
            'I_min': df['Current_measured'].min(),
            'I_max': df['Current_measured'].max(),
            
            # Temperature features
            'T_avg': df['Temperature_measured'].mean(),
            'T_min': df['Temperature_measured'].min(),
            'T_max': df['Temperature_measured'].max(),
            
            # Capacity (Ground Truth from metadata given by dataset)
            'Capacity': meta_row['Capacity'] if 'Capacity' in meta_row else np.nan
        }
        return features
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

# Filter only 'discharge' cycles as they contain the capacity degradation info
discharge_meta = metadata[metadata['type'] == 'discharge'].copy()

# Run extraction
print("Extracting features from CSV files... This may take a moment.")
feature_list = []
for index, row in discharge_meta.iterrows():
    feats = extract_cycle_features(row, DATA_DIR)
    if feats:
        feature_list.append(feats)

dataset_df = pd.DataFrame(feature_list)
print(f"Extracted {len(dataset_df)} records.")

# Drop records with missing values (e.g. empty CSVs or missing Capacity)
dataset_df.dropna(inplace=True)
dataset_df = dataset_df[dataset_df['Capacity'] > 0] # Filter out 0 capacity
print(f"Records after dropping NaNs and 0 Capacity: {len(dataset_df)}")

# 3. Calculate SoH and RUL
# Compute SoH and RUL per Battery ID
dataset_df = dataset_df.sort_values(by=['battery_id', 'cycle'])

def calculate_soh_rul(group):
    # 1. Nominal Capacity (Max capacity in the first few cycles or absolute max)
    # We iterate to find the max capacity observed to be robust
    nominal_capacity = group['Capacity'].max()
    

    if nominal_capacity == 0 or np.isnan(nominal_capacity):
        group['SoH'] = np.nan
        group['RUL'] = np.nan
        return group
    
    # Calculate SoH
    group['SoH'] = group['Capacity'] / nominal_capacity
    
    # Calculate RUL
    # Assume the last cycle in the dataset is the failure point
    total_cycles = group['cycle'].max()
    group['RUL'] = total_cycles - group['cycle']
    
    return group

dataset_df = dataset_df.groupby('battery_id').apply(calculate_soh_rul).reset_index(drop=True)
dataset_df.dropna(subset=['SoH', 'RUL'], inplace=True)

# Save the new comprehensive dataset
dataset_df.to_csv("battery_dataset_with_soh_rul.csv", index=False)
print("New dataset generated and saved: battery_dataset_with_soh_rul.csv")

# 4. Prepare Data for Deep LSTM Model
# Use sliding window sequences for better temporal accuracy
features = ['V_avg', 'V_min', 'V_max', 'I_avg', 'T_avg', 'T_min', 'T_max']
target_rul = 'RUL'
target_soh = 'SoH'

# Normalize Features
scaler = MinMaxScaler()
dataset_df[features] = scaler.fit_transform(dataset_df[features])

def create_sequences(data, seq_length, feature_cols, target_cols):
    xs = []
    ys_rul = []
    ys_soh = []
    
    for bid in data['battery_id'].unique():
        group = data[data['battery_id'] == bid]
        
        if len(group) <= seq_length:
            continue
            
        group_features = group[feature_cols].values
        group_rul = group['RUL'].values
        group_soh = group['SoH'].values
        
        for i in range(len(group) - seq_length):
            x = group_features[i:(i + seq_length)]
            # Predict only for the last step in the sequence
            xs.append(x)
            ys_rul.append(group_rul[i + seq_length])
            ys_soh.append(group_soh[i + seq_length])
            
    return np.array(xs), np.array(ys_rul), np.array(ys_soh)

# Increased sequence length for better history context
SEQ_LENGTH = 20 

X, y_rul, y_soh = create_sequences(dataset_df, SEQ_LENGTH, features, [target_rul, target_soh])

print(f"Input Shape: {X.shape}")
print(f"Target RUL Shape: {y_rul.shape}")

# Split Data
X_train, X_test, y_rul_train, y_rul_test, y_soh_train, y_soh_test = train_test_split(
    X, y_rul, y_soh, test_size=0.2, random_state=42
)

print(f"Train Shape: {X_train.shape}")
print(f"Test Shape: {X_test.shape}")

# 5. Build Model (Bidirectional LSTM with Attention)
input_layer = Input(shape=(SEQ_LENGTH, len(features)))

# 1. Feature Extraction (1D CNN)
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

# 2. Bidirectional LSTM (Deep Temporal Context)
# Return sequences=True to allow Attention to work on all time steps
x = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'))(x)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(x)
x = Dropout(0.3)(x)

# 3. Attention Mechanism
# Query, Value, and Key are all 'x' (Self-Attention)
attention = Attention()([x, x])
# Combine attention output with original features
x = Concatenate()([x, attention])

# 4. Global Pooling to flatten for Dense layers
x = GlobalAveragePooling1D()(x)

# 5. Split Branches for Specialized Learning

# --- RUL Branch ---
rul_x = Dense(64, activation='relu')(x)
rul_x = BatchNormalization()(rul_x)
rul_x = Dropout(0.2)(rul_x)
rul_x = Dense(32, activation='relu')(rul_x)
rul_output = Dense(1, name='rul_output')(rul_x)

# --- SoH Branch (High Accuracy Focus) ---
soh_x = Dense(64, activation='relu')(x)
soh_x = BatchNormalization()(soh_x)
soh_x = Dropout(0.2)(soh_x)
soh_x = Dense(32, activation='relu')(soh_x)
soh_output = Dense(1, activation='sigmoid', name='soh_output')(soh_x)

model = Model(inputs=input_layer, outputs=[rul_output, soh_output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Very fine Learning Rate
    loss={'rul_output': 'mse', 'soh_output': 'mse'},
    metrics={'rul_output': 'mae', 'soh_output': 'mae'}
)

model.summary()

# Callbacks for better training
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 6. Train Model
history = model.fit(
    X_train, {'rul_output': y_rul_train, 'soh_output': y_soh_train},
    validation_data=(X_test, {'rul_output': y_rul_test, 'soh_output': y_soh_test}),
    epochs=200, 
    batch_size=64, # Increased batch size for smoother gradients
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

print("Training Complete.")

# 7. Evaluation
print("\n--- Model Evaluation ---")
results = model.evaluate(X_test, {'rul_output': y_rul_test, 'soh_output': y_soh_test}, verbose=0)

# Predictions for R2 Score
pred = model.predict(X_test)
pred_rul = pred[0].flatten()
pred_soh = pred[1].flatten()

# Calculate Metrics
rul_mae = mean_absolute_error(y_rul_test, pred_rul)
rul_rmse = np.sqrt(mean_squared_error(y_rul_test, pred_rul))
rul_r2 = r2_score(y_rul_test, pred_rul)

soh_mae = mean_absolute_error(y_soh_test, pred_soh)
soh_rmse = np.sqrt(mean_squared_error(y_soh_test, pred_soh))
soh_r2 = r2_score(y_soh_test, pred_soh)

print(f"\nRemaining Useful Life (RUL) Metrics:")
print(f"  - MAE (Average Error): {rul_mae:.2f} cycles")
print(f"  - RMSE (Root Mean Sq Error): {rul_rmse:.2f} cycles")
print(f"  - R2 Score (Accuracy): {rul_r2*100:.2f}%")

print(f"\nState of Health (SoH) Metrics:")
print(f"  - MAE (Average Error): {soh_mae:.4f}")
print(f"  - RMSE (Root Mean Sq Error): {soh_rmse:.4f}")
print(f"  - R2 Score (Accuracy): {soh_r2*100:.2f}%")

# Save Model and Scaler
model.save('battery_lstm_model.keras')
print("\nModel saved as battery_lstm_model.keras")

joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")
