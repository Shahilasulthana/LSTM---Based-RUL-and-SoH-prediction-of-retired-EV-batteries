# 🔋 Battery SoH & RUL Prediction Using LSTM

## 📌 Project Overview

Lithium-ion batteries degrade over time due to repeated charge–discharge cycles, temperature variations, and load conditions. Accurate estimation of **State of Health (SoH)** and **Remaining Useful Life (RUL)** is essential for:

- Electric Vehicles (EVs)
- Battery Energy Storage Systems (BESS)
- Battery reuse and second-life decision-making
- Preventive maintenance and safety monitoring

This project implements a **data-driven deep learning (LSTM-based) system** to predict battery **SoH and RUL** using **time-series sensor data** from real-world battery experiments.

---

## 🎯 Problem Statement

Traditional battery health estimation relies on physics-based models that:

- Require deep electrochemical expertise
- Are difficult to generalize
- Do not scale well to large battery fleets

### Objective of this Project

To design an **efficient, scalable, and data-driven system** that predicts:

- Battery **State of Health (SoH)**
- **Remaining Useful Life (RUL)** in months
- Battery **health status**
- **End-of-Life (EOL)** condition

using only **sensor-level time-series data**.

---

## 📊 Dataset Description

The project uses the **NASA Lithium-Ion Battery Degradation Dataset**.

### Dataset Characteristics

- **7,565 individual battery CSV files**
- Each CSV file represents **one battery**
- Each row represents a **time-step measurement**
- Data includes different operating phases:
  - Charging
  - Discharging
  - Rest
  - Load and impedance measurements

### Core Sensor Parameters Used

- `Voltage_measured`
- `Current_measured`
- `Temperature_measured`
- `Current_load`
- `Voltage_load`
- `Time`

> 📌 **Important**  
> SoH and RUL are **not provided** in the raw dataset and are **derived through feature engineering**.

---

## 🧠 Key Concepts

### 🔹 State of Health (SoH)

SoH indicates how healthy a battery is compared to its initial condition.

\[
\text{SoH} = \frac{\text{Estimated Current Capacity}}{\text{Initial Capacity}}
\]

- **SoH = 1.0** → New battery  
- **SoH ≤ 0.7** → End of Life (industry-standard threshold)

---

### 🔹 Remaining Useful Life (RUL)

RUL estimates how much usable life remains before a battery reaches end-of-life.

- Computed in **cycles**
- Converted to **months** assuming:
  - 1 cycle ≈ 1 day
  - 30 cycles ≈ 1 month

---

## 🛠️ Methodology

### 1️⃣ Data Loading

- All individual battery CSV files are loaded separately
- Each battery is processed independently to avoid data leakage

### 2️⃣ Data Cleaning

- Only **discharge-phase records** are retained
- Rows with non-physical relevance for degradation are excluded
- NaNs are handled via **phase-aware filtering**, not naive imputation

### 3️⃣ Feature Engineering

- Estimated capacity using **current–time integration**
- Cycle indexing per battery
- Computation of:
  - SoH
  - RUL (months)
  - Health status (`Healthy`, `Degrading`, `Critical`)
  - End-of-Life flag (`EOL_reached`)

### 4️⃣ Sequence Creation

- Sliding time windows are generated **per battery**
- Enables temporal learning using LSTM
- Preserves degradation continuity

---

## 🤖 Model Architecture

### 🔹 LSTM (Long Short-Term Memory)

LSTM is used because:

- Battery degradation is a **time-dependent process**
- LSTM captures **long-term temporal dependencies**
- It is widely adopted in **Prognostics and Health Management (PHM)**

### Model Design

- **Input:** Time-series sequences (e.g., 20 timesteps × features)
- **Architecture:**
  - LSTM (64 units)
  - Dropout
  - LSTM (32 units)
  - Dense output layer
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Early stopping** to prevent overfitting

Separate models are trained for:
- **SoH prediction**
- **RUL prediction**

---

## 📁 Project Structure

```text
rul_soh_predictor/
│
├── archive/cleaned_dataset/
│   ├── data/                 # Individual battery CSV files
│   ├── extra_infos/          # Additional dataset information
│   └── metadata.csv          # Optional battery metadata
│
├── battery_dataset_with_soh_rul.csv   # Engineered dataset
│
├── battery_lstm_model.keras           # Trained LSTM model
├── scaler.pkl                         # Feature scaling object
│
├── battery_rul_soh_prediction.py      # Model training script
├── predict_manual.py                  # Manual input prediction
├── test_healthy_prediction.py         # Test predictions on healthy batteries
├── validate_dataset.py                # Dataset validation & sanity checks
│
├── app.py                             # Application entry point (future deployment)
├── .gitignore
└── README.md

```
<img width="1600" height="809" alt="image" src="https://github.com/user-attachments/assets/e9f9717e-12f0-4c8b-a65d-8876af5c42f4" />

---

<img width="1600" height="810" alt="image" src="https://github.com/user-attachments/assets/534fa632-5f6a-4b88-8df1-2661e1360015" />

---

<img width="1600" height="802" alt="image" src="https://github.com/user-attachments/assets/4a85a78c-1455-41a3-a4af-40148859baa9" />
