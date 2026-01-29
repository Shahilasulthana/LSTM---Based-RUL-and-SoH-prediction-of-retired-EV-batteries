
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
try:
    df = pd.read_csv('battery_dataset_with_soh_rul.csv')
    print("Dataset Loaded Successfully.")
    print(f"Total Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nSample Data (First 5 rows):")
    print(df[['battery_id', 'cycle', 'Capacity', 'SoH', 'RUL']].head().to_string(index=False))

    print("\nSample Data (Last 5 rows):")
    print(df[['battery_id', 'cycle', 'Capacity', 'SoH', 'RUL']].tail().to_string(index=False))

    # Check for NaNs
    print(f"\nNaN values in SoH: {df['SoH'].isna().sum()}")
    print(f"NaN values in RUL: {df['RUL'].isna().sum()}")

    # Check stats per battery
    print("\nSummary per Battery:")
    for bid in df['battery_id'].unique()[:5]: # Show first 5 batteries
        subset = df[df['battery_id'] == bid]
        print(f"Battery {bid}: Cycles={len(subset)}, Max SoH={subset['SoH'].max():.4f}, Min SoH={subset['SoH'].min():.4f}, Max RUL={subset['RUL'].max()}")

except FileNotFoundError:
    print("Error: battery_dataset_with_soh_rul.csv not found. Please run the generation script first.")
