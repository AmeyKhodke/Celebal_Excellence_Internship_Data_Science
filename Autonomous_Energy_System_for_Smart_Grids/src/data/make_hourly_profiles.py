import os
import pandas as pd
import numpy as np
from src import config

def generate_hourly_profiles():
    """Aggregate half-hourly data into typical daily usage profiles per household."""
    print("Starting half-hourly load profile aggregation pipeline...")
    
    # Define outputs path
    output_path = config.PROCESSED_DATA_DIR / "household_hourly_profiles.csv"
    
    # 1. Identify which households were processed in daily dataset
    # We load them from household_features.csv if it exists (clustering output)
    # or look in processed_daily_data.csv, or read from raw daily block_0.csv
    households = None
    if os.path.exists(config.CLEANED_HOUSEHOLD_FEATURES_PATH):
        print(f"Loading household list from {config.CLEANED_HOUSEHOLD_FEATURES_PATH}...")
        df_households = pd.read_csv(config.CLEANED_HOUSEHOLD_FEATURES_PATH)
        households = set(df_households[config.HOUSEHOLD_COL].unique())
    elif os.path.exists(config.PROCESSED_DAILY_DATA_PATH):
        print(f"Loading household list from {config.PROCESSED_DAILY_DATA_PATH}...")
        df_daily = pd.read_csv(config.PROCESSED_DAILY_DATA_PATH)
        households = set(df_daily[config.HOUSEHOLD_COL].unique())
    else:
        # Fallback to daily block_0.csv
        daily_block_0 = os.path.join(config.DAILY_DATA_DIR, "block_0.csv")
        if os.path.exists(daily_block_0):
            print(f"Loading household list from raw daily {daily_block_0}...")
            df_daily = pd.read_csv(daily_block_0)
            households = set(df_daily[config.HOUSEHOLD_COL].unique())
            
    if not households:
        raise FileNotFoundError("Could not find any processed or raw daily smart meter data. Run make_dataset first.")
        
    print(f"Targeting {len(households)} households for hourly profile extraction.")
    
    # 2. Load half-hourly block_0.csv
    hh_block_0_path = os.path.join(config.DATA_DIR, "halfhourly_dataset", "halfhourly_dataset", "block_0.csv")
    if not os.path.exists(hh_block_0_path):
        raise FileNotFoundError(f"Half-hourly block_0 not found at {hh_block_0_path}")
        
    print(f"Reading half-hourly smart meter records from {hh_block_0_path}...")
    # Read in chunks or normally since E: now has space
    # Let's read the csv
    df_hh = pd.read_csv(hh_block_0_path)
    print(f"Loaded {len(df_hh)} rows of half-hourly readings. Filtering for target households...")
    
    # Filter for target households
    df_hh = df_hh[df_hh[config.HOUSEHOLD_COL].isin(households)].copy()
    print(f"Filtered dataset contains {len(df_hh)} rows.")
    
    # Clean energy column
    energy_col = "energy(kWh/hh)"
    if energy_col not in df_hh.columns:
        # Check if column name has spaces or differences
        actual_col = [col for col in df_hh.columns if "energy" in col]
        if actual_col:
            energy_col = actual_col[0]
            
    df_hh[energy_col] = pd.to_numeric(df_hh[energy_col], errors="coerce")
    df_hh = df_hh.dropna(subset=[energy_col])
    
    # Convert tstp to datetime and extract half-hour index (0 to 47)
    print("Processing timestamps and grouping...")
    df_hh["tstp"] = pd.to_datetime(df_hh["tstp"])
    df_hh["half_hour_index"] = df_hh["tstp"].dt.hour * 2 + df_hh["tstp"].dt.minute // 30
    
    # 3. Group and compute typical consumption profile per household per half-hour
    grouped = df_hh.groupby([config.HOUSEHOLD_COL, "half_hour_index"])[energy_col].mean().reset_index()
    
    # Pivot to wide format: LCLid, 0, 1, 2, ..., 47
    profiles = grouped.pivot(index=config.HOUSEHOLD_COL, columns="half_hour_index", values=energy_col)
    
    # Handle missing intervals if any
    if profiles.isnull().any().any():
        profiles = profiles.interpolate(axis=1, limit_direction="both").fillna(0.0)
        
    # Save the output
    print(f"Saving {len(profiles)} household hourly load profiles to {output_path}...")
    profiles.to_csv(output_path)
    print("Half-hourly load profile aggregation completed successfully.")

if __name__ == "__main__":
    generate_hourly_profiles()
