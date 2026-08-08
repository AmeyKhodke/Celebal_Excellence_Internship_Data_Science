import pandas as pd
import numpy as np
from src import config

def add_temporal_features(df):
    """Add calendar-based time features."""
    print("Generating temporal features...")
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    df["month"] = df[config.DATE_COL].dt.month
    df["day_of_week"] = df[config.DATE_COL].dt.dayofweek
    df["day_of_year"] = df[config.DATE_COL].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Simple season encoding (Winter=1, Spring=2, Summer=3, Autumn=4)
    df["season"] = df["month"].apply(lambda m: (m%12 + 3)//3)
    return df

def add_lag_features(df):
    """Generate lag features grouped by household to prevent cross-consumer data leakage."""
    print("Generating lag features per household...")
    # Make sure dataframe is sorted by household and day
    df = df.sort_values(by=[config.HOUSEHOLD_COL, config.DATE_COL]).reset_index(drop=True)
    
    for lag in config.LAG_DAYS:
        df[f"lag_{lag}"] = df.groupby(config.HOUSEHOLD_COL)[config.TARGET_COL].shift(lag)
        
    return df

def add_rolling_features(df):
    """Generate rolling average and standard deviation features per household."""
    print("Generating rolling statistical features...")
    df = df.sort_values(by=[config.HOUSEHOLD_COL, config.DATE_COL]).reset_index(drop=True)
    
    for window in config.ROLLING_WINDOWS:
        grouped = df.groupby(config.HOUSEHOLD_COL)[config.TARGET_COL]
        # Shift first by 1 day to prevent using target_col of today (data leakage) in predicting today's energy
        df[f"rolling_mean_{window}"] = grouped.shift(1).rolling(window=window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = grouped.shift(1).rolling(window=window, min_periods=1).std()
        df[f"rolling_max_{window}"] = grouped.shift(1).rolling(window=window, min_periods=1).max()
        
    # Fill rolling standard deviation NaNs (caused by single initial values in window) with 0
    std_cols = [c for c in df.columns if "rolling_std" in c]
    df[std_cols] = df[std_cols].fillna(0)
    
    return df

def add_weather_interactions(df):
    """Generate weather interaction features."""
    print("Generating weather interaction features...")
    # Calculate temperature difference from a comfortable indoor temp of 18 degrees Celsius
    if "temperatureMax" in df.columns and "temperatureMin" in df.columns:
        df["temp_avg"] = (df["temperatureMax"] + df["temperatureMin"]) / 2
        df["heating_degree_days"] = (18.0 - df["temp_avg"]).clip(lower=0)
        df["cooling_degree_days"] = (df["temp_avg"] - 18.0).clip(lower=0)
    return df

def build_features_pipeline(input_path=None):
    """Load preprocessed dataset and generate all ML feature sets."""
    if input_path is None:
        input_path = config.PROCESSED_DAILY_DATA_PATH
        
    print(f"Loading merged dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Convert day column
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    
    # Run pipelines
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_weather_interactions(df)
    
    # Drop rows that don't have enough lags to train on
    # (Since lag 14 is the maximum lag, drop the first 14 records of each household)
    print("Removing initial rows with incomplete history (due to lag construction)...")
    clean_df = df.dropna(subset=[f"lag_{max(config.LAG_DAYS)}"]).reset_index(drop=True)
    
    print(f"Feature engineering completed. Resulting dataset shape: {clean_df.shape}")
    return clean_df

if __name__ == "__main__":
    features_df = build_features_pipeline()
    # Save a sample or output to processed directory if run directly
    output_path = config.PROCESSED_DATA_DIR / "features_dataset.csv"
    print(f"Saving features dataset to {output_path}...")
    features_df.to_csv(output_path, index=False)
