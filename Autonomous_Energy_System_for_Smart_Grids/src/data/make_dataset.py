import os
import pandas as pd
import numpy as np
import glob
from src import config

def clean_smart_meter_data(df):
    """Clean the raw smart meter dataset."""
    print("Cleaning smart meter data...")
    # Convert day to datetime
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    
    # Check for missing values in core target columns
    if df[config.TARGET_COL].isnull().any():
        print(f"Imputing {df[config.TARGET_COL].isnull().sum()} missing values in {config.TARGET_COL}...")
        df[config.TARGET_COL] = df[config.TARGET_COL].fillna(df[config.TARGET_COL].median())
        
    # Standardize columns to numeric where appropriate
    numeric_cols = ["energy_median", "energy_mean", "energy_max", "energy_count", "energy_std", "energy_sum", "energy_min"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Drop records with null energy sum
    df = df.dropna(subset=[config.TARGET_COL])
    
    # Sort values chronologically for each household
    df = df.sort_values(by=[config.HOUSEHOLD_COL, config.DATE_COL]).reset_index(drop=True)
    return df

def load_and_clean_weather():
    """Load and preprocess DarkSky daily weather data."""
    print("Loading daily weather data...")
    weather_df = pd.read_csv(config.WEATHER_DAILY_PATH)
    
    # Convert weather timestamp (e.g. '2011-11-11 00:00:00') to simple date
    weather_df["date_str"] = pd.to_datetime(weather_df["time"]).dt.strftime("%Y-%m-%d")
    weather_df["date_str"] = pd.to_datetime(weather_df["date_str"])
    
    # Pick important weather features
    weather_features = [
        "date_str", "temperatureMax", "temperatureMin", "windSpeed", 
        "humidity", "visibility", "cloudCover", "precipType"
    ]
    # Filter columns that exist
    weather_features = [col for col in weather_features if col in weather_df.columns]
    weather_df = weather_df[weather_features]
    
    # Impute missing cloud cover or numeric features if any
    weather_df["precipType"] = weather_df["precipType"].fillna("none")
    weather_df["precipType_encoded"] = weather_df["precipType"].astype("category").cat.codes
    weather_df = weather_df.drop(columns=["precipType"])
    
    # Fill remaining NaNs with column median
    for col in weather_df.select_dtypes(include=[np.number]).columns:
        weather_df[col] = weather_df[col].fillna(weather_df[col].median())
        
    return weather_df

def load_holidays():
    """Load and preprocess UK bank holidays dataset."""
    print("Loading holidays data...")
    holidays_df = pd.read_csv(config.HOLIDAYS_PATH)
    holidays_df["date_str"] = pd.to_datetime(holidays_df["Bank holidays"])
    holidays_df["is_holiday"] = 1
    return holidays_df[["date_str", "is_holiday"]]

def load_household_info():
    """Load household metadata (Tariff and Acorn categories)."""
    print("Loading household metadata...")
    household_df = pd.read_csv(config.HOUSEHOLD_INFO_PATH)
    # Fill missing values
    household_df["stdorToU"] = household_df["stdorToU"].fillna("Std")
    household_df["Acorn_grouped"] = household_df["Acorn_grouped"].fillna("Comfortable")
    return household_df[[config.HOUSEHOLD_COL, "stdorToU", "Acorn_grouped"]]

def run_preprocessing_pipeline():
    """Run full ETL pipeline to generate consolidated daily dataset."""
    print("Starting ETL preprocessing pipeline...")
    
    # Get the files to process (block_0.csv by default)
    block_files = sorted(glob.glob(os.path.join(config.DAILY_DATA_DIR, "block_*.csv")))
    if not block_files:
        raise FileNotFoundError(f"No block files found in {config.DAILY_DATA_DIR}")
        
    selected_files = block_files[:config.NUM_BLOCKS_TO_PROCESS]
    print(f"Loading and processing {len(selected_files)} block file(s): {selected_files}")
    
    # Load and concat blocks
    dfs = []
    for filepath in selected_files:
        print(f"Reading block file: {filepath}")
        dfs.append(pd.read_csv(filepath))
    raw_df = pd.concat(dfs, ignore_index=True)
    
    # Preprocess smart meter readings
    clean_df = clean_smart_meter_data(raw_df)
    
    # Load auxiliary datasets
    weather_df = load_and_clean_weather()
    holidays_df = load_holidays()
    household_df = load_household_info()
    
    # Merge auxiliary data
    print("Merging smart meter data with auxiliary datasets...")
    # Merge with households info
    merged_df = pd.merge(clean_df, household_df, on=config.HOUSEHOLD_COL, how="left")
    
    # Merge with weather (merge key is day -> date_str)
    merged_df = pd.merge(merged_df, weather_df, left_on=config.DATE_COL, right_on="date_str", how="left")
    merged_df = merged_df.drop(columns=["date_str"])
    
    # Merge with holidays (merge key is day -> date_str)
    merged_df = pd.merge(merged_df, holidays_df, left_on=config.DATE_COL, right_on="date_str", how="left")
    merged_df = merged_df.drop(columns=["date_str"])
    merged_df["is_holiday"] = merged_df["is_holiday"].fillna(0).astype(int)
    
    # Save output
    output_path = config.PROCESSED_DAILY_DATA_PATH
    print(f"Saving merged dataset of shape {merged_df.shape} to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    print("ETL preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
