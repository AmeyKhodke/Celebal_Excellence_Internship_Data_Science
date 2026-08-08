import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src import config
from src.data.feature_engineering import build_features_pipeline

def prepare_data_for_modeling(df):
    """Clean data types, encode categorical variables, and define features."""
    print("Preparing data and encoding categoricals...")
    
    # Categorical variables encoding
    # stdorToU: 'Std' vs 'ToU'
    df["stdorToU_encoded"] = (df["stdorToU"] == "ToU").astype(int)
    
    # Acorn_grouped: Map categories to integer codes
    acorn_cats = sorted(df["Acorn_grouped"].astype(str).unique().tolist())
    acorn_map = {cat: idx for idx, cat in enumerate(acorn_cats)}
    df["acorn_encoded"] = df["Acorn_grouped"].astype(str).map(acorn_map)
    
    # Save the categories mapping for inference/dashboard
    encoding_info = {
        "acorn_map": acorn_map,
        "acorn_categories": acorn_cats
    }
    
    # Select feature columns
    feature_cols = [
        "month", "day_of_week", "day_of_year", "is_weekend", "season", "is_holiday",
        "temperatureMax", "temperatureMin", "windSpeed", "humidity", "visibility", "cloudCover",
        "heating_degree_days", "cooling_degree_days",
        "stdorToU_encoded", "acorn_encoded"
    ]
    
    # Add lag and rolling feature names dynamically
    lag_cols = [f"lag_{lag}" for lag in config.LAG_DAYS]
    rolling_cols = []
    for w in config.ROLLING_WINDOWS:
        rolling_cols.extend([f"rolling_mean_{w}", f"rolling_std_{w}", f"rolling_max_{w}"])
        
    feature_cols.extend(lag_cols + rolling_cols)
    
    # Filter features that actually exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    return df, feature_cols, encoding_info

def train_forecaster_pipeline():
    """Train consumption forecaster and save the model."""
    print("Running forecaster training pipeline...")
    
    # Generate engineered features
    df = build_features_pipeline()
    
    df, feature_cols, encoding_info = prepare_data_for_modeling(df)
    
    # Chronological Split
    # Split training on dates: train on older data, test on the latest dates
    unique_dates = sorted(df[config.DATE_COL].unique())
    split_idx = int(len(unique_dates) * 0.8)  # 80/20 split
    split_date = unique_dates[split_idx]
    
    train_mask = df[config.DATE_COL] < split_date
    test_mask = df[config.DATE_COL] >= split_date
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    X_train = train_df[feature_cols]
    y_train = train_df[config.TARGET_COL]
    
    X_test = test_df[feature_cols]
    y_test = test_df[config.TARGET_COL]
    
    print(f"Train samples: {X_train.shape[0]} (before {split_date.date()})")
    print(f"Test samples: {X_test.shape[0]} (on/after {split_date.date()})")
    print(f"Features used: {feature_cols}")
    
    # Training Model (LightGBM with fallback to scikit-learn Random Forest)
    try:
        import lightgbm as lgb
        print("Training LightGBM regressor model...")
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        model_type = "lightgbm"
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        print("LightGBM not installed. Falling back to RandomForestRegressor...")
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        model_type = "random_forest"
        
    # Predictions & Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    
    # Calculate Mean Absolute Percentage Error (MAPE), handling division by zero/near-zero
    y_test_non_zero = y_test.copy()
    y_test_non_zero[y_test_non_zero == 0] = 0.01
    mape = np.mean(np.abs((y_test_non_zero - preds) / y_test_non_zero)) * 100
    
    metrics = {
        "model_type": model_type,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }
    
    print(f"\nForecasting Performance metrics:")
    print(f"  Model Type: {model_type}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} kWh")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} kWh")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Feature Importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10).to_string(index=False))
    
    # Save Model Artifacts
    model_save_path = config.MODELS_DIR / "consumption_forecaster.pkl"
    meta_save_path = config.MODELS_DIR / "model_metadata.json"
    
    print(f"Saving forecasting model to {model_save_path}...")
    joblib.dump(model, model_save_path)
    
    # Metadata save
    metadata = {
        "metrics": metrics,
        "feature_cols": feature_cols,
        "encoding_info": encoding_info
    }
    with open(meta_save_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Forecasting model training completed successfully.")

if __name__ == "__main__":
    train_forecaster_pipeline()
