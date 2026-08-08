import os
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src import config

def extract_household_features(df):
    """Aggregate customer logs into household-level behavioral profiles."""
    print("Aggregating daily consumption into household behavior profiles...")
    
    # Calculate peak ratio per row first
    df["row_peak_ratio"] = df["energy_max"] / (df["energy_mean"] + 0.001)
    
    # Group by household
    grouped = df.groupby(config.HOUSEHOLD_COL)
    
    household_features = grouped.agg(
        mean_consumption=(config.TARGET_COL, "mean"),
        std_consumption=(config.TARGET_COL, "std"),
        max_consumption=(config.TARGET_COL, "max"),
        min_consumption=(config.TARGET_COL, "min"),
        peak_ratio=("row_peak_ratio", "mean"),
        stdorToU=("stdorToU", "first"),
        Acorn_grouped=("Acorn_grouped", "first")
    ).reset_index()
    
    # Handle NaNs in std for households with 1 day of data
    household_features["std_consumption"] = household_features["std_consumption"].fillna(0)
    return household_features

def name_clusters(cluster_centers, labels):
    """Assign human-readable, consistent names to clusters based on average consumption."""
    # Find the average mean_consumption for each cluster label
    cluster_means = []
    for i in range(config.N_CLUSTERS):
        cluster_means.append((i, cluster_centers[i, 0])) # Index 0 represents scaled mean_consumption
        
    # Sort cluster indices by their mean consumption
    sorted_clusters = sorted(cluster_means, key=lambda x: x[1])
    
    names_map = {}
    names = [
        "Eco-Saver (Low Usage)",
        "Standard Saver (Moderate Stable)",
        "Comfort Consumer (Moderate Peaky)",
        "High Load (Heavy Consumer)"
    ]
    
    for rank, (original_cluster_idx, _) in enumerate(sorted_clusters):
        names_map[original_cluster_idx] = names[rank]
        
    return names_map

def train_clustering_pipeline():
    """Extract profiles, cluster households, name them, and save the model objects."""
    print("Running customer clustering pipeline...")
    
    # Load processed data
    processed_path = config.PROCESSED_DAILY_DATA_PATH
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed daily data not found at {processed_path}. Run make_dataset first.")
        
    df = pd.read_csv(processed_path)
    
    # Extract customer features
    household_features = extract_household_features(df)
    
    # Select features to cluster
    features = config.CLUSTERING_FEATURES
    X = household_features[features]
    
    print(f"Features used for clustering: {features}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-Means
    print(f"Fitting K-Means model with K={config.N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Assign consistent human-readable cluster names
    cluster_names_map = name_clusters(kmeans.cluster_centers_, cluster_labels)
    
    # Add clusters to dataframe
    household_features["cluster_id"] = cluster_labels
    household_features["usage_pattern"] = household_features["cluster_id"].map(cluster_names_map)
    
    # Print cluster distribution and characteristics
    print("\nCluster Aggregated Profile Summary:")
    summary = household_features.groupby("usage_pattern")[features].mean()
    print(summary.to_string())
    
    # Save Outputs
    scaler_save_path = config.MODELS_DIR / "clustering_scaler.pkl"
    kmeans_save_path = config.MODELS_DIR / "clustering_model.pkl"
    features_save_path = config.CLEANED_HOUSEHOLD_FEATURES_PATH
    
    print(f"Saving scaler to {scaler_save_path}...")
    joblib.dump(scaler, scaler_save_path)
    
    print(f"Saving clustering model to {kmeans_save_path}...")
    joblib.dump(kmeans, kmeans_save_path)
    
    print(f"Saving household cluster profiles to {features_save_path}...")
    household_features.to_csv(features_save_path, index=False)
    
    print("Customer clustering training pipeline completed successfully.")

if __name__ == "__main__":
    train_clustering_pipeline()
