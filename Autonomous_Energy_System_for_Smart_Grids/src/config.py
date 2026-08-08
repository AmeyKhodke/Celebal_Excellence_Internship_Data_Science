import os
from pathlib import Path

# Paths
BASE_DIR = Path("e:/B TECH IT/Celebal Internship/Autonomous_Energy_System_for_Smart_Grids")
DATA_DIR = BASE_DIR / "Dataset"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# Sub-datasets
DAILY_DATA_DIR = DATA_DIR / "daily_dataset" / "daily_dataset"
HOUSEHOLD_INFO_PATH = DATA_DIR / "informations_households.csv"
WEATHER_DAILY_PATH = DATA_DIR / "weather_daily_darksky.csv"
WEATHER_HOURLY_PATH = DATA_DIR / "weather_hourly_darksky.csv"
HOLIDAYS_PATH = DATA_DIR / "uk_bank_holidays.csv"

# Preprocessed Outputs
PROCESSED_DATA_DIR = DATA_DIR / "processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PROCESSED_DAILY_DATA_PATH = PROCESSED_DATA_DIR / "processed_daily_data.csv"
CLEANED_HOUSEHOLD_FEATURES_PATH = PROCESSED_DATA_DIR / "household_features.csv"

# Data selection configurations (limits memory usage)
NUM_BLOCKS_TO_PROCESS = 1  # Process block_0.csv by default for fast verification

# Model targets & features
TARGET_COL = "energy_sum"
DATE_COL = "day"
HOUSEHOLD_COL = "LCLid"

# Forecasting Config
FORECAST_HORIZON = 7  # Predict next 7 days of consumption
LAG_DAYS = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [3, 7, 14]

# Clustering Config
N_CLUSTERS = 4
CLUSTERING_FEATURES = ["mean_consumption", "std_consumption", "max_consumption", "min_consumption", "peak_ratio"]

# Tariff Configurations (UK pence/kWh or currency units/kWh)
# Time-of-Use (ToU) vs Standard tariff rates
STANDARD_RATE = 0.1422  # Flat rate (£ per kWh)

# ToU rates based on London Hydro dynamic tariff experiments
TOU_PEAK_RATE = 0.2486       # Peak rate (£ per kWh) between 16:00 and 20:00
TOU_OFFPEAK_RATE = 0.0500   # Off-peak rate (£ per kWh) between 00:00 and 07:00
TOU_STANDARD_RATE = 0.1180  # Standard rate (£ per kWh) for other hours
