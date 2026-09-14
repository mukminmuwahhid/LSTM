"""Central configuration for the electricity-consumption LSTM pipeline."""
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"

ELECTRICITY_RAW_PATH = DATA_DIR / "electricitysupp.csv"
WEATHER_RAW_PATH = DATA_DIR / "weather.xlsx"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_merged.csv"
FEATURED_DATA_PATH = PROCESSED_DIR / "featured.csv"

MODEL_PATH = MODELS_DIR / "lstm_electricity_model.keras"
FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = MODELS_DIR / "target_scaler.joblib"
FEATURE_LIST_PATH = MODELS_DIR / "feature_columns.json"

for _d in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Target / site -------------------------------------------------------
TARGET_COL = "electricity"
SITE_NAME = "bukit mahkota bangi"

# --- Feature engineering --------------------------------------------------
# Raw weather columns worth keeping (snow/snowdepth are constant 0 in this
# tropical climate and are dropped by data_cleaning.load_weather).
WEATHER_FEATURE_COLS = [
    "temp",
    "feelslike",
    "dew",
    "humidity",
    "precip",
    "precipprob",
    "windgust",
    "windspeed",
    "winddir",
    "sealevelpressure",
    "cloudcover",
    "visibility",
    "solarradiation",
    "solarenergy",
    "uvindex",
]

LAG_HOURS = [1, 2, 3, 24, 168]  # 1h, 2h, 3h, 1 day, 1 week
ROLLING_WINDOWS = [3, 24]  # hours, mean + std

# --- Windowing / split -----------------------------------------------------
LOOKBACK = 24  # hours of history fed to the LSTM
HORIZON = 1  # predict 1 step (1 hour) ahead
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # remaining 0.15 goes to test
RANDOM_SEED = 42

# --- Model / training -------------------------------------------------------
LSTM_UNITS = [64, 32]
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 12
REDUCE_LR_PATIENCE = 6
