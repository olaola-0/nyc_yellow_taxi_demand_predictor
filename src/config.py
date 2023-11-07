import os 
from dotenv import load_dotenv

from src.paths import PARENT_DIR

# Load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / '.env')

# Project name for Hopsworks
HOPSWORKS_PROJECT_NAME = 'NYC_YELLOW_TAXI'
# Retrieve the API key for Hopsworks, raise an exception if it is not found
try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except KeyError:
    raise Exception('Create a .env file in the project root with the HOPSWORKS_API_KEY')

# Feature group name and version
FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1

# Feature view name and version
FEATURE_VIEW_NAME = 'time_series_hourly_feature_view'
FEATURE_VIEW_VERSION = 1

# Model naming
MODEL_NAME = "yellow_taxi_demand_predictor_next_hour"

FEATURE_VIEW_MONITORING = 'predictions_vs_actuals_for_monitoring_feature_view'

# Configuration for the number of historical data points required for prediction
N_FEATURES = 24 * 28  # This equals the number of hours in 28 days

# Configuration for the maximum allowed mean absolute error for the production model
MAX_MAE = 10.0  # Threshold for the MAE metric
