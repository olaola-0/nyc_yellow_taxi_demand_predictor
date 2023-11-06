import os 
from dotenv import load_dotenv

from src.paths import PARENT_DIR
from src.feature_store_api import FeatureGroupConfig, FeatureViewConfig

#load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR/ '.env')

HOPSWORKS_PROJECT_NAME = 'NYC_YELLOW_TAXI'
try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Create a .env file in the project root with the HOPSWORKS_API_KEY')

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name='time_series_hourly_feature_group',
    version=1,
    description='Feature group with hourly time-series data of historical taxi rides',
    primary_key=['pickup_location_id', 'pickup_hour'],
    event_time='pickup_hour',
)

FEATURE_VIEW_NAME = 'time_series_hourly_feature_view'
FEATURE_VIEW_VERSION = 1
FEATURE_VIEW_METADATA = FeatureViewConfig(
    name='time_series_hourly_feature_view',
    version=1,
    feature_group=FEATURE_GROUP_METADATA,
)

MODEL_NAME = "yellow_taxi_demand_predictor_next_hour"

MAX_MAE = 10.0