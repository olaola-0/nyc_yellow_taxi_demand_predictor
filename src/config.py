import os 
from dotenv import load_dotenv

from src.paths import PARENT_DIR

#load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR/ '.env')

HOPSWORKS_PROJECT_NAME = 'NYC_YELLOW_TAXI'
try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Create a .env file in the project root with the HOPSWORKS_API_KEY')

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1