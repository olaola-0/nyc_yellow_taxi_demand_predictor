import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from src.paths import PARENT_DIR
from src.data import load_raw_data, transform_raw_data_into_time_series_data
from src.feature_store_api import get_or_create_feature_group, FEATURE_GROUP_METADATA

# Load environment variables from .env file in the parent directory
load_dotenv(PARENT_DIR / '.env')

# Constants for Hopsworks project and API key
HOPSWORKS_PROJECT_NAME = 'taxi_demand'
HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']

def get_historical_rides() -> pd.DataFrame:
    """
    Retrieves historical taxi rides data for a specified range of years.

    Downloads raw data for each year starting from a specified start year 
    to the current year and combines them into a single DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing historical rides data.
    """
    from_year = 2023
    to_year = datetime.now().year
    print(f'Downloading raw data from {from_year} to {to_year}')

    rides = pd.DataFrame()
    for year in range(from_year, to_year + 1):
        rides_one_year = load_raw_data(year)
        rides = pd.concat([rides, rides_one_year])

    return rides

def run():
    """
    Main function to backfill the feature group with historical taxi rides data.

    Retrieves historical data, transforms it into time-series format, and 
    inserts it into the feature group in the feature store.

    Steps:
    1. Retrieve historical rides data.
    2. Transform raw data into time-series format.
    3. Insert the transformed data into the feature group.
    """
    rides = get_historical_rides()
    ts_data = transform_raw_data_into_time_series_data(rides)

    # Ensure UTC timezone is set for consistency in time-series data
    ts_data['pickup_hour'] = pd.to_datetime(ts_data['pickup_hour'], utc=True)

    feature_group = get_or_create_feature_group(FEATURE_GROUP_METADATA)

    # Insert data into the feature group without waiting for the job to finish
    feature_group.insert(ts_data, write_options={"wait_for_job": False})

if __name__ == '__main__':
    run()
