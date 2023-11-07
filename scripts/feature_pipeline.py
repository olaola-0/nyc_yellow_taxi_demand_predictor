"""
This script is responsible for running the feature pipeline for the NYC Yellow Taxi Demand Prediction application. It fetches the raw data from the data warehouse, transforms it into time-series data, and inserts it into the feature group in the feature store.

The script is designed to fetch raw data for the last 28 days to add redundancy to the pipeline. This allows for recovery in case of any failures, enabling the re-processing of data for any missing hours during subsequent runs.

The time-series transformation aggregates the ride data per pickup location and hour. After the transformation, the script obtains a pointer to the feature group and inserts the processed data, ensuring that the job completes before the script exits.

The script accepts a command-line argument --datetime, which specifies the date and time for which the pipeline should run. If this argument is not provided, the current UTC date and time are used.
"""

from datetime import datetime, timedelta
from argparse import ArgumentParser

import pandas as pd

from src import config
from src.data import fetch_ride_events_from_data_warehouse, transform_raw_data_into_time_series_data
from src.feature_store_api import get_or_create_feature_group, FEATURE_GROUP_METADATA
from src.logger import get_logger

# Initialize the logger
logger = get_logger()

def run(date: datetime):
    """
    Executes the feature pipeline process.

    Parameters:
    date (datetime): The date and time for which the feature pipeline should run.
    
    The function performs the following steps:
    1. Fetches raw ride event data from the data warehouse for the last 28 days.
    2. Transforms the raw data into time-series data.
    3. Inserts the transformed data into the feature group in the feature store.
    """

    # Fetch raw ride data for the last 28 days to ensure robustness against failures.
    logger.info('Fetching raw data from data warehouse')
    rides = fetch_ride_events_from_data_warehouse(
        from_date=(date - timedelta(days=28)),
        to_date=date
    )

    # Transform the raw data into time-series format, aggregating by pickup location and hour.
    logger.info('Transforming raw data into time-series data')
    ts_data = transform_raw_data_into_time_series_data(rides=rides)

    # Ensure the 'pickup_hour' column is in the proper datetime format.
    ts_data['pickup_hour'] = pd.to_datetime(ts_data['pickup_hour'], utc=True)

    # Get a pointer to the feature group within the feature store.
    logger.info('Getting pointer to the feature group we want to save data to')
    feature_group = get_or_create_feature_group(feature_group_metadata=FEATURE_GROUP_METADATA)

    # Insert the transformed data into the feature group and wait for the operation to complete.
    logger.info('Start job to insert data into feature group')
    feature_group.insert(ts_data, write_options={"wait_for_job": True})
    logger.info('Finished job to insert data into feature group')


if __name__ == '__main__':
    # Set up argument parsing for the script.
    parser = ArgumentParser()
    parser.add_argument('--datetime',
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
                        help='Datetime for the feature pipeline to run (format: YYYY-MM-DD HH:MM:SS)')
    args = parser.parse_args()

    # Determine the current date based on the provided argument or the system clock.
    if args.datetime:
        current_date = pd.to_datetime(args.datetime)
    else:
        current_date = pd.to_datetime(datetime.utcnow()).floor('H')
    
    # Print out the date for which the feature pipeline is being run.
    print(f'Running feature pipeline for current_date={current_date}')
    
    # Execute the feature pipeline process.
    run(current_date)