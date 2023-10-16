from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR


def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """
    Download a specific month's raw ride data from a given URL and save it as a .parquet file.
    Parameters:
    - year (int): The year of the desired data.
    - month (int): The month of the desired data.
    Returns:
    - Path: The path to the downloaded .parquet file.
    Raises:
    - Exception: If the URL does not return a 200 status code.
    """
    # Construct the URL using the given year and month
    URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"

    # Send a GET request to the URL
    response = requests.get(URL)

    # If the response is successful (status code 200)
    if response.status_code == 200:
        # Define the path where the .parquet file will be saved
        path = RAW_DATA_DIR /f"rides_{year}-{month:02d}.parquet"
        
        # Write the content of the response to the .parquet file
        open(path, "wb").write(response.content)
        
        return path
    else:
        # Raise an exception if the URL is not available
        raise Exception(f"{URL} is not available")


def validate_raw_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Validate and filter the raw data based on the given year and month.
    
    Parameters:
    - rides (pd.DataFrame): DataFrame containing the raw ride data.
    - year (int): The year to be used for filtering.
    - month (int): The month to be used for filtering.
    
    Returns:
    - pd.DataFrame: DataFrame containing the validated and filtered ride data.
    """
    this_month_start = f"{year}-{month:02d}-01"
    next_month_start = f"{year}-{month:02d}-01" if month < 12 else f"{year + 1}-01-01"
    
    # Filtering the data based on the specified year and month
    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]

    return rides


def load_raw_data(year: int, months: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load raw data either from local storage or by downloading it, 
    and then validate and filter the data.
    
    Parameters:
    - year (int): The year of the desired data.
    - months (Optional[List[int]]): List of months of the desired data.
    
    Returns:
    - pd.DataFrame: DataFrame containing the loaded, validated, and filtered ride data.
    """
    rides = pd.DataFrame()

    if months is None:
        months = list(range(1, 13))
    elif isinstance(months, int):
        months = [months]

    # Looping through each specified month
    for month in months:
        local_file = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        
        # Check if the file already exists locally
        if not local_file.exists():
            try:
                # Try downloading the file if it doesn't exist locally
                print(f"Downloading file {year}-{month:02d}")
                download_one_file_of_raw_data(year, month)
            except:
                print(f"{year}-{month:02d} file is not available.")
                continue
        else:
            print(f"File {year}-{month:02d} already in local storage")

        # Load, validate, and concatenate the data
        rides_one_month = pd.read_parquet(local_file)
        rides_one_month = rides_one_month[["tpep_pickup_datetime", "PULocationID"]]
        rides_one_month.rename(columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "PULocationID": "pickup_location_id",
        }, inplace=True)
        
        rides_one_month = validate_raw_data(rides_one_month, year, month)
        rides = pd.concat([rides, rides_one_month])

    rides = rides[["pickup_datetime", "pickup_location_id"]]

    return rides


def add_missing_slots(agg_rides: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in missing hourly slots for each pickup location with zero rides.
    Parameters:
    - agg_rides (pd.DataFrame): DataFrame with aggregated ride counts.
    Returns:
    - pd.DataFrame: DataFrame with missing hourly slots filled.
    """
    
    # Get unique location IDs
    location_ids = agg_rides["pickup_location_id"].unique()

    # Create a full hourly range based on the min and max pickup hours
    full_range = pd.date_range(
        agg_rides["pickup_hour"].min(), 
        agg_rides["pickup_hour"].max(), freq="H"
    )
    
    output = pd.DataFrame()

    # Loop through each location ID to fill missing hours
    for location_id in tqdm(location_ids):
        agg_rides_i = agg_rides.loc[agg_rides.pickup_location_id == location_id, ['pickup_hour', 'rides']]

        agg_rides_i.set_index("pickup_hour", inplace=True)
        agg_rides_i.index = pd.DatetimeIndex(agg_rides_i.index)
        agg_rides_i = agg_rides_i.reindex(full_range, fill_value=0)

        agg_rides_i["pickup_location_id"] = location_id

        output = pd.concat([output, agg_rides_i])

    # Reset index and rename columns for the final output
    output = output.reset_index().rename(columns={"index": "pickup_hour"})

    return output


def transform_raw_data_into_time_series_data(rides: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw ride data into aggregated time series data.
    
    Parameters:
    - rides (pd.DataFrame): DataFrame with raw ride data.
    
    Returns:
    - pd.DataFrame: DataFrame with ride counts aggregated by hour and location.
    """
    
    # Extract the hour from the pickup datetime
    rides["pickup_hour"] = rides["pickup_datetime"].dt.floor('H')
    
    # Group by pickup hour and location, then count the number of rides
    agg_rides = rides.groupby(["pickup_hour", "pickup_location_id"]).size().reset_index()
    agg_rides.rename(columns={0: "rides"}, inplace=True)
    
    # Fill in missing hourly slots with zero rides
    agg_rides_all_slots = add_missing_slots(agg_rides=agg_rides)

    return agg_rides_all_slots

def transform_ts_data_into_features_and_targets(
        ts_data: pd.DataFrame,
        input_seq_len: int,
        step_size: int
) -> pd.DataFrame:
    """
    Transforms the time series data into features and targets for model training.
    
    Parameters:
    - ts_data (pd.DataFrame): DataFrame containing the time series data.
    - input_seq_len (int): Number of previous hours to use for feature creation.
    - step_size (int): Step size to move the window to create subsequences.
    
    Returns:
    - features (pd.DataFrame): DataFrame containing the features.
    - targets (pd.Series): Series containing the target values.
    """
    
    # Ensuring the input DataFrame has the expected columns
    assert set(ts_data.columns) == {"pickup_hour", "rides", "pickup_location_id"}

    # Getting unique location IDs
    location_ids = ts_data["pickup_location_id"].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    # Looping through each location ID to process data location-wise
    for location_id in tqdm(location_ids):
        # Filtering the data for one specific location
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location_id == location_id, ["pickup_hour", "rides"]
        ].sort_values(by=["pickup_hour"])
        
        # Getting indices for slicing the data into subsequences
        indices = get_cutoff_indices(
            ts_data_one_location,
            input_seq_len,
            step_size
        )
        
        # Creating arrays to hold features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []

        # Populating the feature and target arrays
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]["rides"].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]["rides"].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]["pickup_hour"])

        # Creating DataFrames for features and targets
        features_one_location = pd.DataFrame(
            x,
            columns=[f"rides_previous_{i+1}_hour" for i in reversed(range(input_seq_len))]
        )
        features_one_location["pickup_hour"] = pickup_hours
        features_one_location["pickup_location_id"] = location_id
        
        targets_one_location = pd.DataFrame(y, columns=[f"target_rides_next_hour"])

        # Concatenating the features and targets of each location
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])
    
    # Resetting index for clean DataFrames
    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets["target_rides_next_hour"]

def transform_ts_data_into_features_and_targets_optimized(
        ts_data: pd.DataFrame,
        input_seq_len: int,
        step_size: int
) -> (pd.DataFrame, pd.Series):
    """
    Transforms the time series data into features and targets for model training.
    
    Parameters:
    - ts_data (pd.DataFrame): DataFrame containing the time series data.
    - input_seq_len (int): Number of previous hours to use for feature creation.
    - step_size (int): Step size to move the window to create subsequences.
    
    Returns:
    - features (pd.DataFrame): DataFrame containing the features.
    - targets (pd.Series): Series containing the target values.
    """
    
    # Getting unique location IDs from the data
    location_ids = ts_data["pickup_location_id"].unique()
    all_features = []  # List to store features DataFrames for each location
    all_targets = []  # List to store targets Series for each location

    # Processing data per location
    for location_id in tqdm(location_ids):
        # Filtering data for one specific location
        ts_data_one_location = ts_data.loc[ts_data.pickup_location_id == location_id, ["pickup_hour", "rides"]]
        
        # Getting indices for creating subsequences
        indices = get_cutoff_indices(ts_data_one_location, input_seq_len, step_size)
        
        # Pre-allocating numpy arrays for features and targets
        x = np.zeros((len(indices), input_seq_len))
        y = np.zeros(len(indices))
        pickup_hours = []  # List to store pickup hours corresponding to targets

        # Populating the feature and target arrays using calculated indices
        for i, (start, mid, end) in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[start:mid]["rides"].values
            y[i] = ts_data_one_location.iloc[mid:end]["rides"].values[0]
            pickup_hours.append(ts_data_one_location.iloc[mid]["pickup_hour"])
        
        # Creating a DataFrame for features and appending it to the list
        features_one_location = pd.DataFrame(x, columns=[f"rides_previous_{i+1}_hour" for i in reversed(range(input_seq_len))])
        features_one_location["pickup_hour"] = pickup_hours
        features_one_location["pickup_location_id"] = location_id
        
        # Creating a Series for targets and appending it to the list
        targets_one_location = pd.Series(y, name="target_rides_next_hour")
        
        all_features.append(features_one_location)
        all_targets.append(targets_one_location)
    
    # Concatenating all DataFrames and Series in the lists to get final features and targets
    features = pd.concat(all_features).reset_index(drop=True)
    targets = pd.concat(all_targets).reset_index(drop=True)

    return features, targets


def get_cutoff_indices(data: pd.DataFrame, n_features: int, step_size: int) -> list:
    """
    Calculate indices for slicing the DataFrame into subsequences for model training.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing the time series data.
    - n_features (int): The number of features to include in each subsequence.
    - step_size (int): The step size to move the window for each new subsequence.
    
    Returns:
    - list: A list of tuples where each tuple contains the start, middle, and end indices of each subsequence.
    """
    
    # Determine the last valid index position in the DataFrame
    stop_position = len(data) - 1

    # Initialize the indices for the first, middle, and last positions of the subsequence
    subseq_first_idx = 0
    subseq_mid_idx = n_features
    subseq_last_idx = n_features + 1
    indices = []

    # Loop until the end index of the subsequence is within the DataFrame
    while subseq_last_idx <= stop_position:
        # Append the current set of indices as a tuple to the list
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        
        # Update the indices for the next iteration based on the step size
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indices


