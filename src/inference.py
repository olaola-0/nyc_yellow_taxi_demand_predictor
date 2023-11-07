from datetime import datetime, timedelta
import hopsworks
import pandas as pd
import numpy as np
import src.config as config
from src.feature_store_api import get_or_create_feature_view, FEATURE_VIEW_METADATA, FEATURE_VIEW_PREDICTIONS_METADATA


def get_hopsworks_project() -> hopsworks.project.Project:
    """
    Authenticate and retrieve the Hopsworks project instance.

    Returns:
        hopsworks.project.Project: An authenticated Hopsworks project object.
    """
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for the demand at various pickup locations.

    Parameters:
        model: The trained model that will be used for making predictions.
        features: A DataFrame containing the features required by the model.

    Returns:
        A DataFrame with pickup locations and their corresponding predicted demand.
    """
    # Generate predictions using the provided model and features
    predictions = model.predict(features)

    # Prepare results DataFrame with the required information
    results = pd.DataFrame({
        'pickup_location_id': features['pickup_location_id'].values,
        'predicted_demand': predictions.round(0)  # Round predictions to the nearest whole number
    })

    return results


def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """
    Load a batch of features from the feature store based on the current date.

    Parameters:
        current_date: The date for which features should be loaded.

    Returns:
        A DataFrame containing features for each pickup location up to the current date.
    """
    # Number of historical features to fetch
    n_features = config.N_FEATURES

    # Retrieve the configured feature view from the feature store
    feature_view = get_or_create_feature_view(feature_view_metadata=FEATURE_VIEW_METADATA)

    # Calculate the time range for fetching features
    fetch_data_to = pd.to_datetime(current_date - timedelta(hours=1), utc=True)
    fetch_data_from = pd.to_datetime(current_date - timedelta(days=28), utc=True)

    # Fetch the time series data within the specified time range
    ts_data = feature_view.get_batch_data(
        start_time=fetch_data_from - timedelta(days=1),
        end_time=fetch_data_to + timedelta(days=1),
    )

    # Ensure the 'pickup_hour' column is in UTC timezone for consistency
    ts_data['pickup_hour'] = pd.to_datetime(ts_data['pickup_hour'], utc=True)

    # Filter data to the desired time range
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

    # Check that the fetched data is complete
    location_ids = ts_data['pickup_location_id'].unique()
    if len(ts_data) != config.N_FEATURES * len(location_ids):
        raise ValueError("Time-series data is not complete. Make sure your feature pipeline is up and running.")

    # Sort the data by location ID and pickup hour
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

    # Transform time-series data into feature vectors for each location ID
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data[ts_data.pickup_location_id == location_id]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['rides'].values

    # Convert numpy arrays into a pandas DataFrame with appropriate column names
    features = pd.DataFrame(x, columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))])
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features


def load_model_from_registry():
    """
    Load the trained model from the model registry.

    Returns:
        The trained model ready for making predictions.
    """
    import joblib
    from pathlib import Path

    # Get the Hopsworks project and model registry
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    # Retrieve the model from the registry using configured name and version
    model = model_registry.get_model(name=config.MODEL_NAME, version=config.MODEL_VERSION)
    
    # Download the model to a local directory and load it
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / 'model.pkl')
    
    return model

def load_predictions_from_store(from_pickup_hour: datetime, to_pickup_hour: datetime) -> pd.DataFrame:
    """
    Load and return the predictions for taxi demand from the feature store for a specified time range.

    This function retrieves the predictions data from a feature view within the feature store. The data
    is filtered to include only the predictions between the specified start and end pickup hours.

    Parameters:
        from_pickup_hour (datetime): The start datetime for fetching predictions.
        to_pickup_hour (datetime): The end datetime for fetching predictions.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions data for the specified time range.

    Raises:
        ValueError: If the feature view cannot be accessed or the data is not found for the given time range.
    """
    # Retrieve or create the feature view for predictions based on the metadata configuration
    predictions_fv = get_or_create_feature_view(FEATURE_VIEW_PREDICTIONS_METADATA)
    
    # Log the action of fetching data for the specified time range
    print(f'Fetching predictions for `pickup_hours` between {from_pickup_hour} and {to_pickup_hour}')
    
    # Get the batch data from the feature view, extending the range by one day on both ends
    predictions = predictions_fv.get_batch_data(
        start_time=from_pickup_hour - timedelta(days=1),
        end_time=to_pickup_hour + timedelta(days=1)
    )

    # Ensure the 'pickup_hour' column is in UTC for consistency with other time data
    predictions['pickup_hour'] = pd.to_datetime(predictions['pickup_hour'], utc=True)
    
    # Convert the input times to UTC to match the 'pickup_hour' column
    from_pickup_hour = pd.to_datetime(from_pickup_hour, utc=True)
    to_pickup_hour = pd.to_datetime(to_pickup_hour, utc=True)

    # Filter the predictions to include only the rows within the specified time range
    predictions = predictions[predictions.pickup_hour.between(from_pickup_hour, to_pickup_hour)]

    # Sort the predictions data by 'pickup_hour' and 'pickup_location_id' for organized output
    predictions.sort_values(by=['pickup_hour', 'pickup_location_id'], inplace=True)

    return predictions

