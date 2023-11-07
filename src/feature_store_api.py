from typing import Optional, List
from dataclasses import dataclass

import hsfs
import hopsworks

import src.config as config
from src.logger import get_logger

logger = get_logger()

@dataclass
class FeatureGroupConfig:
    """
    A data class representing the configuration of a feature group.
    
    Attributes:
        name (str): The name of the feature group.
        version (int): The version of the feature group.
        description (str): A brief description of the feature group.
        primary_key (List[str]): A list of columns that will act as primary keys.
        event_time (str): The event time column name.
        online_enabled (Optional[bool]): Flag to enable online serving. Defaults to False.
    """
    name: str
    version: int
    description: str
    primary_key: List[str]
    event_time: str
    online_enabled: Optional[bool] = False

@dataclass
class FeatureViewConfig:
    """
    A data class representing the configuration of a feature view.
    
    Attributes:
        name (str): The name of the feature view.
        version (str): The version of the feature view.
        feature_group (FeatureGroupConfig): The feature group configuration that this view is based on.
    """
    name: str
    version: str
    feature_group: FeatureGroupConfig

def get_feature_store() -> hsfs.feature_store.FeatureStore:
    """
    Logs into the Hopsworks project and retrieves the feature store.
    
    Returns:
        hsfs.feature_store.FeatureStore: The feature store of the Hopsworks project.
    """
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    return project.get_feature_store()

def get_or_create_feature_group(
        feature_group_metadata: FeatureGroupConfig
        ) -> hsfs.feature_group.FeatureGroup:
    """
    Retrieves or creates a feature group in the feature store based on the provided metadata.
    
    Args:
        feature_group_metadata (FeatureGroupConfig): The metadata for the feature group.
        
    Returns:
        hsfs.feature_group.FeatureGroup: The retrieved or newly created feature group.
    """
    return get_feature_store().get_or_create_feature_group(
        name=feature_group_metadata.name,
        version=feature_group_metadata.version,
        description=feature_group_metadata.description,
        primary_key=feature_group_metadata.primary_key,
        event_time=feature_group_metadata.event_time,
        online_enabled=feature_group_metadata.online_enabled
    )

def get_or_create_feature_view(feature_view_metadata: FeatureViewConfig) -> hsfs.feature_view.FeatureView:
    """
    Retrieves or creates a feature view in the feature store based on the provided metadata.
    
    Args:
        feature_view_metadata (FeatureViewConfig): The metadata for the feature view.
        
    Returns:
        hsfs.feature_view.FeatureView: The retrieved or newly created feature view.
    """
    feature_store = get_feature_store()

    feature_group = feature_store.get_feature_group(
        name=feature_view_metadata.feature_group.name,
        version=feature_view_metadata.feature_group.version
    )

    # Attempt to create a feature view if it doesn't exist
    try:
        feature_store.create_feature_view(
            name=feature_view_metadata.name,
            version=feature_view_metadata.version,
            query=feature_group.select_all()
        )
    except Exception as e:
        logger.info(f"Feature view already exists or another error occurred: {e}")

    # Retrieve the feature view
    feature_store = get_feature_store()
    feature_view = feature_store.get_feature_view(
        name=feature_view_metadata.name,
        version=feature_view_metadata.version
    )

    return feature_view


# Instantiate the FeatureGroupConfig and FeatureViewConfig with the values from config.py
FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name=config.FEATURE_GROUP_NAME,
    version=config.FEATURE_GROUP_VERSION,
    description='Feature group with hourly time-series data of historical taxi rides',
    primary_key=['pickup_location_id', 'pickup_hour'],
    event_time='pickup_hour'
)

FEATURE_VIEW_METADATA = FeatureViewConfig(
    name=config.FEATURE_VIEW_NAME,
    version=config.FEATURE_VIEW_VERSION,
    feature_group=FEATURE_GROUP_METADATA
)

# Configuration for feature group & view predictions
FEATURE_GROUP_PREDICTIONS_METADATA = FeatureGroupConfig(
    name='model_predictions_feature_group',
    version=1,
    description="Predictions generate by our production model",
    primary_key = ['pickup_location_id', 'pickup_hour'],
    event_time='pickup_hour',
)

FEATURE_VIEW_PREDICTIONS_METADATA = FeatureViewConfig(
    name='model_predictions_feature_view',
    version=1,
    feature_group=FEATURE_GROUP_PREDICTIONS_METADATA,
)