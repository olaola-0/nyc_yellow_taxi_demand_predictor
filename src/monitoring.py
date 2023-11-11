from datetime import datetime, timedelta

import pandas as pd

import src.config as config
from src.feature_store_api import get_feature_store, get_or_create_feature_group, FEATURE_GROUP_METADATA, FEATURE_GROUP_PREDICTIONS_METADATA


def load_predictions_and_actual_values_from_store(from_date: datetime, to_date: datetime) -> pd.DataFrame:

    predictions_fg = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA)
    actual_fg = get_or_create_feature_group(FEATURE_GROUP_METADATA)

    query = predictions_fg.select_all()\
        .join(actual_fg.select_all(), on=['pickup_hour', 'pickup_location_id'])\
        .filter(predictions_fg.pickup_hour >= from_date)\
        .filter(predictions_fg.pickup_hour <= to_date)
    
    feature_store = get_feature_store()
    try:
        feature_store.create_feature_view(
            name=config.FEATURE_VIEW_MONITORING,
            version=1,
            query=query
        )
    except:
        print('Feature view already existed. Skip creation')

    monitoring_fv = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_MONITORING,
        version=1
    )

    monitoring_df = monitoring_fv.get_batch_data(
        start_time=from_date - timedelta(days=7),
        end_time=to_date +timedelta(days=7)
    )

    monitoring_df = monitoring_df[monitoring_df.pickup_hour.between(from_date, to_date)]

    return monitoring_df