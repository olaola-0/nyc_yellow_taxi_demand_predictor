from typing import Optional
from datetime import datetime

import pandas as pd
import fire

from src.inference import load_batch_of_features_from_store, load_model_from_registry, get_model_predictions
from src.feature_store_api import get_or_create_feature_group, FEATURE_GROUP_PREDICTIONS_METADATA
from src.model_registry_api import get_latest_model_from_registry
from src.logger import get_logger

logger = get_logger()

def inference(current_date: Optional[pd.Timestamp] = pd.to_datetime(datetime.utcnow()).floor('H')) -> None:
    """
    Run the inference pipeline to make predictions using features from the feature store
    and the latest model from the model registry.

    Parameters:
        current_date (Optional[pd.Timestamp]): The current timestamp for which to make predictions.
            Defaults to the current time floored to the nearest hour.

    This function performs the following steps:
    - Loads a batch of features corresponding to the `current_date`.
    - Retrieves the latest model from the model registry.
    - Generates predictions using the loaded features and model.
    - Saves the predictions to the feature store.
    """
    try:
        logger.info(f'Running inference pipeline for {current_date}')

        # Step 1: Load features
        logger.info('Loading batch of features from the feature store')
        features = load_batch_of_features_from_store(current_date=current_date)
        logger.info(f'Loaded {len(features)} features')

        # Step 2: Load model
        logger.info('Loading model from the model registry')
        model = get_latest_model_from_registry()
        # Consider adding model input schema validation here

        # Step 3: Generate predictions
        logger.info('Generating predictions')
        predictions = get_model_predictions(model=model, features=features)
        logger.info(f'Generated predictions with shape: {predictions.shape}')
        predictions['pickup_hour'] = current_date

        # Step 4: Save predictions
        logger.info('Saving predictions to the feature store')
        feature_group = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA)
        feature_group.insert(predictions, write_options={"wait_for_job": False})
        logger.info('Predictions saved to the feature store')

        logger.info('Inference DONE!')
    except Exception as e:
        logger.error(f'Inference pipeline failed: {e}')
        raise

if __name__ == '__main__':
    # Use Python Fire to handle command-line arguments
    fire.Fire(inference)
