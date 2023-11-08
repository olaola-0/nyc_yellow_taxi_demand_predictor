from typing import Optional
from pathlib import Path

import hopsworks
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

import src.config as config
from src.paths import MODELS_DIR
from src.logger import get_logger

logger = get_logger()

def get_model_registry():
    """
    Log in to the Hopsworks project and retrieve the model registry.
    
    Returns:
        The model registry associated with the Hopsworks project.
    """
    try:
        # Ensure that the API key and project name are present
        if not config.HOPSWORKS_PROJECT_NAME or not config.HOPSWORKS_API_KEY:
            raise ValueError("Hopsworks project name or API key is not set in the configuration.")

        # Log in to Hopsworks using the project name and API key specified in the configuration
        project = hopsworks.login(
            project=config.HOPSWORKS_PROJECT_NAME,
            api_key_value=config.HOPSWORKS_API_KEY
        )
        return project.get_model_registry()
    except Exception as e:
        logger.error(f"Failed to log in to Hopsworks and retrieve the model registry: {e}")
        raise

def push_model_to_registry(
        model: Pipeline,
        X_train_sample: pd.DataFrame,
        y_train_sample: pd.DataFrame,
        test_mae: float,
        description: Optional[str] = '',
) -> int:
    """
    Push the trained model to the Hopsworks model registry.
    
    Parameters:
        model (Pipeline): The trained machine learning model.
        X_train_sample (pd.DataFrame): Sample input features used for training.
        y_train_sample (pd.DataFrame): Sample target values used for training.
        test_mae (float): Mean absolute error of the model on the test dataset.
        description (Optional[str]): Description of the model.
        
    Returns:
        int: The version of the model pushed to the registry.
    """
    try:
        # Serialize the model to a file
        model_path = MODELS_DIR / 'model.pkl'
        joblib.dump(model, model_path)

        # Create schema for input and output based on training samples
        input_schema = Schema(X_train_sample)
        output_schema = Schema(y_train_sample)
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

        # Get the model registry object
        model_registry = get_model_registry()
        # Create a model entity in the registry with the specified schema and metrics
        model_version = model_registry.sklearn.create_model(
            name="yellow_taxi_demand_predictor_next_hour",
            metrics={"test_mae": test_mae},
            description=description,
            input_example=X_train_sample.iloc[0].to_dict(),
            model_schema=model_schema
        )
        # Save the serialized model to the registry
        model_version.save(model_path)
        
        logger.info(f'Model version {model_version.version} pushed to registry successfully.')
        return model_version.version
    except Exception as e:
        logger.error(f"Failed to push model to registry: {e}")
        raise

def get_latest_model_from_registry() -> Pipeline:
    """
    Retrieve the latest version of the model from the Hopsworks model registry.
    
    Returns:
        Pipeline: The latest version of the trained model.
    """
    try:
        # Get the model registry object
        model_registry = get_model_registry()
        # Retrieve the latest model from the registry based on the name specified in the configuration
        model_metadata = model_registry.get_models(name=config.MODEL_NAME, sort_by="CREATION_TIME", sort_order="DESC")[0]
        logger.info(f'Loading model version {model_metadata.version}')

        # Download the model artifacts
        model_dir = model_metadata.download()
        model = joblib.load(Path(model_dir) / 'model.pkl')

        logger.info('Model loaded successfully from registry.')
        return model
    except Exception as e:
        logger.error

