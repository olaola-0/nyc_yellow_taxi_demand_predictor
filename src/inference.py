from datetime import datetime, timedelta

import hopsworks

import pandas as pd
import numpy as np

import src.config as config

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_model_predictons(model, features: pd.DataFrame) -> pd.DataFrame:
    
    