import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
import lightgbm as lgb

def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average number of rides in the last 4 weeks at the same hour and day.
    
    Parameters:
    - X (pd.DataFrame): A DataFrame containing historical ride data.
    
    Returns:
    - pd.DataFrame: The original DataFrame with an added column for the average rides in the last 4 weeks.
    """
    X['average_rides_last_4_weeks'] = 0.25*(
        X[f'rides_previous_{7*24}_hour'] + \
        X[f'rides_previous_{2*7*24}_hour'] + \
        X[f'rides_previous_{3*7*24}_hour'] + \
        X[f'rides_previous_{4*7*24}_hour']
    )
    return X


class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    A transformer that extracts temporal features from datetime columns.
    
    This transformer adds two new columns to the DataFrame:
    - hour: The hour of the day.
    - day_of_week: The day of the week.
    
    The `pickup_hour` column is dropped after the features are extracted.
    """
    def fit(self, X, y=None):
        """Fit method. No fitting necessary for this transformer."""
        return self
    
    def transform(self, X, y=None):
        """
        Transform method to execute the feature engineering steps.
        
        Parameters:
        - X (pd.DataFrame): A DataFrame containing a datetime column `pickup_hour`.
        
        Returns:
        - pd.DataFrame: The DataFrame with added temporal features and dropped `pickup_hour`.
        """
        X_ = X.copy()
        
        # Extracting the hour and day of the week from the `pickup_hour` column
        X_["hour"] = X_['pickup_hour'].dt.hour
        X_["day_of_week"] = X_['pickup_hour'].dt.dayofweek
        
        return X_.drop(columns=['pickup_hour'])

def get_pipeline(**hyperparams) -> Pipeline:
    """
    Create and return a pipeline that preprocesses the data and fits a LightGBM regressor.
    
    Parameters:
    - **hyperparams: Arbitrary keyword arguments to pass to the LightGBM regressor.
    
    Returns:
    - Pipeline: A scikit-learn pipeline that preprocesses the data and fits a regressor.
    """
    # Creating a transformer to add the average rides feature
    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        average_rides_last_4_weeks, validate=False)
    
    # Creating a transformer to add temporal features
    add_temporal_features = TemporalFeaturesEngineer()

    # Creating a pipeline with data preprocessing steps and a LightGBM regressor
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )
