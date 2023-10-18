from datetime import datetime
from typing import Tuple

import pandas as pd

def train_test_split(
        df: pd.DataFrame,
        cutoff_date: datetime,
        target_column_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the DataFrame into training and test sets based on a cutoff date.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with features and target.
    - cutoff_date (datetime): The date on which to split the data.
    - target_column_name (str): The name of the target column in the DataFrame.
    
    Returns:
    - Tuple: X_train, y_train, X_test, y_test DataFrames and Series.
    """
    
    # Filtering rows based on the cutoff date to create training and test sets
    train_data = df[df.pickup_hour < cutoff_date].reset_index(drop=True)
    test_data = df[df.pickup_hour >= cutoff_date].reset_index(drop=True)
    
    # Separating features and target for both training and test sets
    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]
    
    return X_train, y_train, X_test, y_test
