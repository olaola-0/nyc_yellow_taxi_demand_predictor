from typing import Optional
from datetime import timedelta

import pandas as pd
import plotly.express as px


def plot_one_sample(
        features: pd.DataFrame,
        targets: pd.Series,
        example_id: int,
        predictions: Optional[pd.Series] = None,
):
    """
    Plot a single sample of time series data, its actual target value, and optionally, a prediction.
    
    Parameters:
    - features (pd.DataFrame): DataFrame containing feature values.
    - targets (pd.Series): Series containing the actual target values.
    - example_id (int): The index of the sample to be plotted.
    - predictions (Optional[pd.Series]): Series containing predicted values, default is None.
    
    Returns:
    - plotly.graph_objs._figure.Figure: A Plotly Figure object to be displayed or saved.
    """
    
    # Extract the features and target for the specified example_id
    features_ = features.iloc[example_id]
    target_ = targets.iloc[example_id]

    # Identify columns that contain time series data
    ts_columns = [c for c in features.columns if c.startswith("rides_previous_")]
    
    # Get time series values and the corresponding target value
    ts_values = [features_[c] for c in ts_columns] + [target_]
    
    # Generate date range based on the pickup_hour
    ts_dates = pd.date_range(
        features_["pickup_hour"] - timedelta(hours=len(ts_columns)),
        features_["pickup_hour"],
        freq="H"
    )

    # Create the plot title
    title = f"Pick up hour={features_['pickup_hour']}, location_id={features_['pickup_location_id']}"
    
    # Create a line plot of the time series data
    fig = px.line(
        x=ts_dates, y=ts_values,
        template="plotly_dark",
        markers=True, title=title
    )

    # Add a scatter plot of the actual target value
    fig.add_scatter(x=ts_dates[-1:], y=[target_],
                    line_color="green",
                    mode="markers", marker_size=10, name="actual value")
    
    # If predictions are provided, add a scatter plot of the predicted value
    if predictions is not None:
        prediction_ = predictions.iloc[example_id]
        fig.add_scatter(x=ts_dates[-1], y=[prediction_],
                        line_color="red",
                        mode="markers", marker_symbol="x", marker_size=15,
                        name="prediction")
    
    return fig
