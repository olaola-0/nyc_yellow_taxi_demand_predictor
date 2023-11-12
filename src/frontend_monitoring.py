"""
This Streamlit script creates an interactive dashboard for monitoring the performance of NYC yellow taxi demand prediction model.
It visualizes the mean absolute error (MAE) of predictions both hourly and location-wise.
The script utilizes caching to enhance performance and reduce data fetching time.
Data visualization is handled using Plotly for an interactive user experience.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import mean_absolute_error
from src.monitoring import load_predictions_and_actual_values_from_store

st.set_page_config(layout="wide")

# Setting the current date for the monitoring dashboard
current_date = pd.to_datetime(datetime.utcnow()).floor('H')
st.title('Monitoring dashboard 🔎')

# Initialize progress bar in the sidebar
progress_bar = st.sidebar.header('⚙️ Work in Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 3


# Data Loading Function with Caching
@st.cache_data
def _load_predictions_and_actual_values_from_store(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """Caches and returns the result of loading predictions and actual values from the store.

    Args:
        from_date (datetime): The start date for fetching data.
        to_date (datetime): The end date for fetching data.

    Returns:
        pd.DataFrame: A DataFrame containing model predictions and actual values.
        - 'pcikup_loction_id'
        - 'predicted_demand'
        - 'pickup_hour'
        - 'rides'
    """
    return load_predictions_and_actual_values_from_store(from_date, to_date)


# Fetching Model Predictions and Actual Values
with st.spinner(text="Fetching model predictions and actual values from the store"):
    monitoring_df = _load_predictions_and_actual_values_from_store(
        from_date=current_date - timedelta(days=14),
        to_date=current_date
    )
    st.sidebar.write('✅ Model predictions and actual values arrived')
    progress_bar.progress(1/N_STEPS)


# Plotting Aggregate MAE Hour-by-Hour
with st.spinner(text="Plotting aggregate MAE hour-by-hour"):
    st.header('Mean Absolute Error (MAE) hour-by-hour')
    # Aggregate and calculate MAE for each hour
    mae_per_hour = (
        monitoring_df
        .groupby('pickup_hour')
        .apply(lambda g: mean_absolute_error(g['rides'], g['predicted_demand']))
        .reset_index()
        .rename(columns={0: 'mae'})
        .sort_values(by='pickup_hour')
    )

    # Plotting the MAE using Plotly
    fig = px.bar(
        mae_per_hour,
        x='pickup_hour', y='mae',
        template='plotly_dark',
    )
    st.plotly_chart(fig, use_container_width=True)
    progress_bar.progress(2/N_STEPS)


# Plotting MAE Hour-by-Hour for Top Locations
with st.spinner(text="Plotting MAE hour-by-hour for top locations"):
    st.header('Mean Absolute Error (MAE) per location and hour')
    # Identify top locations by demand
    top_locations_by_demand = (
        monitoring_df
        .groupby('pickup_location_id')['rides']
        .sum()
        .sort_values(ascending=False)
        .head(10)['pickup_location_id']
    )

    # Iterate and plot MAE for each top location
    for location_id in top_locations_by_demand:
        mae_per_hour = (
            monitoring_df[monitoring_df.pickup_location_id == location_id]
            .groupby('pickup_hour')
            .apply(lambda g: mean_absolute_error(g['rides'], g['predicted_demand']))
            .reset_index()
            .rename(columns={0: 'mae'})
            .sort_values(by='pickup_hour')
        )

        fig = px.bar(
            mae_per_hour,
            x='pickup_hour', y='mae',
            template='plotly_dark',
        )
        st.subheader(f'Location ID: {location_id}')
        st.plotly_chart(fig, use_container_width=True)

    progress_bar.progress(3/N_STEPS)
