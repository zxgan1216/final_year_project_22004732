import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import plotly.graph_objs as go
from scipy.special import softmax
import time

# Base directories
DATASET_DIR = './FYP_Kubernetes/cleaned_dataset/'
MODEL_DIRECTORY = './FYP_Kubernetes/ARIMA_Models/'
ARIMA_ACCURACY_FILE = './FYP_Kubernetes/ARIMA_Forecast_Results/arima_summary_results.csv'

# Pod statuses
STATUSES = ["Running", "Pending", "Succeeded", "Failed", "Unknown"]

# Function to load data for a selected node
def load_data(node_index):
    file_path = os.path.join(DATASET_DIR, f"node_node_{node_index}_dataset.csv")
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        return data
    else:
        st.error(f"Data file not found for Node {node_index}. Please check the dataset.")
        return pd.DataFrame()

# Function to load ARIMA model for a specific node and status
def load_arima_model(node_index, status):
    model_path = os.path.join(MODEL_DIRECTORY, f"Node_{node_index}_Status_{status}_ARIMA_Model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        st.error(f"ARIMA model for Node {node_index}, Status {status} not found.")
        return None

# Function to forecast probabilities for all statuses
def forecast_probabilities(data, node_index, statuses, forecast_steps):
    probabilities = {}
    for status in statuses:
        # Load the ARIMA model
        model = load_arima_model(node_index, status)
        if model is None:
            continue
        
        # Resample and preprocess pod status data
        pod_status_series = data[f'pod_status_{status}'].resample('1T').mean().fillna(method='ffill').fillna(method='bfill')

        # Forecast the next 'forecast_steps'
        forecast = model.forecast(steps=forecast_steps)
        
        # Apply softmax to normalize probabilities
        probabilities[status] = softmax(forecast.values.reshape(-1, 1), axis=0).flatten() * 100  # Convert to percentage
    
    # Create a DataFrame for the probabilities
    probability_df = pd.DataFrame(
        probabilities,
        index=pd.date_range(start=data.index[-1], periods=forecast_steps, freq='1T')
    )
    return probability_df

# Function to plot the heatmap using Plotly
def plot_heatmap(probability_df, node_index):
    heatmap = go.Figure(data=go.Heatmap(
        z=probability_df.values.T,
        x=probability_df.index.strftime("%H:%M:%S"),
        y=probability_df.columns,
        colorscale="Viridis",
        colorbar=dict(title="Probability"),
    ))
    heatmap.update_layout(
        title=f"Forecasted Probabilities for Pod Statuses - Node {node_index}",
        xaxis_title="Time (HH:MM:SS)",
        yaxis_title="Pod Status",
        template="plotly_dark",
    )
    st.plotly_chart(heatmap, use_container_width=True)

# Function to display ARIMA accuracy (AIC)
def display_arima_accuracy(node_index, arima_summary_df):
    st.markdown(f"### ARIMA Model Accuracy for Node {node_index}")
    node_accuracy_df = arima_summary_df[arima_summary_df["Node"] == f"Node_{node_index}"]
    if not node_accuracy_df.empty:
        st.table(node_accuracy_df)
    else:
        st.warning(f"No ARIMA accuracy data available for Node {node_index}.")

# Streamlit app
def app(node):
    st.title("ARIMA Model Forecast Dashboard")
    st.markdown("Analyze the next n-step forecasted probabilities for pod statuses using the ARIMA model.")

    # Extract node index from the passed node
    node_index = int(node.split(" ")[1])  # Extract the node number (e.g., Node 0 -> 0)

    # Load data for the selected node
    st.markdown(f"### Data for Node {node_index}")
    data = load_data(node_index)
    if data.empty:
        return

    st.write("Original Data (20 rows):", data.head(20))

    # Load ARIMA accuracy file
    arima_summary_df = pd.read_csv(ARIMA_ACCURACY_FILE)

    # Display ARIMA accuracy (AIC) for the node
    display_arima_accuracy(node_index, arima_summary_df)

    # User choose for the number of forecast steps
    forecast_steps = st.slider("Select the number of forecast steps:", 1, 30, 15)

    # Forecast probabilities
    st.markdown(f"### Forecasted Probabilities for Node {node_index}")
    probability_df = forecast_probabilities(data, node_index, STATUSES,forecast_steps)
    
    if not probability_df.empty:
        # Format probabilities with 20 decimal places
        probability_df_formatted = probability_df.applymap(lambda x: f"{x:.6f}%")
        
        # Display the probabilities in a table
        st.markdown("#### Forecasted Probabilities Table")
        st.dataframe(probability_df_formatted, use_container_width=True)
        
        # Display the probabilities as a heatmap
        st.markdown("#### Forecasted Probabilities Heatmap")
        plot_heatmap(probability_df, node_index)
    else:
        st.warning("No probabilities to display. Ensure ARIMA models are trained for the selected node.")

    time.sleep(1)  # Refresh interval in seconds
    st.experimental_rerun()