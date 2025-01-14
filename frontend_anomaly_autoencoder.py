import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score

# Base directories
DATASET_DIR = './FYP_Kubernetes/cleaned_dataset/'
MODEL_DIR = './FYP_Kubernetes/Autoencoder_Model/'

# File naming format
FILE_PREFIX = 'node_node_'
FILE_SUFFIX = '_dataset.csv'

# Function to load data for a selected node
def load_data(node_index):
    file_path = os.path.join(DATASET_DIR, f"{FILE_PREFIX}{node_index}{FILE_SUFFIX}")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"Data file not found for Node_{node_index}. Please check the dataset.")
        return pd.DataFrame()

# Function to preprocess data (drop unnecessary columns)
def preprocess_data(data):
    drop_columns = ['pod_status_Pending', 'pod_status_Running',
                    'pod_status_Succeeded', 'pod_status_Failed', 'pod_status_Unknown', 'node_name']
    return data.drop(columns=drop_columns, inplace=False, errors='ignore')

# Function to load the autoencoder model for a node
def load_autoencoder_model(node_index):
    model_path = os.path.join(MODEL_DIR, f"Autoencoder_Model_Node_{node_index}.h5")
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error(f"Autoencoder model not found for Node_{node_index}. Please train the model first.")
        return None

# Function to calculate accuracy of the Autoencoder model
def calculate_model_accuracy(autoencoder, data):
    # Ensure only numeric data
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.fillna(numeric_data.mean())  # Handle missing values

    # Split data into 80% train and 20% test
    train_size = int(0.8 * len(numeric_data))
    train_data = numeric_data[:train_size]
    test_data = numeric_data[train_size:]

    # Predict reconstruction for test data
    ae_predictions = autoencoder.predict(test_data)

    # Calculate reconstruction loss
    reconstruction_loss = np.mean(np.power(ae_predictions - test_data.values, 2), axis=1)

    # Set threshold based on the 95th percentile of train reconstruction loss
    train_predictions = autoencoder.predict(train_data)
    train_loss = np.mean(np.power(train_predictions - train_data.values, 2), axis=1)
    threshold = np.percentile(train_loss, 95)

    # Label anomalies (0 for anomalies, 1 for normal)
    y_pred = np.where(reconstruction_loss > threshold, 0, 1)
    y_true = np.ones(len(y_pred))  # Assuming all test samples are normal

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# Function to analyze and display anomalous features
def analyze_anomalous_features(autoencoder, data):
    # Ensure only numeric data
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.fillna(numeric_data.mean())  # Handle missing values

    # Keep the corresponding timestamps
    timestamps = data['timestamp']

    # Predict reconstructed data
    ae_predictions = autoencoder.predict(numeric_data)

    # Calculate per-feature anomaly scores
    ae_feature_anomaly_scores = np.power(ae_predictions - numeric_data.values, 2)

    # Average anomaly score across features for each row
    ae_loss = np.mean(ae_feature_anomaly_scores, axis=1)

    # Identify anomalous features
    anomalous_features = []
    for row_idx, scores in enumerate(ae_feature_anomaly_scores):
        feature_indices = np.where(scores > np.percentile(scores, 95))[0]  # Features with top 5% anomaly scores
        feature_names = numeric_data.columns[feature_indices]
        anomalous_features.append(", ".join(feature_names))

    # Count total anomalies detected
    total_anomalies = sum([1 for features in anomalous_features if features])

    # Prepare a DataFrame summarizing results
    example_results = pd.DataFrame({
        "Timestamp": timestamps,
        "Autoencoder Loss": ae_loss,
        "Anomalous Features": anomalous_features
    })

    # Display results in an interactive table
    st.markdown("### Model Outputs and Anomalous Features")
    st.dataframe(example_results, use_container_width=True)  # Show all rows and fit to page width

# Main Streamlit app
def app(node):
    st.title("Autoencoder Anomaly Detection Dashboard")
    st.markdown("Analyze anomalies for a node using the Autoencoder model.")

    # Extract node index from the passed node
    node_index = int(node.split(" ")[1])  # Extract the node number

    # Load and preprocess data
    st.markdown(f"### Data for {node}")
    data = load_data(node_index)
    if data.empty:
        return

    st.write("Original Data (20 rows):", data.head(20))
    data = preprocess_data(data)
    st.write("Processed Data (20 rows):", data.head(20))

    # Ensure the timestamp column exists
    if 'timestamp' not in data.columns:
        st.error("Timestamp column not found in the dataset.")
        return

    # Load the autoencoder model
    autoencoder = load_autoencoder_model(node_index)
    if autoencoder is None:
        return

    # Calculate and display model accuracy
    accuracy = calculate_model_accuracy(autoencoder, data)
    st.markdown(f"### Autoencoder Model Accuracy: **{accuracy:.2%}**")

    # Analyze and display anomalous features
    analyze_anomalous_features(autoencoder, data)