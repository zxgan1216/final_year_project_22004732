import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objs as go

# Base directories
DATASET_DIR = './cleaned_dataset'
MODEL_DIR = './One_Class_SVM_Model/'
ACCURACY_FILE = './Accuracy_Summary/one_class_svm_accuracy_summary.csv'

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

# Function to load column-specific One-Class SVM models
def load_column_model(node_index, column_name):
    model_path = os.path.join(MODEL_DIR, f"One_Class_SVM_Model_Node_{node_index}_Column_{column_name}.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        st.error(f"Model not found for Node_{node_index}, Column {column_name}. Please train the model first.")
        return None

# Function to detect anomalies for a single column
def detect_column_anomalies(data, column, model):
    if model is None or data.empty or column not in data.columns:
        return pd.DataFrame(), 0

    # Use only the specified column for prediction
    column_data = data[[column]].dropna()
    data['anomalies'] = np.where(model.predict(column_data) == -1, 'Anomaly', 'Normal')
    anomalies = data[data['anomalies'] == 'Anomaly']

    return anomalies, len(anomalies)

# Function to visualize anomalies for a single column
def visualize_column_anomalies(data, column):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data[column],
        mode='lines+markers',
        name='Data Points',
        marker=dict(color=np.where(data['anomalies'] == 'Anomaly', 'red', 'blue'))
    ))
    fig.update_layout(
        title=f"Anomaly Detection for {column}",
        xaxis_title="Timestamp",
        yaxis_title=column,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# Function to display anomalies for a single column in a table
def display_column_anomalies_table(anomalies, column):
    anomalies_table = anomalies[['timestamp', column]].dropna()
    st.table(anomalies_table)

# Function to display the count of anomalies for a column
def display_column_anomalies_count(column, count):
    st.write(f"Number of anomalies detected in {column}: **{count}**")

# Function to load accuracy data
def load_accuracy_data():
    if os.path.exists(ACCURACY_FILE):
        return pd.read_csv(ACCURACY_FILE)
    else:
        st.error("Accuracy summary file not found. Please ensure it is in the correct location.")
        return pd.DataFrame()

# Function to display accuracy for the selected node
def display_node_accuracy(node_index):
    accuracy_data = load_accuracy_data()
    if accuracy_data.empty:
        return

    # Filter the accuracy data for the selected node
    node_row = accuracy_data[accuracy_data['Node'] == f"Node_{node_index}"]
    if node_row.empty:
        st.error(f"Accuracy data not found for Node_{node_index}.")
    else:
        accuracy = node_row['Accuracy'].values[0]
        anomalies_detected = node_row['Anomalies Detected'].values[0]
        st.markdown(f"### Model Accuracy for Node_{node_index}: **{accuracy:.2%}**")

# Function to track new anomalies
def track_new_anomalies(node_index, column, current_anomalies):
    anomaly_key = f"node_{node_index}_column_{column}_anomalies"
    if anomaly_key not in st.session_state:
        st.session_state[anomaly_key] = set()
    current_anomaly_timestamps = set(current_anomalies['timestamp'].dropna().unique())
    new_anomalies = current_anomaly_timestamps - st.session_state[anomaly_key]
    st.session_state[anomaly_key] = current_anomaly_timestamps
    return new_anomalies

# Main Streamlit app
def app(node):
    st.title("One-Class SVM Anomaly Detection Dashboard")
    st.markdown("Analyze anomalies for each feature using One-Class SVM models trained for each column.")

    # Extract node index from the passed node
    node_index = int(node.split(" ")[1])  # Extract the node number
# Initialize session state variables
    if "refresh_pressed" not in st.session_state:
        st.session_state["refresh_pressed"] = False

    if "anomalies_summary" not in st.session_state:
        st.session_state["anomalies_summary"] = {}

    # Load data for the selected node
    data = load_data(node_index)
    if data.empty:
        st.error(f"No data available for Node_{node_index}")
        return

    # Process data
    data = preprocess_data(data)

    # Detect anomalies immediately when Refresh button is clicked
    if st.button("🔄 Refresh Data"):
        st.session_state["refresh_pressed"] = True
        anomalies_summary = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            model = load_column_model(node_index, column)
            if model is None:
                continue

            anomalies, anomaly_count = detect_column_anomalies(data, column, model)
            new_anomalies = track_new_anomalies(node_index, column, anomalies)
            if new_anomalies:
                anomalies_summary[column] = list(new_anomalies)

        # Update session state with new anomalies
        st.session_state["anomalies_summary"] = anomalies_summary

    # Display anomalies summary at the top
    if st.session_state["refresh_pressed"]:
        if st.session_state["anomalies_summary"]:
            st.markdown("<h3 style='color:red;'>🚨 Anomalies Detected</h3>", unsafe_allow_html=True)
            for column, timestamps in st.session_state["anomalies_summary"].items():
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid red;
                        background-color: rgba(255, 0, 0, 0.1);
                        padding: 10px;
                        border-radius: 5px;
                        margin-bottom: 10px;
                    ">
                        <strong>{column}:</strong> {', '.join(map(str, timestamps))}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<h3 style=' border: 1px solid green; background-color: rgba(0, 128, 0, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 10px; color:green;'>✅ No Anomalies Detected</h3>", unsafe_allow_html=True)
        st.session_state["refresh_pressed"] = False

    # Display accuracy for the selected node
    display_node_accuracy(node_index)

    # Iterate through each column to load its model and process anomalies
    for column in data.select_dtypes(include=[np.number]).columns:
        # Add a border for each feature section
        st.markdown("---")
        st.markdown(
            f"""
            <div style="border: 2px solid; padding: 10px; margin-top: 30px; border-radius: 10px;">
                <h3 style="color: #ffffff;">Anomaly Analysis for {column}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Load the column-specific model
        model = load_column_model(node_index, column)
        if model is None:
            continue

        # Detect anomalies for the column
        anomalies, anomaly_count = detect_column_anomalies(data, column, model)

        # Show alert if anomalies are detected
        if anomaly_count > 0:
            # Add space above the warning
            st.markdown("<br>", unsafe_allow_html=True)
            st.warning(f"⚠️ {anomaly_count} anomalies detected in {column}!")
            visualize_column_anomalies(data, column)
            display_column_anomalies_table(anomalies, column)
        else:
            st.success(f"No anomalies detected in {column}.")
