import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objs as go

# Base directories
SCALER_DIR = './FYP_Kubernetes/Scalers/'
MODEL_DIR = './FYP_Kubernetes/LSTM_Model/'

# Features and statuses
FEATURES = [
    "cpu_allocation_efficiency", "memory_allocation_efficiency", "disk_io",
    "network_latency", "node_temperature", "node_cpu_usage", "node_memory_usage",
    "cpu_request", "cpu_limit", "memory_request", "memory_limit",
    "cpu_usage", "memory_usage", "network_bandwidth_usage"
]
STATUSES = ["Running", "Pending", "Succeeded", "Failed", "Unknown"]

# Function to load scaler for a feature
def load_scaler(feature):
    scaler_path = os.path.join(SCALER_DIR, f"{feature}_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            return pickle.load(f)

# Function to load LSTM model for a node
def load_lstm_model(node_index):
    model_path = os.path.join(MODEL_DIR, f"LSTM_Model_Node_{node_index}.h5")
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error(f"LSTM model not found for Node {node_index}. Please train the model first.")
        return None

# Function to scale input features
def scale_features(input_features, scalers):
    scaled_features = []
    for feature, value in input_features.items():
        scaler = scalers.get(feature)
        if scaler:
            scaled_value = scaler.transform([[value]])[0][0]
            scaled_features.append(scaled_value)
        else:
            scaled_features.append(value)
    return np.array(scaled_features).reshape(1, 1, -1)  # Reshape for LSTM input

# Function to display probabilities in a table
def display_probabilities_table(probabilities):
    probabilities_df = pd.DataFrame(
        probabilities, index=STATUSES, columns=["Probability"]
    )
    probabilities_df["Probability"] = probabilities_df["Probability"].apply(lambda x: f"{x:.4f}")
    st.markdown("### Predicted Probabilities")
    st.dataframe(probabilities_df)

# Function to visualize probabilities as a bar chart
def plot_probabilities_bar_chart(probabilities):
    fig = go.Figure(data=[
        go.Bar(x=STATUSES, y=probabilities, marker=dict(color="skyblue"))
    ])
    fig.update_layout(
        title="Predicted Probabilities for Pod Statuses",
        xaxis_title="Pod Status",
        yaxis_title="Probability",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# Main Streamlit app
def app(node):
    st.title("LSTM Pod Status Prediction Dashboard")
    st.markdown("Enter the feature values below or upload a CSV file to predict the probabilities of pod statuses.")

    # Extract node index from the passed node
    node_index = int(node.split(" ")[1])  # Extract the node number

    # Load LSTM model for the selected node
    lstm_model = load_lstm_model(node_index)
    if lstm_model is None:
        return

    # Load scalers for scaling features
    scalers = {feature: load_scaler(feature) for feature in FEATURES}

    # Option to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file with the required feature columns", type="csv")

    if uploaded_file:
        try:
            # Load the uploaded CSV
            data = pd.read_csv(uploaded_file)

            # Ensure the required features are present in the file
            missing_features = [feature for feature in FEATURES if feature not in data.columns]
            if missing_features:
                st.error(f"The uploaded file is missing the following required features: {', '.join(missing_features)}")
            else:
                # Scale and predict for each row
                scaled_data = np.array([
                    scale_features(row, scalers)[0] for _, row in data[FEATURES].iterrows()
                ])
                predictions = lstm_model.predict(scaled_data)

                # Display predictions for all rows
                predictions_df = pd.DataFrame(predictions*100, columns=STATUSES)
                st.markdown("### Predictions for Uploaded CSV File")
                st.dataframe(predictions_df,use_container_width=True)

                # Visualize average probabilities as a bar chart
                average_probabilities = predictions.mean(axis=0)
                st.markdown("### Average Predicted Probabilities")
                plot_probabilities_bar_chart(average_probabilities*100)

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
    else:
        # Manual input for features
        st.markdown("### Input Feature Values Manually")
        input_features = {}
        for feature in FEATURES:
            input_features[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")

        # Predict probabilities when the user clicks the button
        if st.button("Predict Pod Status"):
            # Scale the input features
            scaled_features = scale_features(input_features, scalers)

            # Predict probabilities using the LSTM model
            probabilities = lstm_model.predict(scaled_features)[0]

            # Display the probabilities in a table
            display_probabilities_table(probabilities)

            # Visualize the probabilities as a bar chart
            plot_probabilities_bar_chart(probabilities)

# Run the app
if __name__ == "__main__":
    selected_node = st.sidebar.selectbox("Select Node", [f"Node {i}" for i in range(50)])
    app(selected_node)