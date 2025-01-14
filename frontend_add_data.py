import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib

def add_data():
    # Initialize an empty list to store input data
    data = []

    # User Inputs
    st.title("Node Monitoring and Metrics Input")

    st.subheader("Manually Add Data")

    node_name = st.number_input("Node Name", min_value=0, max_value=50, step=1)

    # Date and time inputs
    date_input = st.date_input("Select Date")
    time_input = st.time_input("Select Time")

    # Combine date and time into a single timestamp
    timestamp = datetime.combine(date_input, time_input).strftime("%Y-%m-%d %H:%M:%S")

    namespace = st.selectbox("Namespace", ["dev", "default", "kube-system", "prod"])
    cpu_allocation_efficiency = st.number_input("CPU Allocation Efficiency", format="%.4f")
    memory_allocation_efficiency = st.number_input("Memory Allocation Efficiency", format="%.4f")
    disk_io = st.number_input("Disk I/O", format="%.4f")
    network_latency = st.number_input("Network Latency", format="%.4f")
    node_temperature = st.number_input("Node Temperature", format="%.4f")
    node_cpu_usage = st.number_input("Node CPU Usage", format="%.4f")
    node_memory_usage = st.number_input("Node Memory Usage", format="%.4f")
    event_type = st.selectbox("Event Type", ["Warning", "Error", "Normal"])
    event_message = st.selectbox("Event Message", ["Killed", "Failed", "Completed", "OOMKilled", "Started"])
    scaling_event = st.selectbox("Scaling Event", ["TRUE", "FALSE"])
    pod_lifetime_seconds = st.number_input("Pod Lifetime Seconds", format="%.4f")
    cpu_request = st.number_input("CPU Request", format="%.4f")
    cpu_limit = st.number_input("CPU Limit", format="%.4f")
    memory_request = st.number_input("Memory Request", format="%.4f")
    memory_limit = st.number_input("Memory Limit", format="%.4f")
    cpu_usage = st.number_input("CPU Usage", format="%.4f")
    memory_usage = st.number_input("Memory Usage", format="%.4f")
    pod_status = st.selectbox("Pod Status", ["Running", "Pending", "Succeeded", "Failed", "Unknown"])
    restart_count = st.number_input("Restart Count", min_value=0, step=1)
    uptime_seconds = st.number_input("Uptime Seconds", format="%.4f")
    deployment_strategy = st.selectbox("Deployment Strategy", ["RollingUpdate", "Recreate"])
    scaling_policy = st.selectbox("Scaling Policy", ["Auto", "Manual"])
    network_bandwidth_usage = st.number_input("Network Bandwidth Usage", format="%.4f")

    # Add button to submit and save the manually entered data
    if st.button("Submit"):
        # Store the input data as a dictionary
        record = {
            "timestamp": timestamp,
            "cpu_allocation_efficiency": cpu_allocation_efficiency,
            "memory_allocation_efficiency": memory_allocation_efficiency,
            "disk_io": disk_io,
            "network_latency": network_latency,
            "node_temperature": node_temperature,
            "node_cpu_usage": node_cpu_usage,
            "node_memory_usage": node_memory_usage,
            "cpu_request": cpu_request,
            "cpu_limit": cpu_limit,
            "memory_request": memory_request,
            "memory_limit": memory_limit,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "node_name": f"node_{node_name}",  # Format node name
            "network_bandwidth_usage": network_bandwidth_usage,
            "pod_status": pod_status,
        }
        # Append the record to the data list
        data.append(record)
        st.success("Data saved successfully!")

    # Display the data in a DataFrame
    if data:
        df = pd.DataFrame(data)
        
        # Data cleaning
        df["timestamp"] = pd.to_datetime(df["timestamp"])  # Convert timestamp to datetime format
        
        # Load scalers
        scaler_columns = [
            "disk_io", "network_latency", "node_temperature", "node_cpu_usage",
            "node_memory_usage", "cpu_request", "cpu_limit", "memory_request",
            "memory_limit", "cpu_usage", "memory_usage", "network_bandwidth_usage"
        ]
        for column in scaler_columns:
            scaler_path = f"./FYP_Kubernetes/Scalers/{column}_scaler.pkl"
            scaler = joblib.load(scaler_path)
            df[column] = scaler.transform(df[[column]])

        # One-hot encoding for pod_status with all possible categories
        pod_status_categories = ["Running", "Pending", "Succeeded", "Failed", "Unknown"]
        df = pd.get_dummies(df, columns=["pod_status"], prefix="pod_status")
        for category in pod_status_categories:
            column_name = f"pod_status_{category}"
            if column_name not in df.columns:
                df[column_name] = 0

        # Save the cleaned data to the respective node files
        for node in df["node_name"].unique():
            node_data = df[df["node_name"] == node]
            file_path = f"./FYP_Kubernetes/cleaned_dataset/node_{node}_dataset.csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if os.path.exists(file_path):
                # Append to the existing file
                existing_data = pd.read_csv(file_path)
                node_data = pd.concat([existing_data, node_data], ignore_index=True)
            node_data.to_csv(file_path, index=False)
        
        st.write("Cleaned Data:")
        st.dataframe(df)

        # Feature Engineering (Newly Added)
        feature_engineered_data = pd.DataFrame(data)

        node_cpu_usage_col = "cpu_usage"  # Replace with actual column name
        node_memory_usage_col = "memory_usage"  # Replace with actual column name
        network_bandwidth_usage_col = "network_bandwidth_usage"  # Replace with actual column name
        node_temperature_col = "node_temperature"  # Replace with actual column name

        # Check for existing feature-engineered dataset for deltas
        for node in feature_engineered_data["node_name"].unique():
            file_path = f"./Dataset/Feature_Engineered/node_{node}_feature_engineered.csv"
            if os.path.exists(file_path):
                existing_data = pd.read_csv(file_path)
                last_row = existing_data.iloc[[-1]]  # Get the last row
                st.write(f"Last Row of Node {node_name}")
                st.dataframe(last_row)
                node_cpu_usage_last=last_row[node_cpu_usage_col].values[0]
                node_memory_usage_last=last_row[node_memory_usage_col].values[0]
                node_network_bandwidth_usage_last=last_row[network_bandwidth_usage_col].values[0]
                node_node_temperature_last=last_row[node_temperature_col].values[0]
                    

        # Perform Feature Engineering
        feature_engineered_data['cpu_utilization_rate'] = feature_engineered_data['cpu_usage'] / feature_engineered_data['cpu_limit']
        feature_engineered_data['memory_utilization_rate'] = feature_engineered_data['memory_usage'] / feature_engineered_data['memory_limit']
        feature_engineered_data['cpu_request_utilization_rate'] = feature_engineered_data['cpu_usage'] / feature_engineered_data['cpu_request']
        feature_engineered_data['memory_request_utilization_rate'] = feature_engineered_data['memory_usage'] / feature_engineered_data['memory_request']
        feature_engineered_data['node_cpu_usage_delta'] = feature_engineered_data['node_cpu_usage'] - node_cpu_usage_last
        feature_engineered_data['node_memory_usage_delta'] = feature_engineered_data['node_memory_usage'] - node_memory_usage_last
        feature_engineered_data['network_bandwidth_usage_delta'] = feature_engineered_data['network_bandwidth_usage'] - node_network_bandwidth_usage_last
        feature_engineered_data['node_temperature_delta'] = feature_engineered_data['node_temperature'] - node_node_temperature_last

        # Fill NaN values with 0
        feature_engineered_data = feature_engineered_data.fillna(0)

        # Add Resource Ratios
        feature_engineered_data['CPU_to_Memory_Ratio'] = feature_engineered_data['cpu_usage'] / (feature_engineered_data['memory_usage'] + 1e-5)
        feature_engineered_data['CPU_to_Disk_Ratio'] = feature_engineered_data['cpu_usage'] / (feature_engineered_data['disk_io'] + 1e-5)
        feature_engineered_data['CPU_to_Network_Ratio'] = feature_engineered_data['cpu_usage'] / (feature_engineered_data['network_bandwidth_usage'] + 1e-5)
        feature_engineered_data['CPU_to_Temperature_Ratio'] = feature_engineered_data['cpu_usage'] / (feature_engineered_data['node_temperature'] + 1e-5)

        feature_engineered_data['Memory_to_CPU_Ratio'] = feature_engineered_data['memory_usage'] / (feature_engineered_data['cpu_usage'] + 1e-5)
        feature_engineered_data['Memory_to_Disk_Ratio'] = feature_engineered_data['memory_usage'] / (feature_engineered_data['disk_io'] + 1e-5)
        feature_engineered_data['Memory_to_Network_Ratio'] = feature_engineered_data['memory_usage'] / (feature_engineered_data['network_bandwidth_usage'] + 1e-5)
        feature_engineered_data['Memory_to_Temperature_Ratio'] = feature_engineered_data['memory_usage'] / (feature_engineered_data['node_temperature'] + 1e-5)

        feature_engineered_data['Disk_to_CPU_Ratio'] = feature_engineered_data['disk_io'] / (feature_engineered_data['cpu_usage'] + 1e-5)
        feature_engineered_data['Disk_to_Memory_Ratio'] = feature_engineered_data['disk_io'] / (feature_engineered_data['memory_usage'] + 1e-5)
        feature_engineered_data['Disk_to_Network_Ratio'] = feature_engineered_data['disk_io'] / (feature_engineered_data['network_bandwidth_usage'] + 1e-5)
        feature_engineered_data['Disk_to_Temperature_Ratio'] = feature_engineered_data['disk_io'] / (feature_engineered_data['node_temperature'] + 1e-5)

        feature_engineered_data['Network_to_CPU_Ratio'] = feature_engineered_data['network_bandwidth_usage'] / (feature_engineered_data['cpu_usage'] + 1e-5)
        feature_engineered_data['Network_to_Memory_Ratio'] = feature_engineered_data['network_bandwidth_usage'] / (feature_engineered_data['memory_usage'] + 1e-5)
        feature_engineered_data['Network_to_Disk_Ratio'] = feature_engineered_data['network_bandwidth_usage'] / (feature_engineered_data['disk_io'] + 1e-5)
        feature_engineered_data['Network_to_Temperature_Ratio'] = feature_engineered_data['network_bandwidth_usage'] / (feature_engineered_data['node_temperature'] + 1e-5)

        feature_engineered_data['Temperature_to_CPU_Ratio'] = feature_engineered_data['node_temperature'] / (feature_engineered_data['cpu_usage'] + 1e-5)
        feature_engineered_data['Temperature_to_Memory_Ratio'] = feature_engineered_data['node_temperature'] / (feature_engineered_data['memory_usage'] + 1e-5)
        feature_engineered_data['Temperature_to_Disk_Ratio'] = feature_engineered_data['node_temperature'] / (feature_engineered_data['disk_io'] + 1e-5)
        feature_engineered_data['Temperature_to_Network_Ratio'] = feature_engineered_data['node_temperature'] / (feature_engineered_data['network_bandwidth_usage'] + 1e-5)
        
        # Peak Detection
        feature_engineered_data['cpu_usage_peak'] = (feature_engineered_data['cpu_usage'] > 0.9 * feature_engineered_data['cpu_limit']).astype(int)
        feature_engineered_data['memory_usage_peak'] = (feature_engineered_data['memory_usage'] > 0.9 * feature_engineered_data['memory_limit']).astype(int)

        # Drop `node_name` and `pod_status` before saving
        feature_engineered_data = feature_engineered_data.drop(columns=["pod_status"], errors="ignore")

        # Save Feature-Engineered Data
        for node in feature_engineered_data["node_name"].unique():
            node_data = feature_engineered_data[feature_engineered_data["node_name"] == node]
            file_path = f"./FYP_Kubernetes/Feature_Engineered/node_{node}_feature_engineered.csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if os.path.exists(file_path):
                # Append to existing file
                existing_data = pd.read_csv(file_path)
                node_data = pd.concat([existing_data, node_data], ignore_index=True)
            node_data.to_csv(file_path, index=False)
        
        st.write("Feature-Engineered Data:")
        st.dataframe(feature_engineered_data)