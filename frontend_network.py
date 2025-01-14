import streamlit as st
import pandas as pd
import os
import plotly.graph_objs as go
import time 

# Base path for the dataset
BASE_PATH = "./FYP_Kubernetes/Feature_Engineered"

# Function to load data based on node selection
def load_data(node):
    file_path = os.path.join(BASE_PATH, f"node_node_{node}_feature_engineered.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "network_latency",
                "network_bandwidth_usage",
                "Network_to_CPU_Ratio",
                "Network_to_Memory_Ratio",
                "Network_to_Disk_Ratio",
                "Network_to_Temperature_Ratio",
            ]
        )  # Default columns

# Function to generate graph data with area charts
def generate_graph_data(data, column):
    if not data.empty and column in data:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=data["timestamp"],
                    y=data[column],
                    mode="lines",
                    fill="tozeroy",
                    line=dict(shape="spline", width=2, color="#007bff"),  # Blue color
                )
            ],
            layout=go.Layout(
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=dict(
                    title="Timestamp",
                    showgrid=False,
                    zeroline=False,
                    tickformat="%H:%M:%S",
                ),
                yaxis=dict(
                    title=column.replace("_", " ").capitalize(),
                    showgrid=True,
                    zeroline=False,
                ),
                paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                title=dict(
                    text=f"{column.replace('_', ' ').capitalize()} Over Time",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=18, color="#ffffff"),
                ),
            ),
        )
        return fig
    return go.Figure()

# App function for CPU page
def app(node):
    st.title("Network Usage Overview")
    st.write(f"Currently displaying data for: **{node}**")

    # Load data for the selected node
    node_index = int(node.split(" ")[1])  # Extract the node number
    data = load_data(node_index)

    # Function to display graphs and tables
    def display_graph_and_table(column_name):
        st.markdown(f"#### {column_name.replace('_', ' ').capitalize()}")
        fig = generate_graph_data(data, column_name)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"#### Data Table: {column_name.replace('_', ' ').capitalize()}")

        # Format data: Remove index and increase decimal precision
        formatted_data = data[["timestamp", column_name]].dropna().reset_index(drop=True)
        formatted_data[column_name] = formatted_data[column_name].apply(lambda x: f"{x:.20f}")
        st.dataframe(formatted_data, use_container_width=True)

    # Display for each metric
    metrics = [
            "network_latency",
            "network_bandwidth_usage",
            "Network_to_CPU_Ratio",
            "Network_to_Memory_Ratio",
            "Network_to_Disk_Ratio",
            "Network_to_Temperature_Ratio",
        ]

    for metric in metrics:
        st.markdown("---")
        display_graph_and_table(metric)


    time.sleep(1)  # Refresh interval in seconds
    st.experimental_rerun()