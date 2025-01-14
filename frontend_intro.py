import streamlit as st
import pandas as pd

def app () : 

    st.title("Enhancing Fault Detection in Kubernetes Node Deployments Through Anomaly Detection and Predictive Maintenance")
    st.write("**By Prof. Dr. Teh Ying Wah and Gan Zi Xiang**")
    st.markdown("---")



    st.header("Introduction")
    st.write("""
    An open-source framework for containerisation, Docker facilitates the creation, deployment, and administration of containers. Developers can use Docker to package and run software in containers—loosely segregated environments—alongside their dependencies. 

    The open-source container orchestration platform Kubernetes, also referred to as K8s, automates the deployment, scaling, and management of containerised applications. Regardless of their underlying infrastructure, users can effortlessly manage sophisticated containerised apps and services with Kubernetes.

    Due to the extremely dynamic and resource-intensive nature of the environment, traditional monitoring techniques frequently overlook early indicators of instability in Kubernetes nodes. The complexity of identifying and resolving problems is increased by frequent scaling, varied workloads, and changing configurations. This drawback emphasizes the necessity of sophisticated, data-driven strategies to proactively spot irregularities and anticipate malfunctions before they become serious incidents.

    This project focuses on Pod-level monitoring through anomaly detection and predictive maintenance, particularly for managing performance and reliability in Dockerized deployments. Node-level monitoring emphasizes overall resource health, availability, and performance across the nodes that host containerized applications.

    The project aims to create models that can detect anomalous behaviours and predict possible failures by concentrating on node-level indicators including CPU utilisation, memory consumption, disk I/O, and network speed. Anomaly detection algorithms identify odd patterns suggestive of new difficulties, while predictive maintenance uses historical data to predict problems and enable preventative interventions.

    Expected outcomes include improved fault identification, reduced downtime, and an extensive dashboard for real-time monitoring.
    """)

    st.image("./FYP_Kubernetes/Kubernetes-Logo-500x281.png")
    st.markdown("---")

    # Data Source Section
    st.header("Data Source")
    st.write("""
    - **Kubernetes Metrics and Logs**: Includes CPU usage, memory usage, disk I/O, and network traffic data from Kubernetes nodes and pods. 
      - [Dataset Link](https://www.kaggle.com/datasets/nickkinyae/kubernetes-resource-and-performancemetricsallocation)
    """)

    data1=pd.read_csv("./FYP_Kubernetes/Dataset/kubernetes_performance_metrics_dataset.csv")
    data2=pd.read_csv("./FYP_Kubernetes/Dataset/kubernetes_resource_allocation_dataset.csv")

    st.write("Performance Metrics Dataset")
    st.write(data1.head())

    st.write("Resource Allocation Dataset")
    st.write(data2.head())
    
    st.markdown("---")

    # Modelling Section
    st.header("Modelling")
    st.subheader("Anomaly Detection")
    st.write("""
    1. **Isolation Forest**: Highest Accuracy: 0.9661  
    2. **One Class SVM**: Highest Accuracy: 0.9836  
    3. **Autoencoder**: Highest Accuracy: 0.9508  
    """)

    st.subheader("Predictive Maintenance")
    st.write("""
    4. **LSTM**  
    5. **ARIMA**  
    """)
    st.markdown("---")

    # Highlights Section
    st.header("Highlights of the Project")
    st.write("1. Real-time Anomalies Detection ")
    st.image("./FYP_Kubernetes/anomalies.png", caption="Visualization of Real-time Anomalies Detection")

    # Highlight 2
    st.write("2. Prediction of Pod Status")
    st.image("./FYP_Kubernetes/pod_predict.png", caption="Prediction of Pod Status Over Time")  # Replace with your image path or URL

    # Highlight 3
    st.write("3. Visualization of real-time node health")
    st.image("./FYP_Kubernetes/Visualisation.png", caption="Real-time Node Health Visualization")  # Replace with your image path or URL

    st.markdown("---")
    st.write("This project combines cutting-edge anomaly detection and predictive maintenance strategies to enhance fault detection and ensure the reliable operation of Kubernetes-based deployments.")