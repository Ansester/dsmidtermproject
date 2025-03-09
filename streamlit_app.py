import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

def data_exploration():
    # Page Title
    st.title("Crime Data Insights")
    
    # Introduction
    st.markdown(
        """
        ## Why This Data is Important
        Understanding crime statistics is essential for law enforcement, policymakers, and researchers. 
        This dataset provides insights into crime patterns across various communities, allowing for data-driven decision-making.
        
        ### What Can Be Achieved with This Data?
        - **Crime Rate Analysis**: Identify high-crime areas and trends over time.
        - **Predictive Modeling**: Use machine learning to predict crime rates based on demographic and socioeconomic factors.
        - **Policy Recommendations**: Help authorities allocate resources more effectively.
        """
    )
    
    # Basic Dataset Information
    st.subheader("Dataset Overview")
    
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    st.subheader("Column Data Types")
    st.write(df.dtypes)
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    # Allow users to explore the first few rows
    st.subheader("Preview the Data")
    st.write(df.head())
    
    # Allow users to select columns for visualization
    columns = st.multiselect("Select columns to visualize", df.columns)
    if columns:
        st.write(df[columns].describe())


def linear_regression():
    st.title("Linear Regression")
    st.write("Perform linear regression on your dataset.")
    st.write("(Feature to be implemented based on dataset)")

def data_visualization():
    st.title("Data Visualization")
    import streamlit.components.v1 as components

    # Replace with your actual Looker dashboard URL
    look_dashboard_url = "https://lookerstudio.google.com/embed/reporting/6889d02f-1069-4428-8a08-0b736a175046/page/qOc7E"
    # Embed Looker dashboard using iframe
    components.iframe(look_dashboard_url, height=600)


def conclusions():
    st.title("Conclusions")
    st.write("Summarize the insights from your analysis.")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu("Main Menu", ["Data Exploration", "Linear Regression", "Data Visualization", "Conclusions"],
                           icons=["database", "graph-up", "bar-chart", "clipboard-check"],
                           menu_icon="menu-button-wide", default_index=0)

# Page Routing
if selected == "Data Exploration":
    data_exploration()
elif selected == "Linear Regression":
    linear_regression()
elif selected == "Data Visualization":
    data_visualization()
elif selected == "Conclusions":
    conclusions()
