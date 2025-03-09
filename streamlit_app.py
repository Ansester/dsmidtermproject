import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

def load_data():
    file_path = "crimedata.csv"  # Update path if necessary
    return pd.read_csv(file_path)

df = load_data()

def data_exploration():
    # Page Title
    st.title("Regression Analysis of Crime Incidents and Societal Trends")
    
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
    df2 = df[["medIncome","racepctblack","racePctWhite","racePctAsian","racePctHisp","PctRecImmig10","ViolentCrimesPerPop"]]
    df2 = df2.dropna(axis=0,how='any')
    X = df2[["medIncome","racepctblack","racePctWhite","racePctAsian","racePctHisp","PctRecImmig10"]]
    y = df2["ViolentCrimesPerPop"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    prediction = lr.predict(X_test)
    from sklearn import metrics
    st.write(metrics.mean_absolute_error(y_test,prediction))

    val1 = st.slider("Select Income", min_value=5000, max_value=300000, value=30000)
    
    max_val = 100
    total = 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val2 = st.number_input("% Black", min_value=0, max_value=max_val - total, value=25)
        total += val2
    with col2:
        val3 = st.number_input("% White", min_value=0, max_value=max_val - total, value=25)
        total += val3
    with col3:
        val4 = st.number_input("% Asian", min_value=0, max_value=max_val - total, value=25)
        total += val4
    with col4:
        val5 = st.number_input("% Hispanic", min_value=0, max_value=max_val - total, value=25)
    
    val6 = st.slider("Select Percentage of Immigrants", min_value=0, max_value=100, value=10)
    
    # Creating a DataFrame row
    input = pd.DataFrame({
        "medIncome": [val1],
        "racepctblack": [val2],
        "racePctWhite": [val3],
        "racePctAsian": [val4],
        "racePctHisp": [val5],
        "PctRecImmig10": [val6],
    })

    output = lr.predict(input)

    st.write(output)

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
