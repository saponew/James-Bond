import streamlit as st
import pandas as pd
import plotly as plt


st.title("Predicting Corporate Bond Default")
st.header("Dataset Intro")
st.write("How'd you get allat data?!")

add_selectbox = st.sidebar.selectbox(
    "What would you like to know?"
    ('Data Cleaning', 'Feature Engineering', 'Model Training')
)

add_slider = st.sidebar.slider(
    "Select a range of values",
    0.0, 100.0, (25.0, 75.0)
)

with add_selectbox: 
    left_column, right_column = st.beta_columns(2)
    left_column.button