import streamlit as st
import pandas as pd
import plotly as plt
import requests
from pathlib import Path

# st.title("James Bond")
# st.image('https://images.hindustantimes.com/rf/image_size_630x354/HT/p2/2020/06/08/Pictures/_d670cd18-a96f-11ea-9c49-07241376e8f9.jpg')
# st.header("Objective")
# st.text(
#     "Corporate Credit rating prediction model, pure default to rating (change), ")










# st.header("Dataset Intro")
# st.text("Kwame working w/ a company that is tyring to predict corporate bond default.  Our take on this project took a couple twists and turns along the way.")  
# st.text("Our initial goal was to predict whether a bond would default or not.  Once we realized that this would be much more involved than a 2 week project would allow.")
# st.text("The company was kind enough to give us partial access to some of the data and features they used as features in their")





# st.write(pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# }))

sidebar = st.sidebar.title("Panels")

select = st.sidebar.selectbox("Select",("James Bond", "Objectives", "Dataset", "Model Selection", "Model Evaluation", "Phase 2"))


    

if select == 'James Bond':
    st.title("James Bond")
    st.header("Corporate Bond Rating Prediction Model")
    st.image('https://images.hindustantimes.com/rf/image_size_630x354/HT/p2/2020/06/08/Pictures/_d670cd18-a96f-11ea-9c49-07241376e8f9.jpg')

    st.write("""
     
    ### Dataset Intro
    - Kwame working w/ a company that is tyring to predict corporate bond default.  Our take on this project took a couple twists and turns along the way.
    
    - Our initial goal was to predict whether a bond would default or not.  Once we realized that this would be much more involved than a 2 week project would allow.
    
    - The company was kind enough to give us partial access to some of the data that they pull into their models.  



    """)
    
if select == 'Objectives':
    st.title("Objectives")

    st.header("Data Collection & Cleaning")
    st.write("""
    ### 
    - Corporate credit rating prediction model
    - Changed from pure default to ratings 
    - Background corporate credit ratings
    - Who’s the stakeholder?



    """)

if select == 'Dataset':
    st.subheader("Data Collection & Cleaning")
    st.write("""
    ### Received assignment from M
    


    """)
    data = Path('/Users/wesleysapone/Desktop/streamlit/data/xgboost_model_df.csv')
    df = pd.read_csv(data)

    st.write(df)

if select == 'Model Selection':
    st.subheader("Logic behind model choices")
    st.write(
        """
        ### 
    
    
        """
    )


if select == 'Model Evaluation':
    st.subheader("Economic")
    st.write(
        """
        ### 
        
        
        
        """
    )

st.header(select)
if select == 'Phase 2':
    pass 
    r = requests.get("https://docs.google.com/presentation/d/1sEb5d1Mnd54tOAWrbJ122fDnsRDyVLqET8sxhIAlT9Q/edit#slide=id.gc6f9e470d_0_5")

# add_selectbox = st.sidebar.selectbox(
#     "What would you like to know?"
#     ('Data Cleaning', 'Feature Engineering', 'Model Training')
# )

# add_slider = st.sidebar.slider(
#     "Select a range of values",
#     0.0, 100.0, (25.0, 75.0)
# )

# with add_selectbox: 
#     left_column, right_column = st.beta_columns(2)
#     left_column.button