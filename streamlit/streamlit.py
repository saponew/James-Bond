import streamlit as st
import pandas as pd
import requests
from pathlib import Path
import plotly.figure_factory as ff 
import matplotlib.pyplot as plt

@st.cache
def load_data(nrows):
    data = Path('/Users/wesleysapone/James-Bond/streamlit/data/feature_df.csv')
    df = pd.read_csv(data, nrows=nrows)
    
    return df

@st.cache
def load_target(nrows):
    data = Path('/Users/wesleysapone/James-Bond/streamlit/data/xgboost_model_df.csv')
    target = pd.read_csv(data, nrows=nrows)
    target = target['rank_change'].values.reshape(-1,1)
    return target

sample_features = load_data(13000) 
# sample_target = load_target(3000)

sidebar = st.sidebar.title("Panels")

select = st.sidebar.selectbox("Select",("James Bond", "Objectives", "Dataset", "Model Selection", "Model Evaluation", "Phase 2"), 4)



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

    # st.header("corporate bond rating change prediction")
    st.write("""
    
    - Develop a machine learning model that predicts a change in corporate credit ratings over future 12 months for a given corporate bond
    - Stake holders for this model includes bond investors and risk professionals in charge of bond collateral
    - We experimented with a number of classification models incl. logistic regression, random forest and GradientBoost
    - To train our model we obtained  a number of large datasets with corporate financial data, credit ratings, bond/equity pricing and macro data
    - We refined our feature set by selecting relevant features across various datasets and merging datables
    - We also  experimented with a variety of rating target variables and ultimately chose 
    - Future improvements include enhancing our datasets, model refinement and  optimizing target variables

    """)

    st.header("Credit Migration")
    st.write("""
    - A credit rating quantifies the creditworthiness of a borrower in general terms or with respect to a particular debt or financial obligation
    - A credit rating or score can be assigned to any entity that seeks to borrow money—an individual, a corporation, a state or provincial authority, or a sovereign government.
""")

if select == 'Dataset':
    st.title("Dataset")

    st.write("""
    ## Data Sources
    
    """)
    st.write("""
    ## challenges wrangling our datasets
    - CORE_US
    - EDI_BOND_PRICE
    - Ratings
    - Data Cleaning
    - Asyncio
    
    
    """)

    data = Path('/Users/wesleysapone/James-Bond/streamlit/data/feature_df.csv')
    df = pd.read_csv(data)
    
    st.write(df)



if select == 'Model Selection':
    st.subheader("Logic behind model choices")
    # st.
    
    st.write(
        """
        ### 
    
    
        """
    )


if select == 'Model Evaluation':
    st.subheader("Economic")
    data = Path('/Users/wesleysapone/James-Bond/streamlit/data/feature_df.csv')
    df = pd.read_csv(data)
    
    st.write(df)

    importance = Path('/Users/wesleysapone/James-Bond/streamlit/data/gboost_importance.csv')
    boost_imp = pd.read_csv(importance)
    boost_imp.set_index('1', inplace=True)
    st.write(boost_imp)
    st.bar_chart(boost_imp['Feature Importances'])

    hst = pd.DataFrame(sample_features[:300], columns = ['pe', 'close_edi', 'liabilities'])
    hst.hist()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)




# # Calculating the confusion matrix
# cm = confusion_matrix(y_test, predictions)
# cm_df = pd.DataFrame(
#     cm, index=["Actual 1","Actual 2", "Actual 3"], 
#                columns=["Predicted 1","Predicted 2", "Predicted 3"])

# acc_score = accuracy_score(y_test, predictions)

# # Displaying results
# print("Confusion Matrix")
# display(cm_df)
# print(f"Accuracy Score : {acc_score}")
# print("Classification Report")
# print(classification_report(y_test, predictions))













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