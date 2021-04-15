import streamlit as st
import pandas as pd
import requests
from pathlib import Path
import plotly.figure_factory as ff 
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.metrics import classification_report_imbalanced
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier

@st.cache
def load_data(nrows):
    data = Path('/Users/wesleysapone/James-Bond/streamlit/data/feature_df.csv')
    df = pd.read_csv(data, nrows=nrows)
    
    return df

@st.cache
def load_target(nrows):
    data = Path('/Users/wesleysapone/James-Bond/streamlit/data/.csv')
    target = pd.read_csv(data, nrows=nrows)
    target = target['rank_change_2'].values.reshape(-1,1)
    return target

@st.cache
def load_log_reg(nrows):
    data = Path("")
    log_importance_df = pd.read_csv("")


def load_model_df(nrows):
    path = Path("/Users/wesleysapone/James-Bond/streamlit/data/model_df.csv")
    model_df = pd.read_csv(path)
    
    return model_df

model_df = load_model_df(13498)

feature_df = load_data(13498) 
# sample_target = load_target(13498)

X = model_df.drop(columns="rank_diff")
y = model_df['rank_diff']

Counter(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=78)

# Creating the scaler instance
scaler = StandardScaler()

# Fitting the scaler
scaler.fit(X_train)

# Transforming the data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a random forest classifier
rf_model = RandomForestClassifier(n_estimators= (int(len(x_var_list)/3)), max_depth=3, max_samples = None, random_state=0)

# Fit the model
rf_model = rf_model.fit(X_train_scaled, y_train)

# Make a prediction of "y" values from the X_test dataset
predictions = rf_model.predict(X_test_scaled)
print(np.unique(predictions))
print(np.unique(y_test))


cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    cm, index=["Actual 1","Actual 2", "Actual 3"], 
               columns=["Predicted 1","Predicted 2", "Predicted 3"])


# Calculating the accuracy score
acc_score = accuracy_score(y_test, predictions)



# Displaying results
print("Confusion Matrix")
display(cm_df)
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(classification_report(y_test, predictions))

print("Balanced accuracy score: %.4f" % balanced_accuracy_score(y_test, predictions))


# Print the imbalanced classification report
print('Classification Report Imbalanced')
print(classification_report_imbalanced(y_test, predictions))


# Get the feature importance array
importances = rf_model.feature_importances_

# List the top 10 most important features
importances_sorted = sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)
importances_sorted[:10]

# Visualize the features by importance
importances_df = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns), reverse=True))
importances_df.set_index(importances_df[1], inplace=True)
importances_df.drop(columns=1, inplace=True)
importances_df.rename(columns={0: 'Feature Importances'}, inplace=True)
importances_sorted = importances_df.sort_values(by='Feature Importances')
importances_sorted.plot(kind='barh', color='lightgreen', title= 'Features Importances', legend=False)


# RANDOM OVERSAMPLING
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)
Counter(y_resampled)


# Create a random forest classifier
rf_ro_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)

# Fit the model
rf_ro_model = rf_ro_model.fit(X_resampled, y_resampled)

# Make a prediction of "y" values from the X_test dataset
predictions = rf_ro_model.predict(X_test_scaled)

# Display the confusion matrix
y_pred = rf_ro_model.predict(X_test_scaled)
confusion_matrix(y_test, y_pred)

balanced_accuracy_score(y_test, y_pred)


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))

# Get the feature importance array
importances = rf_ro_model.feature_importances_

# List the top 10 most important features
importances_sorted = sorted(zip(rf_ro_model.feature_importances_, X.columns), reverse=True)
importances_sorted[:10]

# Visualize the features by importance
importances_df = pd.DataFrame(sorted(zip(rf_ro_model.feature_importances_, X.columns), reverse=True))
importances_df.set_index(importances_df[1], inplace=True)
importances_df.drop(columns=1, inplace=True)
importances_df.rename(columns={0: 'Feature Importances'}, inplace=True)
importances_sorted = importances_df.sort_values(by='Feature Importances')
importances_sorted.plot(kind='barh', color='lightgreen', title= 'Features Importances', legend=False)


from sklearn.metrics import precision_recall_curve

probs_lr = rf_model.predict_proba(X_test)[:, 1]
probs_rf = rf_ro_model.predict_proba(X_test)[:, 1]
precision_lr, recall_lr, _ = precision_recall_curve(y_test, probs_lr, pos_label=1)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, probs_rf, pos_label=1)

plt.plot(recall_lr, precision_lr, marker='.')
plt.plot(recall_rf, precision_rf, marker='x')

# SMOTE OVERSAMPLING
X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy= {-1:10000, 0:10000, 1:10000}).fit_resample(
    X_train_scaled, y_train)

# Create a random forest classifier
rf_model_SM = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)

# Fit the model
rf_model_SM = rf_model_SM.fit(X_resampled, y_resampled)

# Make a prediction of "y" values from the X_test dataset
predictions = rf_model_SM.predict(X_test_scaled)


# Display the confusion matrix
y_pred = rf_model_SM.predict(X_test_scaled)
confusion_matrix(y_test, y_pred)

# Balanced accuracy score
balanced_accuracy_score(y_test, y_pred)

# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))


#SMOTEENN combination sampling
sm = SMOTEENN(random_state=1)
X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)
Counter(y_resampled)

# Create a random forest classifier
rf_model_SO = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)

# Fit the model
rf_model_SO = rf_model_SO.fit(X_resampled, y_resampled)

# Make a prediction of "y" values from the X_test dataset
predictions = rf_model_SO.predict(X_test_scaled)

# Display the confusion matrix
y_pred = rf_model_SO.predict(X_test_scaled)
confusion_matrix(y_test, y_pred)


# Balanced accuracy score
balanced_accuracy_score(y_test, y_pred)


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))













sidebar = st.sidebar.title("Panels")

select = st.sidebar.selectbox("Select",("James Bond", "Objectives", "Dataset", "Datasets Cleaning & Challenges", "Model Selection", "Model Evaluation", "Phase 2"), 3)



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
    Goals = ("""
        
        """)
    st.header("Goals")
    if st.checkbox("Show Project Goals"):
        st.subheader("Assigment from M")
        st.image("https://i2-prod.mirror.co.uk/incoming/article3114398.ece/ALTERNATES/s1200c/Judi-Dench-as-M.jpg")
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
    if st.checkbox("Summary of Credit Migration"):
        st.subheader("")
        st.write("""
            - A credit rating quantifies the creditworthiness of a borrower in general terms or with respect to a particular debt or financial obligation
            - A credit rating or score can be assigned to any entity that seeks to borrow money—an individual, a corporation, a state or provincial authority, or a sovereign government.
            """)

if select == 'Dataset':
    st.title("Dataset")

    st.write("""
    ## Sources
        - We were able to obtain access to SQL databases developed by a 
            company active in corporate credit default probability modelling
        - The databases comprised 41 datatables which we accessed through 
            Microsoft Azure to implement SQL queries and download csv files
    
    ### Key tables:
        - Core_us_fundamentals (676,000 records, ‘93-’21): 
            quarterly updated corporate financial information incl. financial statement data, 
            financial ratios, valuation ratios and sector info  across over 14,000 bond issuers 
        - Edi_bond_price (53m records, ‘09-’21):  
            daily updated bond price, volume, security terms, cusip codes, exchange
        - Ratings (280,552 records, ‘21-’21): 
            historical corporate credit ratings across over 280,000 issuers
        - Cleansed_fred_macro_daily (6,309 records, ‘96-’21): 
            daily readings across 32 macro variables incl. GDP, inflation, interest rates, 
            commodity pricing, vix etc.
    """)
if select == "Datasets Cleaning & Challenges":
    st.title("Data Scientists' Arch-nemesis")
    if st.checkbox("Antagonist"):
        st.header("Blofeld strikes: Dataset Combat")
        st.image("https://static3.srcdn.com/wordpress/wp-content/uploads/2021/04/Donald-Pleasance-as-Ernst-Blofeld.jpg?q=50&fit=crop&w=740&h=370&dpr=1.5")
        st.write("""
            ### Main Data Source : 
                - Company Fundamentals: Quarterly Data of 14000  since 1998 14000
                - Price & Coupon 
                - Credit Ratings
            ### Macro Economic Data: 
                - Key Interest Rate
                - Commodity price 
            ### Target Data: 
            ### Data Cleaning Challenges: 
                - Stale Bond Price,  
                - Not many common cusips between 
                - Usage of AsyncIO 

            """)

    data = Path('/Users/wesleysapone/James-Bond/streamlit/data/feature_df.csv')
    df = pd.read_csv(data)
    
    st.write(df)



if select == 'Model Selection':
    st.title("Technical Briefing with Q")
    st.image("https://carwow-uk-wp-2.imgix.net/2002-Aston-Martin-V12-Vanquish-scaled.jpg?auto=format&cs=tinysrgb&fit=clip&ixlib=rb-1.1.0&q=60&w=1920")
    st.write("""
        ### Logistic Regression 
            - Multi-Variable model used to predict binary outcomes 
        ### Random Forest 
            - 
        ### Gradient Boosting
            - 
    """)


if select == 'Model Evaluation':
    
    
    
    
    
    
    
    
    st.subheader("Economic")
    data = Path('/Users/wesleysapone/James-Bond/streamlit/data/feature_df.csv')
    df = pd.read_csv(data)
    
    st.write(df)

    importance = Path('/Users/wesleysapone/James-Bond/streamlit/data/gboost_importance.csv')
    boost_imp = pd.read_csv(importance)
    boost_imp.set_index('1', inplace=True)
    st.subheader("gradient boost feature importance sorted df")
    st.write(boost_imp)
    st.subheader("gradient boost feature importance bar chart")
    st.bar_chart(boost_imp['Feature Importances'])

    st.subheader("gboost top 3 feature importance concentrations")
    hst = pd.DataFrame(sample_features[:300], columns = ['pe', 'close_edi', 'liabilities'])
    hst.hist()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # log_importance = Path('/Users/wesleysapone/James-Bond/streamlit/data/logistic_importance.csv')
    # log_imp = pd.read_csv(log_importance)
    # log_imp.set_index('1', inplace=True)
    # st.write(log_imp)
    # st.bar_chart(log_imp['Feature Importances'])

    rf_importance = Path('/Users/wesleysapone/James-Bond/streamlit/data/smote_rf_importance_df.csv')
    rf_imp = pd.read_csv(rf_importance)
    rf_imp.set_index('1', inplace=True)
    st.subheader("random forest feature importance sorted df")
    st.write(rf_imp)
    st.subheader("random forest feature importance bar chart")
    st.bar_chart(rf_imp['Feature Importances'])

    st.subheader("random forest classification report")
    st.image("https://keep.google.com/u/0/media/v2/1rX15QEuZDqY9IXJgKrwJc2IVm68OP307Qe591B0s3tbB7qG2z9a2RUKmSab9K6U/1cDHQy4a5eQ_oPfumcK6ac16qdfRxMh68Wg-hCR6m24uv2vsaZEJIoIeRKVLYreA?accept=image/gif,image/jpeg,image/jpg,image/png,image/webp,audio/aac&sz=546")
    

    




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