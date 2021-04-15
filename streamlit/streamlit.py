import streamlit as st
import pandas as pd
import requests
from pathlib import Path
import plotly.figure_factory as ff 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

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

sidebar = st.sidebar.title("Panels")

select = st.sidebar.selectbox("Select",("Deck", "James Bond", "Objectives", "Dataset", "Datasets Cleaning & Challenges", "Model Selection", "Model Evaluation", "Phase 2", "Random Oversampling - Random Forest", "Random Forest Code", "Gradient Boost Code"), 0)
if select == 'Deck':
    
    st.subheader("James Bond")
    # st.image("/Users/wesleysapone/James-Bond/streamlit/images/intro_deck.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide1.png")
    st.subheader("DATA")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide2.png")
    st.subheader("Credit Rating & Migration")
    st.image("https://www.glynholton.com/wp-content/uploads/2013/06/exhibit_default_model_1_v2.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide3.png")
    
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide4.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide5.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide6.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide7.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide8.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide9.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide10.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide11.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide12.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide13.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide14.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide15.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide16.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide17.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide18.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide19.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide20.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide21.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide22.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/slide23.png")



if select == 'James Bond':
    st.title("James Bond")
    st.image('https://images.hindustantimes.com/rf/image_size_630x354/HT/p2/2020/06/08/Pictures/_d670cd18-a96f-11ea-9c49-07241376e8f9.jpg')
    st.header("Forecasting a change in corporate bond ratings using machine learning")
    
    
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/bal_acc_score_gradient_boost.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/bal_acc_score_plain_jane_rf.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/accuracy_raport.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/bal_over_sample_clf.png")
    
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/bal_smote_clf.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/clf_pretty_report.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/cm_rf_rank_diff.png")
    
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/default_param_gb_feat_imp.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/gb_plain_jane_conf_mat_clf_report.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/imb_over_samp.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/imb_param_adj.png")

    st.image("/Users/wesleysapone/James-Bond/streamlit/images/imbalanced_clf_rpt.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/lear_rate_simplegboost.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/learning_rate_gboost_report.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/plain_jane_rf_feature_imp.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/ran_os_clf_report.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/random_over_samp_bal_accuracy_score.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/rf_ro_feat_imp.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/rf_ro_recall_plot.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/smote_imb_clf_rpt.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/smote_rf_bal_acc_score.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/smoteen_bal_acc_score.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/smoteen_imb_clf_rpt.png")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/smoteen_rf_clf.png")
    st.subheader("target count")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/target_count.png")
    
    st.subheader("deadly breif case")
    st.image("https://cdn.pastemagazine.com/www/blogs/lists/2012/10/05/to-russia-with-love-briefcase.jpg")
    st.image("/Users/wesleysapone/James-Bond/streamlit/images/deadly_breif_case.png")
    st.subheader("Rocket Pack: Gradient Boost")
    st.image("https://cdn.pastemagazine.com/www/blogs/lists/2012/10/05/thunderball-rocket-pack.jpg")
    # st.image("")
    # st.image("")
    # st.image("")
    # st.image("")
    # st.image("")
    # st.image("")
    # st.image("")
    # st.image("")
    
    # st.write("""
     
    # ### Dataset Intro
    # - Kwame working w/ a company that is tyring to predict corporate bond default.  Our take on this project took a couple twists and turns along the way.
    
    # - Our initial goal was to predict whether a bond would default or not.  Once we realized that this would be much more involved than a 2 week project would allow.
    
    # - The company was kind enough to give us partial access to some of the data that they pull into their models.  



    # """)
    
if select == 'Objectives':
    st.title("Objectives")

    st.header("Goals")
    if st.checkbox("Show Project Goals"):
        st.subheader("Assigment from M")
        st.image("https://i2-prod.mirror.co.uk/incoming/article3114398.ece/ALTERNATES/s1200c/Judi-Dench-as-M.jpg")
        st.write(""" 
                - Develop a machine learning model that predicts a change in corporate credit ratings 
                    over future 12 months for a given corporate bond
                - Stake holders for this model includes bond investors and risk professionals in charge 
                    of bond collateral
                - We experimented with a number of classification models incl. logistic regression, 
                    random forest and GradientBoost
                - To train our model we obtained  a number of large datasets with corporate financial data, 
                    credit ratings, bond/equity pricing and macro data
                - We refined our feature set by selecting relevant features across various datasets and 
                    merging datables
                - We also  experimented with a variety of rating target variables and ultimately chose 
                - Future improvements include enhancing our datasets, model refinement and 
                    optimizing target variables
            """)

    st.header("Credit Migration")
    if st.checkbox("Summary of Credit Migration"):
        st.image("https://www.glynholton.com/wp-content/uploads/2013/06/exhibit_default_model_1_v2.png")
        st.subheader("")
        st.write("""
            ### General
            - Credit Rating quantifies the creditworthiness of borrower in general terms or with respect to a particular debt or financial obligation 
            - Credit rating is sought by an Entity looking to borrow money - corporation, province authority, sovereign government etc.
            - Common practice to map frequency of defaults with Credit Rating. Credit Rating not indicative of probability of Default. 
            - Credit Rating determined by Entity’s past history of borrowing and paying off debts and future economic & growth prospects
            ### Causes
            - Credit Migration is moving of a security issuer from one class of risk into new one. 
            - It leads to downgrade or upgrade from current rating.
            - Credit migration event could cause significant movement in bond prices
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
                - Company Fundamentals: Quarterly Data of 17,776 bonds since 1998
                - Bond price & coupon 
                - Credit Ratings
            ### Macro Economic Data: 
                - Key Interest Rate
                - Commodity price 
            ### Target Data: 
                - Rating
                - Rating Change 
            ### Data Cleaning Challenges: 
                - Stale bond price, bond price not available on all quarterly date 
                - Not many common cusips between EDI Bond and Core Fundamentals
                - Rating publish date not available on all quarterly date
            ### Usage of AsyncIO:
                - Find latest Bond price for corresponding quarterly date
                - Find latest Rating for corresponding 
                - Process the Data
            ### Final Data Set:
                - After Dropping NAs, Data size is 13.5K rows of quarterly data with Bond price and rating


            """)

    data = Path('/Users/wesleysapone/James-Bond/streamlit/data/feature_df.csv')
    df = pd.read_csv(data)
    model_df = model_df.drop(columns="Unnamed: 0")
    st.write(model_df)



if select == 'Model Selection':
    st.title("Technical Briefing with Q")
    st.image("https://carwow-uk-wp-2.imgix.net/2002-Aston-Martin-V12-Vanquish-scaled.jpg?auto=format&cs=tinysrgb&fit=clip&ixlib=rb-1.1.0&q=60&w=1920")
    st.write("""
        ### Logistic Regression 
            - Logistic Regression is a statistical method for predicting binary outcomes from data.
            - Examples of this are "yes" vs "no" or "high credit risk" vs "low credit risk".
            - These are categories that translate to probability of being a 0 or a 1
            - We can calculate logistic regression by adding an activation function as the 
                final step to our linear model.
            - This converts the linear regression output to a probability.
 
        ### Random Forest 
            - Random forest is an ensemble of decision tree algorithms, popular machine learning algorithm across classification and regression problems
            - from sklearn.ensemble import RandomForestClassifier
        Key hyperparameters:
            - max_samples: Default=None (sample size=training set size)
                - Tuning: 
            - max_features:  Default= sqrt(input features)
                - Tuning: 
            - n-estimators: Default= 100
                - Tuning: 
            - max_depth: Default=None (arbitrary)
                - Tuning: 
            - Train_test_split
            - Feature scaling using StandardScaler()
            Model evaluation: balanced accuracy score, classification report, feature importance

        ### Gradient Boosting
            - 
        ### Feature Selection
            - Scope:  Bonds, Debentures, MTN (Medium Term Notes), Notes 
            - Trends vs. Absolute
            - Number of Bonds
            - List 32 features

    """)


if select == 'Model Evaluation':
    st.title("Visualizations")
    if st.checkbox("Random Forest"):
        st.header("Gadget Evaluation and Tuning")
        # st.image("")
        st.subheader("Economic")
        data = Path('/Users/wesleysapone/James-Bond/streamlit/data/feature_df.csv')
        df = pd.read_csv(data)
    
        st.write(model_df)

        importance = Path('/Users/wesleysapone/James-Bond/streamlit/data/gboost_importance.csv')
        boost_imp = pd.read_csv(importance)
        boost_imp.set_index('1', inplace=True)
        st.subheader("gradient boost feature importance sorted df")
        st.write(boost_imp)
        st.subheader("gradient boost feature importance bar chart")
        st.bar_chart(boost_imp['Feature Importances'])

        st.subheader("gboost top 3 feature importance concentrations")
        hst = pd.DataFrame(model_df[:300], columns = ['pe', 'close_edi', 'liabilities'])
        hst.hist()
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # log_importance = Path('/Users/wesleysapone/James-Bond/streamlit/data/logistic_importance.csv')
        # log_imp = pd.read_csv(log_importance)
        # log_imp.set_index('1', inplace=True)
        # st.write(log_imp)
        # st.bar_chart(log_imp['Feature Importances'])
        
        st.subheader("imbalanced random forest over sampling model classification report")
        st.image("/Users/wesleysapone/James-Bond/streamlit/images/imb_over_samp.png")
        rf_importance = Path('/Users/wesleysapone/James-Bond/streamlit/data/smote_rf_importance_df.csv')
        rf_imp = pd.read_csv(rf_importance)
        rf_imp.set_index('1', inplace=True)
        st.subheader("random forest feature importance sorted df")
        st.write(rf_imp)
        st.subheader("random forest feature importance bar chart")
        st.bar_chart(rf_imp['Feature Importances'])

        st.subheader("random forest classification report")
        st.image("https://keep.google.com/u/0/media/v2/1rX15QEuZDqY9IXJgKrwJc2IVm68OP307Qe591B0s3tbB7qG2z9a2RUKmSab9K6U/1cDHQy4a5eQ_oPfumcK6ac16qdfRxMh68Wg-hCR6m24uv2vsaZEJIoIeRKVLYreA?accept=image/gif,image/jpeg,image/jpg,image/png,image/webp,audio/aac&sz=546")
    
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
        rf_model = RandomForestClassifier(n_estimators= (int(len(model_df)/3)), max_depth=3, max_samples = None, random_state=0)

        # Fit the model
        rf_model = rf_model.fit(X_train_scaled, y_train)

        # Make a prediction of "y" values from the X_test dataset
        predictions = rf_model.predict(X_test_scaled)
        # print(np.unique(predictions))
        # print(np.unique(y_test))


        cm = confusion_matrix(y_test, predictions)
        cm_df = pd.DataFrame(
            cm, index=["Actual -4","Actual -3", "Actual -2", "Actual -1", "Actual 0", "Actual +1", "Actual +2"], 
                    columns=["Predicted -4","Predicted -3", "Predicted -2","Predicted -1","Predicted 0", "Predicted 1", "Predicted 2"])


            # Calculating the accuracy score
        acc_score = accuracy_score(y_test, predictions)
        


    # Displaying results
    print("Confusion Matrix")
    # display(cm_df)
    # st.write(cm_df)
    # st.write(print(f"Accuracy Score : {acc_score}"))
    st.write(print("Classification Report"))
    # st.write(print(classification_report(y_test, predictions)))

    # st.write(print("Balanced accuracy score: %.4f" % balanced_accuracy_score(y_test, predictions)))


    # Print the imbalanced classification report
    print('Classification Report Imbalanced')
    # print(classification_report_imbalanced(y_test, predictions))


    # Get the feature importance array
    # importances = rf_model.feature_importances_

    # List the top 10 most important features
    # importances_sorted = sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)
    # importances_sorted[:10]

    # Visualize the features by importance
    # importances_df = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns), reverse=True))
    # importances_df.set_index(importances_df[1], inplace=True)
    # importances_df.drop(columns=1, inplace=True)
    # importances_df.rename(columns={0: 'Feature Importances'}, inplace=True)
    # importances_sorted = importances_df.sort_values(by='Feature Importances')
    # importances_sorted.plot(kind='barh', color='lightgreen', title= 'Features Importances', legend=False)

if select == "Random Oversampling - Random Forest":


    # RANDOM OVERSAMPLING
    ros = RandomOverSampler(random_state=1)
    X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)
    Counter(y_resampled)


    # Create a random forest classifier
    rf_ro_model = RandomForestClassifier(n_estimators=80, max_depth=3, random_state=0)

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

    st.subheader("smote rf balanced accuracy score")
    st.write(balanced_accuracy_score(y_test, y_pred))

    # Print the imbalanced classification report
    indent = '         '
    st.subheader(f"{indent} precision | recall | spe | f1 | geo | iba | support")
    st.write(classification_report_imbalanced(y_test, y_pred))


    # #SMOTEENN combination sampling
    # sm = SMOTEENN(random_state=1)
    # X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)
    # Counter(y_resampled)

    # # Create a random forest classifier
    # rf_model_SO = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)

    # # Fit the model
    # rf_model_SO = rf_model_SO.fit(X_resampled, y_resampled)

    # # Make a prediction of "y" values from the X_test dataset
    # predictions = rf_model_SO.predict(X_test_scaled)

    # # Display the confusion matrix
    # y_pred = rf_model_SO.predict(X_test_scaled)
    # confusion_matrix(y_test, y_pred)


    # # Balanced accuracy score
    # balanced_accuracy_score(y_test, y_pred)


    # # Print the imbalanced classification report
    # print(classification_report_imbalanced(y_test, y_pred))













st.header(select)
if select == 'Phase 2':
    pass 
    r = requests.get("https://docs.google.com/presentation/d/1sEb5d1Mnd54tOAWrbJ122fDnsRDyVLqET8sxhIAlT9Q/edit#slide=id.gc6f9e470d_0_5")






if select == "Random Forest Code":
    st.write("""
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
rf_model = RandomForestClassifier(n_estimators= (int(len(model_df)/3)), max_depth=3, max_samples = None, random_state=0)

# Fit the model
rf_model = rf_model.fit(X_train_scaled, y_train)

# Make a prediction of "y" values from the X_test dataset
predictions = rf_model.predict(X_test_scaled)
# print(np.unique(predictions))
# print(np.unique(y_test))


cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    cm, index=["Actual -4","Actual -3", "Actual -2", "Actual -1", "Actual 0", "Actual +1", "Actual +2"], 
            columns=["Predicted -4","Predicted -3", "Predicted -2","Predicted -1","Predicted 0", "Predicted 1", "Predicted 2"])


# Calculating the accuracy score
acc_score = accuracy_score(y_test, predictions)



# Displaying results
print("Confusion Matrix")
# display(cm_df)
# st.write(cm_df)
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
# importances_df = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns), reverse=True))
# importances_df.set_index(importances_df[1], inplace=True)
# importances_df.drop(columns=1, inplace=True)
# importances_df.rename(columns={0: 'Feature Importances'}, inplace=True)
# importances_sorted = importances_df.sort_values(by='Feature Importances')
# importances_sorted.plot(kind='barh', color='lightgreen', title= 'Features Importances', legend=False)


# RANDOM OVERSAMPLING
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)
Counter(y_resampled)


# Create a random forest classifier
rf_ro_model = RandomForestClassifier(n_estimators=80, max_depth=3, random_state=0)

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

st.subheader("smote rf balanced accuracy score")
st.write(balanced_accuracy_score(y_test, y_pred))

# Print the imbalanced classification report
indent = '         '
st.subheader(f"{indent} precision | recall | spe | f1 | geo | iba | support")
st.write(classification_report_imbalanced(y_test, y_pred))


# #SMOTEENN combination sampling
# sm = SMOTEENN(random_state=1)
# X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)
# Counter(y_resampled)

# # Create a random forest classifier
# rf_model_SO = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)

# # Fit the model
# rf_model_SO = rf_model_SO.fit(X_resampled, y_resampled)

# # Make a prediction of "y" values from the X_test dataset
# predictions = rf_model_SO.predict(X_test_scaled)

# # Display the confusion matrix
# y_pred = rf_model_SO.predict(X_test_scaled)
# confusion_matrix(y_test, y_pred)


# # Balanced accuracy score
# balanced_accuracy_score(y_test, y_pred)


# # Print the imbalanced classification report
# print(classification_report_imbalanced(y_test, y_pred))


"""
    )






