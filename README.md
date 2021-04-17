# James-Bond
## Corporate Bond Rating Change Prediction Model
### Team Members: Wes Sapone, Kwame van Leeuwen, Ketan Patel, Susan Fan

###  Objective: Develop a machine learning model that predicts a change in credit ratings for a corporate bond over the following 12 months

## Summary
Obtained large datasets via SQL from a company active in corporate credit default probability modeling. Data cleaning played a major role in the project. 3 Classification Models: Logistic Regression, Random Forest and Gradient Boost were used to train/test the models and analyze the large historical dataframes. Having an imbalanced dataset with relatively few rating events turned out to be a key challenge. Random Forest and Gradient Boost models seem to outperform Logistic Regression model with balanced accuracy scores of about 83% using the original, imbalanced dataset. Potential future modeling improvements include enhancing the dataset, fine-tuning the models and optimizing target variables.

### Presentation Slides
<a href="https://github.com/saponew/James-Bond/blob/main/images/presentation_slides.pptx"> View Slides </a>

## Data Cleaning                                                                                       

<a href="https://github.com/saponew/James-Bond/blob/main/dataset/core_bonds_rating.py"> Core US Rating Shifted</a>
<br><a href="https://github.com/saponew/James-Bond/blob/main/dataset/core_us_rating_shited.py"> Core Bonds Rating</a>
<br><a href="https://github.com/saponew/James-Bond/blob/main/dataset/credit_rating_move_v1.py"> Credit Rating Move v.1</a>
<br><a href="https://github.com/saponew/James-Bond/blob/main/dataset/credit_rating_move_v3.py"> Credit Rating Move v.3</a>

## Models
### Logistic Regression

<a href="https://github.com/saponew/James-Bond/blob/main/images/03.png"> Logistic Regression Original Data, Default Parameters</a>
<br><a href="https://github.com/saponew/James-Bond/blob/main/images/04.png"> Logistic Regression Visualization</a>

### Random Forest
#### Original Data, Default Parameters 1
<a href="https://github.com/saponew/James-Bond/blob/main/images/05.png"> Random Forest Original Data, Default Parameters 1</a>
<br><a href="https://github.com/saponew/James-Bond/blob/main/images/06.png"> Random Forest Visualization, Default Parameters 1</a>

#### Original Data, Max Depth=5
<a href="https://github.com/saponew/James-Bond/blob/main/images/07.png"> Random Forest Original Data, Max Depth=5</a>
<br><a href="https://github.com/saponew/James-Bond/blob/main/images/08.png"> Random Forest Visualization, Max Depth=5</a>
#### SMOTE Oversampling, Max depth=5
<a href="https://github.com/saponew/James-Bond/blob/main/images/09.png"> Random Forest SMOTE Oversampling, Max depth=5</a>
<br><a href="https://github.com/saponew/James-Bond/blob/main/images/10.png"> Random Forest Visualization, SMOTE Oversampling, Max depth=5</a>
#### SMOTEEN Resampling, Max depth=5
<a href="https://github.com/saponew/James-Bond/blob/main/images/11.png"> Random Forest SMOTEEN Resampling, Max depth=5</a>
<br><a href="https://github.com/saponew/James-Bond/blob/main/images/12.png"> Random Forest Visualization, SMOTEEN Resampling, Max depth=5</a>

### Gradient Boosting
<a href="https://github.com/saponew/James-Bond/blob/main/images/13.png"> Gradient Boosting Original Data</a>
<br><a href="https://github.com/saponew/James-Bond/blob/main/images/14.png"> Gradient Boosting Visualization</a>
