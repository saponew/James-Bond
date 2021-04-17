# James-Bond
## Corporate Bond Rating Change Prediction Model
### Team Members: Wes Sapone, Kwame van Leeuwen, Ketan Patel, Susan Fan

###  Objective: Develop a machine learning model that predicts a change in credit ratings for a corporate bond over the following 12 months 

## Summary
Obtained large datasets via SQL from a company active in corporate credit default probability modeling. Data cleaning played a major role in the project. 3 Classification Models: Logistic Regression, Random Forest and Gradient Boost were used to train/test the models and analyze the large historical dataframes. Having an imbalanced dataset with relatively few rating events turned out to be a key challenge. Random Forest and Gradient Boost models seem to outperform Logistic Regression model with balanced accuracy scores of about 83% using the original, imbalanced dataset. Potential future modeling improvements include enhancing the dataset, fine-tuning the models and optimizing target variables.

## Data Cleaning
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/01.png "AsyncIO Concurrent Task 1")
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/02.png "AsyncIO Concurrent Task 2")

## Logistic Regression
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/03.png "Logistic Regression Original Data, Default Parameters")
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/04.png "Logistic Regression Visualization")

## Random Forest
### Random Forest: Original Data, Default Parameters 1
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/05.png "Random Forest Original Data, Default Parameters 1")
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/06.png "Random Forest Visualization, Default Parameters 1")
### Random Forest: Original Data, Max Depth=5
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/07.png "Random Forest Original Data, Max Depth=5")
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/08.png "Random Forest Visualization, Max Depth=5")
### Random Forest: SMOTE Oversampling, Max depth=5
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/09.png "Random Forest SMOTE Oversampling, Max depth=5")
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/10.png "Random Forest Visualization, SMOTE Oversampling, Max depth=5")
### Random Forest: SMOTE Oversampling, Max depth=5
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/11.png "Random Forest SMOTEEN Resampling, Max depth=5")
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/12.png "Random Forest Visualization, SMOTEEN Resampling, Max depth=5")

## Gradient Boosting
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/13.png "Gradient Boosting Original Data")
![alt text](https://github.com/saponew/James-Bond/blob/main/readme_images/14.png "Gradient Boosting Visualization")


