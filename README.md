# Cardiovascular Disease Detection

<img src="/images/cardio_banner.jpeg"/>

## Motivation

The motivation for this project was to build a series of classification models to predict the likelihood of a patient developing cardiovascular disease based on their lifestyle and physical condition. The best performing model could then be used by healthcare institutions to automate the initial screening process for patients being considered for the disease. This would aim to reduce the time required for patients most at risk to receive a formal diagnosis, improving their chances of recovery.

## Data Collection

The dataset used for this project consisted of information provided by 70,000 adult patients being tested for cardiovascular disease. This included a combination of objective features, such as age and gender, subjective features, such as smoking status and alcohol intake, and features obtained via medical examination, such as glucose and cholesterol levels, and blood pressure. The repository used to source this data can be found [here](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset).

## Data Cleaning

The cleaning process for this project involved:

- Dropping duplicate rows. <br/><br/>
- Calculating more informative measures such as BMI scores. <br/><br/>
- Dropping patients with unreasonable blood pressures and BMI scores. <br/><br/>
- Undersampling from the majority class (negative).

## EDA

Upon exploring the data, I discovered that:

1. Patients generally became more prone to disease as they got older. <br/><br/>
<img src="/images/age_and_gender.png"/>

2. Patients with a high bmi increased their risk of disease. <br/><br/>
<img src="/images/bmi_and_gender.png"/>

3. Patients with high blood pressure were the most at risk of disease. <br/><br/>
<img src="/images/top_12_groups.png"/>

## Modelling

The models used in this project included: 

- **Logistic Regression** <br/><br/>
- **k-Nearest Neighbours** <br/><br/>
- **Naive Bayes Classifier** <br/><br/>
- **Support Vector Machine** <br/><br/>
- **Decision Tree** <br/><br/>
- **Random Forest (with/without boosting)** <br/><br/>
- **Nueral Network** <br/><br/>
- **Soft/Hard Voting Classifiers** <br/><br/>
- **Stacked Models** <br/><br/>

General methodology for each model:

1. Train baseline model. <br/><br/>
2. Validate baseline model using AUC value. <br/><br/>
3. Tune hyperparameters using Grid Search with 5-fold Cross Validation. <br/><br/>
4. Train tuned model. <br/><br/>
5. Find optimal probability threshold for tuned model using ROC curve. <br/><br/>
6. Validate tuned model (with updated threshold) using AUC value, precision and recall. <br/><br/>



## Conclusions
