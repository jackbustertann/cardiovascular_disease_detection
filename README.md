# Cardiovascular Disease Detection

## Motivation

The motivation for this project was to predict the likelihood of a patient developing cardiovascular disease based on their lifestyle and physical condition. This could be used to accelerate the screening process for patients most at risk to improve their chances of recovery.

## Data Collection

The dataset used for this project consisted of information provided by 70,000 adult patients being tested for cardiovascular disease. This included a combination of objective features, such as age and gender, subjective features, such as smoking status and alcohol intake, and features obtained by medical examination, such as glucose and cholesterol levels, and blood pressure. The repository used to source this data can be found [here](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset).

## Data Cleaning

The cleaning process for this project involved:

- Dropping duplicate rows. <br/><br/>
- Calculating more informative measures such as BMI scores and blood pressure differences. <br/><br/>
- Dropping patients with unreasonable blood pressures and BMI scores. <br/><br/>
- Undersampling from the majority class (negative).

## EDA

Upon exploring the data, I discovered that:

1. Patients generally became more prone to disease as they got older. <br/><br/>
<img src="/images/age_and_gender.png" width="600"/>

2. Patients with a high bmi increased their risk of disease. <br/><br/>
<img src="/images/bmi_and_gender.png" width="600"/>

3. Patients with high blood pressure were the most at risk of disease. <br/><br/>
<img src="/images/top_12_groups.png"/>

## Models

## Conclusions
