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

3. Patients with high blood pressure were the group most at risk of disease. <br/><br/>
<img src="/images/top_12_groups.png"/>

## Modelling

General methodology:

1. Train baseline model. <br/><br/>
2. Validate baseline model using AUC value. <br/><br/>
3. Tune hyperparameters using Grid Search with 5-fold Cross Validation. <br/><br/>
4. Train tuned model. <br/><br/>
5. Find optimal probability threshold for tuned model using ROC curve. <br/><br/>
6. Validate tuned model (with updated threshold) using AUC value, precision and recall. <br/><br/>

Baseline models:

<img src="/images/precision_recall_base.png"/>

Ensemble models:

<img src="/images/precision_recall_ensemble.png"/>

## Model Selection

Possible outcomes:

- **True Positive** = diagnosing someone with disease, when they have disease. <br/><br/>
- **True Negative** = diagnosing someone as healthy, when they are healthy. <br/><br/>
- **False Positive** = diagnosing someone with disease, when they are healthy. <br/><br/>
- **False Negative** = diagnosing someone as healthy, when they have disease.

After considering the possible outcomes for disease detection, it was decided that the cost of a false negative was greater than a false positive in the absence of any additional domain-specific information. As a result of this, recall was given more importance than precision during the model selection process.

Model selection:

<img src="/images/precision_recall_plot.png" width="700"/> <br/><br/>

Using the graph above, it was decided that the **Logistic Regression** model had the best precision recall trade-off compared with all the other models tested. In addition to this, it is also parametric meaning that the relative importances of each feature can be easily interpreted from the coefficients.

<img src="/images/lr_coefficients.png" width="550"/> <br/><br/>

## Conclusions

Conclusions:

- Models generally had a bias towards precision, even when the probability threshold was set well below 50%. This suggests that the features used to train the models were not preditive enough for detecting disease with a sufficient level of confidence. To address this limitation in the data, more specific categories for alcohol consumption, smoking level and activity level should be provided to patients in future surveys. <br/><br/>
- Systolic blood pressure was considerably more important than diastolic blood pressure in predicting disease. This suggests that patients with a high diastolic blood pressure, in the absence of the high systolic blood pressure, should not be considered to be at significant risk. This important distinction was not captured during the EDA stage of the project. <br/><br/>
- Age consistently appeared amongst the top three features for most models. Using this information, healthcare institutions could provide targeted health advice to people over the age of 40 to help reduce their likelihood of disease. 
