import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def data_preprocessor(X, bins = False):

    if bins:
        X_binned = X.copy()

        min_age, max_age = X_binned['age'].min(), X_binned['age'].max()
        X_binned['age_category'] = pd.cut(X_binned['age'], bins = [min_age-1,40,45,50,55,60,max_age+1], labels = range(1,7))
        
        min_bmi, max_bmi = X_binned['bmi'].min(), X_binned['bmi'].max()
        X_binned['bmi_zone'] = pd.cut(X_binned['bmi'], bins = [min_bmi-1,17.5,25,30,max_bmi+1], labels = range(1,5))
        
        min_diastolic_bp, max_diastolic_bp = X_binned['blood_pressure_min'].min(), X_binned['blood_pressure_min'].max()
        X_binned['diastolic_zone'] = pd.cut(X_binned['blood_pressure_min'], bins = [min_diastolic_bp-1,80,90,120,max_diastolic_bp+1], labels = range(1,5))
        
        min_systolic_bp, max_systolic_bp = X_binned['blood_pressure_max'].min(), X_binned['blood_pressure_max'].max()
        X_binned['systolic_zone'] = pd.cut(X_binned['blood_pressure_max'], bins = [min_systolic_bp-1,120,140,180,max_systolic_bp+1], labels = range(1,5))
        
        X_binned.drop(columns = ['age', 'bmi', 'blood_pressure_min', 'blood_pressure_max'], inplace = True)
        X_binned = X_binned.astype('category')

        return X_binned

    else:
        numerical_columns = ['age', 'bmi', 'blood_pressure_max', 'blood_pressure_min', 'cholesterol_level', 'glucose_level']
        X_numerical = X[numerical_columns]
        X_categorical = X.drop(columns = numerical_columns)

        scaler = StandardScaler()
        X_scaled= scaler.fit_transform(X_numerical)
        X_scaled = pd.DataFrame(X_scaled, columns = X_numerical.columns, index = X_numerical.index)

        X_processed = pd.merge(X_scaled, X_categorical, left_index = True, right_index = True)
        return X_processed
