import pandas as pd

def data_cleaner(input_file, output_file):

    # importing data

    raw_data = pd.read_csv(input_file, sep = ';')

    # casting data types

    # changing column names
    raw_data.rename(columns = {'ap_hi': 'blood_pressure_max', 'ap_lo': 'blood_pressure_min', 'cholesterol': 'cholesterol_level', 'gluc': 'glucose_level', 'smoke': 'smoking_status', 'alco': 'alcohol_status', 'active': 'activity_level', 'cardio': 'disease'}, inplace = True)
    # converting height to float
    raw_data['height'] = raw_data['height'].astype('float')
    # converting age from days to years
    raw_data['age'] = raw_data['age'] // 365
    # converting gender into a binary variable
    raw_data['gender'] = raw_data['gender'] - 1

    # dealing with outliers

    # changing sign of negative blood pressure values
    raw_data['blood_pressure_min'] = raw_data['blood_pressure_min'].map(lambda x: -1 * x if x < 0 else x)
    raw_data['blood_pressure_max'] = raw_data['blood_pressure_max'].map(lambda x: -1 * x if x < 0 else x)
    # calculating blood pressure difference
    raw_data['blood_pressure_diff'] = raw_data['blood_pressure_max'] - raw_data['blood_pressure_min']
    # dropping rows with diastolic blood pressure > systolic blood pressure
    raw_data = raw_data.loc[raw_data['blood_pressure_diff'] > 0]
    # dropping rows with infeasible blood pressure values
    raw_data = raw_data.loc[(raw_data['blood_pressure_max'] > 70) & (raw_data['blood_pressure_max'] < 200)]
    raw_data = raw_data.loc[(raw_data['blood_pressure_min'] > 40) & (raw_data['blood_pressure_min'] < 160)]

    # calculating bmi
    raw_data['bmi'] = raw_data['weight'] / (raw_data['height'] / 100)**2
    # dropping height and weight columns
    raw_data.drop(columns = ['height', 'weight'], inplace = True)
    # dropping rows with infeasible bmi values
    raw_data = raw_data.loc[(raw_data['bmi'] > 10) & (raw_data['bmi'] < 60)]

    # dropping duplicate values

    raw_data.drop_duplicates(inplace = True)
    # dropping id column
    raw_data.drop(columns = ['id'], inplace = True)

    # balancing classes

    # undersampling from minority class
    no_disease = raw_data.loc[raw_data['disease'] == 0]
    disease = raw_data.loc[raw_data['disease'] == 1]
    n = min(len(no_disease), len(disease))
    no_disease_reduced = no_disease.sample(n = n, random_state = 0)
    disease_reduced = disease.sample(n = n, random_state = 0)

    clean_data = pd.concat([disease_reduced, no_disease_reduced], axis = 0)

    return clean_data.to_csv(output_file, index = False)