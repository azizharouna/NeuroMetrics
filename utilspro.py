import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from collections import Counter

#reading the data 
# Unzipping the provided dataset
with zipfile.ZipFile("data/amp-parkinsons-disease-progression-prediction_2.zip", 'r') as zip_ref:
    zip_ref.extractall("data/amp-parkinsons-disease-progression-prediction_2")

# Load each dataset into a dataframe
train_peptides = pd.read_csv("data/amp-parkinsons-disease-progression-prediction_2/train_peptides.csv")
train_proteins = pd.read_csv("data/amp-parkinsons-disease-progression-prediction_2/train_proteins.csv")
train_clinical_data = pd.read_csv("data/amp-parkinsons-disease-progression-prediction_2/train_clinical_data.csv")
supplemental_clinical_data = pd.read_csv("data/amp-parkinsons-disease-progression-prediction_2/supplemental_clinical_data.csv")




def execute_notebook(notebook_filename):
    with open(notebook_filename, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    ep.preprocess(notebook, {'metadata': {'path': './'}})

    with open(notebook_filename, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)


# Helper function to get the three most common
def three_most_common(lst):
    count = Counter(lst)
    most_common = count.most_common(3)
    return [item[0] for item in most_common]

def optimized_info_df_v5(df):
    # Lists to store results
    modes_1st = []
    modes_2nd = []
    modes_3rd = []

    # Loop through columns and compute modes
    for col in df.columns:
        value_counts = df[col].value_counts()
        top_3 = value_counts.head(3).index.tolist()
        
        modes_1st.append(top_3[0] if len(top_3) > 0 else None)
        modes_2nd.append(top_3[1] if len(top_3) > 1 else None)
        modes_3rd.append(top_3[2] if len(top_3) > 2 else None)

    # Construct the results DataFrame
    result = pd.DataFrame({
        'Column': df.columns,
        'Dtype': df.dtypes.values,
        'Null Count': df.isnull().sum().values,
        'Unique Count': df.nunique().values,
        '% Missing': (df.isnull().sum() / len(df) * 100).values,
        '1st Mode': modes_1st,
        '2nd Mode': modes_2nd,
        '3rd Mode': modes_3rd
    })

    return result



# Function to interpolate UPDRS scores for a single patient's data
def interpolate_updrs(patient_data):
    for col in ['updrs_4']:
        # Interpolate missing values
        patient_data[col] = patient_data[col].interpolate(method='linear', limit_direction='both')
        # Round to nearest integer
        patient_data[col] = patient_data[col].round()
    return patient_data

def categorize_abundance(value, quartiles):
    """
    Categorize abundance based on given quartiles.
    
    Args:
    - value (float): The abundance value to categorize.
    - quartiles (Series): The quartiles for the UniProt group.
    
    Returns:
    - str: The abundance category ("low", "medium", "high", or "very high").
    """
    if value <= quartiles[0.25]:
        return 'low'
    elif value <= quartiles[0.5]:
        return 'medium'
    elif value <= quartiles[0.75]:
        return 'high'
    else:
        return 'very high'




# Define a function to calculate dynamic rolling averages.
def dynamic_rolling_average(series, window_size):
    """
    Calculate rolling average with a dynamic window for the given series.
    
    Parameters:
    - series: A pandas Series for which the rolling average is to be calculated.
    - window_size: The maximum desired window size for calculating the rolling average.
    
    Returns:
    - A pandas Series containing the rolling averages.
    """
    
    result = []  # Initialize an empty list to store the rolling averages.
    
    # Loop through each data point in the series.
    for i in range(1, len(series) + 1):
        
        # For data points before reaching the desired window size, 
        # calculate the average of all previous points.
        if i < window_size:
            avg = series[:i].mean()
        # For data points after reaching the desired window size, 
        # calculate the average of the last 'window_size' data points.
        else:
            avg = series[i - window_size:i].mean()
        
        # Append the calculated average to the result list.
        result.append(avg)
    
    # Convert the result list to a pandas Series and return.
    return pd.Series(result)

# Apply the dynamic_rolling_average function to the 'updrs_3' column for each patient.
# The result is a new column in the dataframe: 'updrs_3_dynamic_rolling_avg'
'''clinical_data_sorted['updrs_3_dynamic_rolling_avg'] = clinical_data_sorted.groupby('patient_id')['updrs_3'].apply(
lambda x: dynamic_rolling_average(x, 3)
).reset_index(level=0, drop=True)'''


def fast_dynamic_rolling_average(series, window_size):
    """
    Calculate rolling average with a dynamic window for the given series.
    
    Parameters:
    - series: A pandas Series for which the rolling average is to be calculated.
    - window_size: The maximum desired window size for calculating the rolling average.
    
    Returns:
    - A pandas Series containing the rolling averages.
    """
    
    # For each index in the series, calculate the rolling mean with the appropriate window size
    # The window size is the minimum of the current index and the desired window_size
    return series.rolling(window=min(window_size, series.index[-1]+1), min_periods=1).mean()



# Apply the fast_dynamic_rolling_average function to the 'updrs_3' column for each patient.
# The result is a new column in the dataframe: 'updrs_3_fast_dynamic_rolling_avg'
''' clinical_data_sorted['updrs_3_fast_dynamic_rolling_avg'] = clinical_data_sorted.groupby('patient_id')['updrs_3'].apply(
    lambda x: fast_dynamic_rolling_average(x, 3)
).reset_index(level=0, drop=True) '''

def dynamic_rolling_average2(grouped_df, col):
    # If there's only one record for the patient, return the original value
    if len(grouped_df) == 1:
        return grouped_df[col]
    
    # Otherwise, compute a rolling average based on available data points
    return grouped_df[col].rolling(window=len(grouped_df), min_periods=1).mean()

'''''
# Group by patient and apply the dynamic rolling average function
clinical_data_sorted['updrs_3_dynamic_avg'] = clinical_data_sorted.groupby('patient_id').apply(dynamic_rolling_average).reset_index(level=0, drop=True)

# Check the number of NaN values in the new column
nan_count = clinical_data_sorted['updrs_3_dynamic_avg'].isna().sum()
print(f"Number of NaNs in the dynamic rolling average column: {nan_count}")

'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

# Frequency encode a column in a dataframe
def frequency_encode(df, column):
    """
    Returns a dataframe with an added frequency encoded column.
    """
    freq = df[column].value_counts()
    df[column + '_encoded'] = df[column].map(freq)
    return df

# Evaluate model with given features
def evaluate_model(df, features, target_column='updrs_3'):
    """
    Returns the evaluation metrics of the model for given features.
    """
    X = df[features]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'features': features.copy(),
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

# Perform the iterative process of adding features and evaluating the model
def iterative_modeling(df, all_features, target_column='updrs_3'):
    """
    Iteratively add features and evaluate the model. Returns a list of evaluation metrics.
    """
    initial_feature = random.choice(all_features)
    additional_features = [feature for feature in all_features if feature != initial_feature]
    current_features = [initial_feature]
    
    results = []
    for feature in additional_features:
        current_features.append(feature)
        metrics = evaluate_model(df, current_features, target_column)
        results.append(metrics)
    
    return results

# Example usage:
# clinical_data_sorted_2 = frequency_encode(clinical_data_sorted_2, 'UniProt')
# all_features = ['updrs_1', 'updrs_2', 'updrs_4', 'time_since_diagnosis', 'medication_numeric', 'UniProt_encoded' ,'NPX', 'PeptideAbundance']
# results = iterative_modeling(clinical_data_sorted_2, all_features)

# Display the results
# for metrics in results:
#     print(f"Features: {metrics['features']}")
#     print(f"Mean Squared Error: {metrics['mse']}")
#     print(f"Mean Absolute Error: {metrics['mae']}")
#     print(f"R-squared: {metrics['r2']}")
#     print("-------------------------")
