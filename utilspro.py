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