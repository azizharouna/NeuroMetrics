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
