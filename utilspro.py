import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#reading the data 
# Unzipping the provided dataset
with zipfile.ZipFile("data/amp-parkinsons-disease-progression-prediction_2.zip", 'r') as zip_ref:
    zip_ref.extractall("data/amp-parkinsons-disease-progression-prediction_2")

# Load each dataset into a dataframe
train_peptides = pd.read_csv("data/amp-parkinsons-disease-progression-prediction_2/train_peptides.csv")
train_proteins = pd.read_csv("data/amp-parkinsons-disease-progression-prediction_2/train_proteins.csv")
train_clinical_data = pd.read_csv("data/amp-parkinsons-disease-progression-prediction_2/train_clinical_data.csv")
supplemental_clinical_data = pd.read_csv("data/amp-parkinsons-disease-progression-prediction_2/supplemental_clinical_data.csv")