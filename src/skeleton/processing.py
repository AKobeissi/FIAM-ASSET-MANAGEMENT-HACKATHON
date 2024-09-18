import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

def preprocess_data(df):
    # Handle missing values
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Handle outliers and scale
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)
    
    # Dimensionality reductionpi0p
    pca = PCA(n_components=0.95)
    df_pca = pd.DataFrame(pca.fit_transform(df_scaled))
    
    return df_pca, pca.components_

# Load data
data = pd.read_csv('hackathon_sample_v2.csv')

# Preprocess
processed_data, pca_components = preprocess_data(data)

