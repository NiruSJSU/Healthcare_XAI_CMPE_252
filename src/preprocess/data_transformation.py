# This file is for cleaning and transforming the data to get it ready for model training
from src.data.dataio import load_pima_indians_data, load_heart_disease_data, load_nhanes_data, save_processed_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def process_pima_indians_dataset(df):
    columns_with_missing_vals = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for column in columns_with_missing_vals:
        df[column] = df[column].replace(0, np.nan)
    
    imputer = SimpleImputer(strategy='median')
    df[columns_with_missing_vals] = imputer.fit_transform(df[columns_with_missing_vals])
    
    scaler = StandardScaler()
    df[columns_with_missing_vals] = scaler.fit_transform(df[columns_with_missing_vals])
    print("Pima Indians dataset cleaned and transformed successfully.")
    return df 
    

if __name__ == "__main__":
    pima_df = load_pima_indians_data()
    print("Pima Indians dataset loaded successfully.")
    print(pima_df.head())
    
    processed_pima_df = process_pima_indians_dataset(pima_df)
    print("Pima Indians dataset processed successfully.")
    print(processed_pima_df.head())
    
    save_processed_data(processed_pima_df, "pima_indians_diabetes_dataset_processed.csv")
    print("Processed Pima Indians dataset saved successfully.")
    
    

