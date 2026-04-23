# This file is for cleaning and transforming the data to get it ready for model training
from src.data.dataio import load_pima_indians_data, load_heart_disease_data, load_nhanes_data, save_processed_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def process_pima_indians_dataset(df):
    df = df.copy()
    
    columns_with_missing_vals = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for column in columns_with_missing_vals:
        df[column] = df[column].replace(0, np.nan)
    
    imputer = SimpleImputer(strategy='median')
    df[columns_with_missing_vals] = imputer.fit_transform(df[columns_with_missing_vals])
    
    scaler = StandardScaler()
    df[columns_with_missing_vals] = scaler.fit_transform(df[columns_with_missing_vals])
    print("Pima Indians dataset cleaned and transformed successfully.")
    return df 

def process_heart_disease_dataset(df):
    df = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns_with_odd_values = ['chol', 'thalach','oldpeak', 'trestbps']
    for column in columns_with_odd_values:
        df[column] = df[column].replace(0, np.nan)
    
    imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    print("Heart disease dataset cleaned and transformed successfully.")
    return df
    

def process_nhanes_dataset(df):
    pass

if __name__ == "__main__":
    pima_df = load_pima_indians_data()
    print("Pima Indians dataset loaded successfully.")
    print(pima_df.head())
    
    processed_pima_df = process_pima_indians_dataset(pima_df)
    print("Pima Indians dataset processed successfully.")
    print(processed_pima_df.head())
    
    save_processed_data(processed_pima_df, "pima_indians_diabetes_dataset_processed.csv")
    print("Processed Pima Indians dataset saved successfully.")
    
    heart_df = load_heart_disease_data()
    print("Heart disease dataset loaded successfully.")
    print(heart_df.head())
    
    processed_heart_df = process_heart_disease_dataset(heart_df)
    print("Heart disease dataset processed successfully.")
    print(processed_heart_df.head())
    
    save_processed_data(processed_heart_df, "heart_disease_processed.csv")
    print("Processed heart disease dataset saved successfully.")
    
    nhanes_df = load_nhanes_data()
    print("NHANES dataset loaded successfully.")
    print(nhanes_df.head())
    
