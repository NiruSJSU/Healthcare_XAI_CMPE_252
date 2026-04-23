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
    print("Pima Indians dataset computation done")
    return df 

def process_heart_disease_dataset(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    columns_with_odd_values = ['chol', 'thalach','oldpeak', 'trestbps']
    for column in columns_with_odd_values:
        df[column] = df[column].replace(0, np.nan)
    
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print("Heart disease dataset computation done")
    return df
    

def process_nhanes_dataset(df):
    df = df.copy()
    
    df.drop(columns=['SEQN'], inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    columns_with_disease_indicators = ['Congestive,Coronary,Heart_attack,Stroke,Angina']
    
    for column in columns_with_disease_indicators:
        if column in df.columns:
            df[column] = df[column].replace({1: 1, 2: 0, 9: np.nan})
            
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    scaler = StandardScaler()
    scalable_cols = [col for col in numeric_cols if col not in columns_with_disease_indicators]
    df[scalable_cols] = scaler.fit_transform(df[scalable_cols])
    
    print("NHANES dataset computation done")
    return df 

if __name__ == "__main__":
    pima_df = load_pima_indians_data()
    print("Pima Indians dataset loaded from csv")
    print(pima_df.head())
    
    processed_pima_df = process_pima_indians_dataset(pima_df)
    print("Pima Indians dataset processed")
    print(processed_pima_df.head())
    
    save_processed_data(processed_pima_df, "pima_indians_diabetes_dataset_processed.csv")
    print("Pima Indians dataset saved to new csv")
    
    heart_df = load_heart_disease_data()
    print("Heart disease dataset loaded from csv")
    print(heart_df.head())
    
    processed_heart_df = process_heart_disease_dataset(heart_df)
    print("Heart disease dataset processed")
    print(processed_heart_df.head())
    
    save_processed_data(processed_heart_df, "heart_disease_processed.csv")
    print("Processed Heart disease dataset saved to new csv")
    
    nhanes_df = load_nhanes_data()
    print("NHANES dataset loaded from csv")
    print(nhanes_df.head())
    
    save_processed_data(process_nhanes_dataset(nhanes_df), "nhanes_cvd_processed.csv")
    print("Processed NHANES dataset saved to new csv")
    
