#This file is used for loading data and setting it up for preprocessing

import pandas as pd
import numpy as np
from pathlib import Path

Base_DIR = Path.cwd()

RAW_DATA_DIR = Base_DIR/ "Data"/ "Raw"
PROCESSED_DATA_DIR = Base_DIR /"Data"/"Processed"

def load_pima_indians_data():
    # Loads the pima indians dataset
    return pd.read_csv(RAW_DATA_DIR/ "pima_indians_diabetes_dataset.csv")

def load_heart_disease_data():
    # Loads the heart disease dataset
    return pd.read_csv(RAW_DATA_DIR/ "heart_disease.csv")

def load_nhanes_data():
    # Loads the nhanes dataset
    return pd.read_csv(RAW_DATA_DIR/ "Nhanes_cvd_raw.csv")

def save_processed_data(df, file_name):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR/ file_name
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    pima_df = load_pima_indians_data()
    print("Pima dataset loaded successfully.")
    print(pima_df.head())
    
    heart_df = load_heart_disease_data()
    print("Heart dataset loaded successfully.")
    print(heart_df.head())
    nhanes_df = load_nhanes_data()
    print("NHANES dataset loaded successfully.")
    print(nhanes_df.head())