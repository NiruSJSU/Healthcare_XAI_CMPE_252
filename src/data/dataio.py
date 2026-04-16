#This file is used for loading data and setting it up for preprocessing

import pandas as pd
import numpy as np

def load_pima_indians_data(file_path):
    # Loads the pima indians dataset
    return pd.read_csv("data/pima_indians_diabetes_dataset.csv")

def load_heart_disease_data(file_path):
    # Loads the heart disease dataset
    return pd.read_csv("data/heart_disease_uci_labelled.csv")

def load_nhanes_data(file_path):
    # Loads the nhanes dataset
    return pd.read_csv("data/Nhanes_cvd_raw.csv")

if __name__ == "__main__":
    pima_indians_df = load_pima_indians_data('data/pima_indians_diabetes_dataset.csv')
    print(pima_indians_df.head())
    heart_df = load_heart_disease_data('data/heart_disease_uci_labelled.csv')
    print(heart_df.head())
    nhanes_df = load_nhanes_data('data/Nhanes_cvd_raw.csv')
    print(nhanes_df.head())