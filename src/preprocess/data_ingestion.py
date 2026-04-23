#This file is for splitting data into training and test sets

from src.data.dataio import load_pima_indians_data, load_heart_disease_data, load_nhanes_data, save_processed_data


def load_raw_data():
    pima_df = load_pima_indians_data()
    heart_df = load_heart_disease_data()
    nhanes_df = load_nhanes_data()
    return pima_df, heart_df, nhanes_df

if __name__ == "__main__":
    pima_df, heart_df, nhanes_df = load_raw_data()
    print("Raw data loaded successfully")
    print("Pima Indians first entries:")
    print(pima_df.head())
    print("\nHeart disease first few entries:")
    print(heart_df.head())
    print("\nNHANES dataset first few entries:")
    print(nhanes_df.head())