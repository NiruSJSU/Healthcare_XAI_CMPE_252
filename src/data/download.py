# This file is for downloading datasets as needed

from ucimlrepo import fetch_ucirepo

def download_heart_disease_dataset():
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
    
    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets 
    
    # metadata 
    print(heart_disease.metadata) 
    
    # variable information 
    print(heart_disease.variables) 

if __name__ == "__main__":
    download_heart_disease_dataset()