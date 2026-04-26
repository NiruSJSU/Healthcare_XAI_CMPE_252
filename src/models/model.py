from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_model_pipeline():
    models = [
        (LogisticRegression(max_iter=1000, class_weight='balanced'), "Logistic Regression"),
        (RandomForestClassifier(random_state=42, class_weight='balanced'), "Random Forest"),
        (SVC(probability=True, random_state=42, class_weight='balanced'), "SVM")
    ]
    return models

