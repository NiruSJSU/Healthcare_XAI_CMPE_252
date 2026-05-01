from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def get_model_pipeline():
    models = [
        (LogisticRegression(max_iter=1000, class_weight='balanced'), "Logistic Regression"),
        (RandomForestClassifier(random_state=42, class_weight='balanced'), "Random Forest"),
        (SVC(probability=True, random_state=42, class_weight='balanced'), "SVM")
        (MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42), "MLP"),
        (LinearDiscriminantAnalysis(), "LDA")
    ]
    return models

