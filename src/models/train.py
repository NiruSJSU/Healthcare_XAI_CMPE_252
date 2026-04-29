import pandas as pd
from src.data.dataio import load_processed_pima, load_processed_heart, load_processed_nhanes
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import joblib
import os

def train_and_verify(model, X, y, name, ax, save_dir):
    # Holding out the final test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Verification (5 fold)
    cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"\n---- {name} ----")
    
    # Final Training
    model.fit(X_train, y_train)

    # --- SAVE MODEL ---
    # Create a filename-safe version of the model/dataset name
    safe_name = name.replace(" ", "_").replace("-", "_")
    save_path = os.path.join(save_dir, f"{safe_name}.joblib")
    
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")

    final_results(model, X_test, y_test, name, ax)

def final_results(model, X_test, y_test, name, ax):
    # Get standard predictions (0 or 1)
    preds = model.predict(X_test)
    
    # ROC-AUC probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.decision_function(X_test)
    
    # Print all results
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1-Score: {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, probs):.4f}")

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(f"{name}", fontsize=7)

if __name__ == "__main__":
    # 1. SETUP DIRECTORIES
    # Anchors directory creation to the location of train.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trained_models_dir = os.path.join(script_dir, "trained_models")
    
    # Ensure the directory exists
    os.makedirs(trained_models_dir, exist_ok=True)
    print(f"Saving models to: {trained_models_dir}")

    # Load processed datasets
    datasets = [
        (load_processed_pima(), "Outcome", "PIMA Diabetes"),
        (load_processed_heart(), "target", "Heart Disease"),
        (load_processed_nhanes(), "target", "NHANES CVD")
    ]

    # Window for Confusion Matrices (3x5 grid)
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.6, wspace=0.4)

    for row, (df, target_col, dataset_title) in enumerate(datasets):
        print(f"\n{'='*43}\nStarting: {dataset_title}\n{'='*43}")
        
        # Remove duplicates
        df = df.drop_duplicates()
        X = df.drop(columns=[target_col])

        # Drop to prevent data leakage
        if dataset_title == "NHANES CVD":
            leaky_cols = ['Stroke', 'Coronary', 'Angina', 'Congestive', 'Heart_attack']
            X = X.drop(columns=[c for c in leaky_cols if c in X.columns])
          
        if dataset_title == "Heart Disease":
            if 'cp' in X.columns:
                X = X.drop(columns=['cp'])

        y = df[target_col]

        # Models
        models = [
            (LogisticRegression(max_iter=1000, class_weight='balanced'), "Logistic Regression"),
            (RandomForestClassifier(random_state=42, class_weight='balanced'), "Random Forest"),
            (SVC(probability=True, random_state=42, class_weight='balanced'), "SVM"),
            (MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42), "MLP"),
            (LinearDiscriminantAnalysis(), "LDA")
        ]

        # Loop through each model & train
        for col, (model_obj, model_name) in enumerate(models):
            full_name = f"{dataset_title} - {model_name}"
            current_ax = axes[row, col]

            train_and_verify(model_obj, X, y, full_name, current_ax, trained_models_dir)

    print("\nTraining complete. Models saved. Opening Confusion Matrices...")
    plt.show()
