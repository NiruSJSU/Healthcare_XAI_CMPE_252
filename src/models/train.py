import pandas as pd
from src.data.dataio import load_processed_pima, load_processed_heart, load_processed_nhanes
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    ConfusionMatrixDisplay)
import matplotlib.pyplot as plt


def train_and_verify(model, X, y, name, ax):
    # Holding out the final test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Verification (5 fold)
    cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"\n---- {name} ----")
    print(f"\n- Verification Results -")
    
    # Final Training (on 80% of training data)
    model.fit(X_train, y_train)


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
    print(f"\n- Results -")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1-Score: {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, probs):.4f}")

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(f"{name}", fontsize=9)



if __name__ == "__main__":
    # Load processed datasets
    datasets = [
        (load_processed_pima(), "Outcome", "PIMA Diabetes"),
        (load_processed_heart(), "target", "Heart Disease"),
        (load_processed_nhanes(), "target", "NHANES CVD")
    ]

    # Window for Confusion Matrices (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.6, wspace=0.4)


    for row, (df, target_col, dataset_title) in enumerate(datasets):
        print(f"\n{'='*43}\nStarting: {dataset_title}\n{'='*43}")   # Current dataset
        
        X = df.drop(columns=[target_col])   # Features

        # Drop to prevent data leakage
        if dataset_title == "NHANES CVD":   # Remove heart diseases
            leaky_cols = ['Stroke', 'Coronary', 'Angina', 'Congestive', 'Heart_attack']
            X = X.drop(columns=[c for c in leaky_cols if c in X.columns])
          

        y = df[target_col]  # Labels

        # Models
        models = [
            (LogisticRegression(max_iter=1000), "Logistic Regression"),
            (RandomForestClassifier(random_state=42), "Random Forest"),
            (SVC(probability=True, random_state=42), "SVM")
        ]

        # Loop through each model & train
        for col, (model_obj, model_name) in enumerate(models):
            full_name = f"{dataset_title} - {model_name}"   # For clarity
            current_ax = axes[row, col]

            train_and_verify(model_obj, X, y, full_name, current_ax)

    print("\nTraining complete. Opening all Confusion Matrices...")
    plt.show()