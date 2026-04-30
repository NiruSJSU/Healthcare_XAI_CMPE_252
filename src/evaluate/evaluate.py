import os
import pandas as pd
import joblib
from datetime import datetime
from src.data.dataio import load_processed_pima, load_processed_heart, load_processed_nhanes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular

def safe_inverse(scaler, data): # Inverse transformation
    import numpy as np
    try:
        return scaler.inverse_transform(data)
    except ValueError:  # If mismatching then adds dummy zeros
        n_features = data.shape[1]
        padded = np.zeros((data.shape[0], scaler.n_features_in_))
        padded[:, :n_features] = data.values if hasattr(data, 'values') else data

        return scaler.inverse_transform(padded)[:, :n_features]
    

def explain_xai(model, X_train, X_test, feature_names, name, run_dir):
    base_filename = name.replace(" ", "_").replace("-", "_")

    # Scalar path choosen based off dataset name
    if "PIMA" in name:
        scaler_path = os.path.join(models_dir, 'pima_scaler.joblib')
    elif "Heart" in name:
        scaler_path = os.path.join(models_dir, 'heart_scaler.joblib')
    else:
        scaler_path = os.path.join(models_dir, 'nhanes_scaler.joblib')
    
    scaler = joblib.load(scaler_path)
    
    # Scaled instances back to clinical units
    raw_values = safe_inverse(scaler, X_test.iloc[0:1])[0]


    # --- LIME Implementation ---

    # Training set back to raw units
    X_train_raw = safe_inverse(scaler, X_train)

    
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_raw, 
        feature_names=feature_names,
        class_names=['No Disease', 'Disease'],
        mode='classification',
        discretize_continuous=True
    )

    exp_lime = explainer_lime.explain_instance(
        raw_values, 
        model.predict_proba, 
        num_features=10
    )

    fig_lime = exp_lime.as_pyplot_figure()
    fig_lime.set_size_inches(9, 5)
    ax_lime = fig_lime.gca()

    # Calculate values
    prediction_prob = model.predict_proba(X_test.iloc[0:1])[0][1]
    diagnosis = "NO DISEASE" if prediction_prob < 0.5 else "DISEASE"
    conf = (1 - prediction_prob if diagnosis == "NO DISEASE" else prediction_prob) * 100

    # Plain text diagnosis at top left
    fig_lime.text(0.02, 0.95, f"RESULT: {diagnosis}\nCONFIDENCE: {conf:.1f}%", 
                 transform=fig_lime.transFigure, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9),
                 fontsize=11, fontweight='bold')

    plt.title(f"LIME Explanation: {name}", pad=20)
    plt.tight_layout(rect=[0.1, 0, 1, 0.95])
    fig_lime.savefig(os.path.join(run_dir, f"{base_filename}_LIME.png"), bbox_inches='tight')
    plt.close(fig_lime)

    # --- SHAP Implementation ---
    background = shap.kmeans(X_train, 10)
    explainer_shap = shap.KernelExplainer(model.predict_proba, background)
    
    shap_values = explainer_shap.shap_values(X_test.iloc[0:1, :])
    
    # Handling different SHAP output formats
    if isinstance(shap_values, list):
        display_values = shap_values[1][0] # 1D array for first instance
        base_value = explainer_shap.expected_value[1]
    elif len(shap_values.shape) == 3:   # Incase shape is: (instances, features, classes)
        display_values = shap_values[0, :, 1]
        base_value = explainer_shap.expected_value[1]
    else:
        display_values = shap_values[0]
        base_value = explainer_shap.expected_value

    
    explanation = shap.Explanation(
        values=display_values, 
        base_values=base_value, 
        data=raw_values,    # Use real units
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=10, show=False)

    # Plain text diagnosis at top left
    plt.text(0.01, 0.95, f"RESULT: {diagnosis}\nCONFIDENCE: {conf:.1f}%", 
             transform=plt.gcf().transFigure, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=12, fontweight='bold')



    plt.title(f"SHAP Waterfall: {name}")
    plt.tight_layout(rect=[0.1, 0, 1, 0.95])
    plt.savefig(os.path.join(run_dir, f"{base_filename}_SHAP.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Robust Pathing for Makefile execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models", "trained_models"))
    results_base_dir = os.path.join(script_dir, "results")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_path = os.path.join(results_base_dir, f"run_{timestamp}")
    os.makedirs(run_folder_path, exist_ok=True)

    datasets = [
        (load_processed_pima(), "Outcome", "PIMA Diabetes"),
        (load_processed_heart(), "target", "Heart Disease"),
        (load_processed_nhanes(), "target", "NHANES CVD")
    ]

    # These must match the names defined in train.py
    model_names = ["Logistic Regression", "Random Forest", "SVM", "MLP", "LDA"]

    for df, target_col, dataset_title in datasets:
        print(f"\nProcessing Dataset: {dataset_title}")
        df = df.drop_duplicates()
        X = df.drop(columns=[target_col])

        # Feature cleaning to match training state
        if dataset_title == "NHANES CVD":
            leaky_cols = ['Stroke', 'Coronary', 'Angina', 'Congestive', 'Heart_attack']
            X = X.drop(columns=[c for c in leaky_cols if c in X.columns])
        if dataset_title == "Heart Disease" and 'cp' in X.columns:
            X = X.drop(columns=['cp'])

        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        for m_name in model_names:
            full_name = f"{dataset_title} - {m_name}"
            # Matches the 'safe_name' logic in the updated train.py
            file_name = f"{full_name.replace(' ', '_').replace('-', '_')}.joblib"
            model_path = os.path.join(models_dir, file_name)
            
            if os.path.exists(model_path):
                print(f"-> Loading: {file_name}")
                model = joblib.load(model_path)
                explain_xai(model, X_train, X_test, X.columns.tolist(), full_name, run_folder_path)
            else:
                print(f"!! Missing: {model_path}")

    print(f"\nAll plots saved to: {run_folder_path}")
