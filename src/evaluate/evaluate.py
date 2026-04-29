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

def explain_xai(model, X_train, X_test, feature_names, name, run_dir):
    base_filename = name.replace(" ", "_").replace("-", "_")

    # --- LIME Implementation ---
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['Negative', 'Positive'],
        mode='classification'
    )
    exp_lime = explainer_lime.explain_instance(X_test.values[0], model.predict_proba, num_features=5)
    
    fig_lime = exp_lime.as_pyplot_figure()
    plt.title(f"LIME Explanation: {name}")
    plt.tight_layout()
    fig_lime.savefig(os.path.join(run_dir, f"{base_filename}_LIME.png"))
    plt.close(fig_lime) 

    # --- SHAP Implementation ---
    background = shap.kmeans(X_train, 10)
    explainer_shap = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer_shap.shap_values(X_test.iloc[0:1, :])
    
    # Handle different SHAP output formats[cite: 1]
    if isinstance(shap_values, list):
        display_values, base_value = shap_values[1], explainer_shap.expected_value[1]
    elif len(shap_values.shape) == 3:
        display_values, base_value = shap_values[0, :, 1], explainer_shap.expected_value[1]
    else:
        display_values, base_value = shap_values, explainer_shap.expected_value

    plt.figure() 
    shap.force_plot(base_value, display_values, X_test.iloc[0:1, :], 
                    feature_names=feature_names, matplotlib=True, show=False)
    plt.title(f"SHAP Force Plot: {name}")
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
