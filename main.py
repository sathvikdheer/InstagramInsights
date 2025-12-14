import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import load_data, clean_and_preprocess_data
from src.model_training import train_all_models
from src.evaluation import evaluate_models, plot_feature_importance, plot_error_distribution
import pandas as pd


RANDOM_STATE = 42

def main():
    print("Starting Predictive Analytics Project...")
    
    
    data_path = 'Instagram_Analytics_PowerBI_Ready.csv'
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please place the file in the project root.")
        return
        
    df = load_data(data_path)
    if df is None:
        return

    
    print("\n--- Phase 1: Data Processing ---")
    X_train, X_test, y_train, y_test, feature_names = clean_and_preprocess_data(df)
    
    if X_train is None:
        print("Data processing failed.")
        return

    
    print("\n--- Phase 2: Model Development ---")
    models = train_all_models(X_train, y_train)

    
    print("\n--- Phase 3: Model Evaluation ---")
    results_df = evaluate_models(models, X_test, y_test)
    
    
    print("\n--- Phase 4: Analysis & Interpretation ---")
    best_model_name = results_df.iloc[0]['Model']
    print(f"Best performing model: {best_model_name}")
    
   
    best_model = models[best_model_name]
    plot_feature_importance(best_model, feature_names)
    
    
    if 'Random Forest' in models and best_model_name != 'Random Forest':
        print("Generating feature importance for Random Forest (for analysis)...")
        plot_feature_importance(models['Random Forest'], feature_names, filename='rf_feature_importance.png')


    plot_error_distribution(models, X_test, y_test, results_df)
    
    print("\nProject execution completed successfully.")

if __name__ == "__main__":
    main()
