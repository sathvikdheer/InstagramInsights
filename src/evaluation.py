import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

def evaluate_models(models, X_test, y_test):
   -
    results = []
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MSE': mse,
            'R2 Score': r2
        })
        
    results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)
    
    print("\n--- Model Evaluation Results ---")
    print(results_df)
    return results_df

def plot_feature_importance(model, feature_names, filename='feature_importance.png'):
   
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title(f"Feature Importance ({type(model).__name__})")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Feature importance plot saved to {filename}")
        plt.close()
    else:
        print(f"Model {type(model).__name__} does not support feature importance plotting.")

def plot_error_distribution(models, X_test, y_test, results_df, filename='error_distribution.png'):
   s the distribution of prediction errors (residuals) for the best model.
    
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    y_pred = best_model.predict(X_test)
    errors = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='k', alpha=0.7)
    plt.title(f"Error Distribution (Residuals) - {best_model_name}")
    plt.xlabel("Prediction Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    print(f"Error distribution plot saved to {filename}")
    plt.close()
 