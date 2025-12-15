import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier

# Add src folder to path to import myXGBoost
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

try:
    from myXGBoost.estimators.regressor import XGBRegressor as MyXGBRegressor
    from myXGBoost.estimators.classifier import XGBClassifier as MyXGBClassifier
except ImportError as e:
    print("Error importing MyXGBoost. Ensure the project structure is correct.")
    print(f"Detail: {e}")
    sys.exit(1)

# Common configuration
COMMON_PARAMS = {
    'n_estimators': 50,
    'max_depth': 3,
    'learning_rate': 0.1,
    'reg_lambda': 1.0,
}

def run_regression_comparison():
    print("\n--- 1. Regression Comparison (California Housing) ---")
    data = fetch_california_housing()
    X, y = data.data[:500], data.target[:500]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MyXGBoost
    my_model = MyXGBRegressor(**COMMON_PARAMS)
    my_model.fit(X_train, y_train)
    my_preds = my_model.predict(X_test)
    
    # Official XGBoost (Exact mode)
    off_model = XGBRegressor(**COMMON_PARAMS, tree_method='exact')
    off_model.fit(X_train, y_train)
    off_preds = off_model.predict(X_test)
    
    return off_preds, my_preds

def run_classification_comparison():
    print("\n--- 2. Classification Comparison (Breast Cancer) ---")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MyXGBoost
    my_model = MyXGBClassifier(**COMMON_PARAMS)
    my_model.fit(X_train, y_train)
    # Probabilities of the positive class (1)
    my_probs = my_model.predict_proba(X_test)[:, 1]
    
    # Official XGBoost (Exact mode)
    # Note: use_label_encoder=False to avoid warnings on recent versions
    off_model = XGBClassifier(**COMMON_PARAMS, tree_method='exact', use_label_encoder=False, eval_metric='logloss')
    off_model.fit(X_train, y_train)
    off_probs = off_model.predict_proba(X_test)[:, 1]
    
    return off_probs, my_probs

def plot_results(reg_data, cls_data):
    print("\n--- Generating Graph ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Utility function for plotting
    def plot_scatter(ax, x, y, title_prefix):
        # Calculate correlation
        corr = np.corrcoef(x, y)[0, 1]
        mae = np.mean(np.abs(x - y))
        
        ax.scatter(x, y, alpha=0.5, c='#3498db', edgecolors='k', s=40)
        
        # Diagonal line
        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        padding = (max_val - min_val) * 0.05
        ax.plot([min_val - padding, max_val + padding], 
                [min_val - padding, max_val + padding], 
                'r--', linewidth=2, label='Identity (y=x)')
        
        ax.set_xlabel('Official XGBoost (Exact)', fontsize=12)
        ax.set_ylabel('MyXGBoost', fontsize=12)
        ax.set_title(f"{title_prefix}\nPearson: {corr:.6f} | MAE: {mae:.6f}", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # Plot 1: Regression
    plot_scatter(axes[0], reg_data[0], reg_data[1], "Regression: Raw Predictions")
    
    # Plot 2: Classification
    plot_scatter(axes[1], cls_data[0], cls_data[1], "Classification: Probabilities (Class 1)")
    
    plt.tight_layout()
    output_file = 'correlation_complete.png'
    plt.savefig(output_file, dpi=300)
    print(f"Image saved: {output_file}")

if __name__ == "__main__":
    # Exécution des tests
    reg_results = run_regression_comparison()
    cls_results = run_classification_comparison()
    
    # Génération visuelle
    plot_results(reg_results, cls_results)
