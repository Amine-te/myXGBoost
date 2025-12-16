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
    'random_state': 42,
    'n_jobs': 1  # Force single thread to avoid conflicts/deadlocks
}

# Create results directory
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(os.path.dirname(script_dir), 'validation_results')
os.makedirs(results_dir, exist_ok=True)

def plot_importance_comparison(feature_names, my_imp, off_imp, title, filename):
    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, my_imp, width, label='MyXGBoost')
    rects2 = ax.bar(x + width/2, off_imp, width, label='Official XGBoost')

    ax.set_ylabel('Importance (Gain)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1 * max(np.max(my_imp), np.max(off_imp)))
    ax.legend()

    fig.tight_layout()
    
    # Save to validation_results directory
    output_path = os.path.join(results_dir, filename)
    plt.savefig(output_path)
    print(f"Graph saved to {output_path}")
    plt.close()

def run_regression_comparison():
    print("\n--- 1. Regression Feature Importance (California Housing) ---")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MyXGBoost
    print("Training MyXGBoost...")
    my_model = MyXGBRegressor(**COMMON_PARAMS)
    my_model.fit(X_train, y_train)
    my_imp = my_model.feature_importances_
    
    # Official XGBoost (Exact mode for fair comparison)
    print("Training Official XGBoost...")
    # Use total_gain to match our implementation (sum of gains)
    off_model = XGBRegressor(**COMMON_PARAMS, tree_method='exact', importance_type='total_gain')
    off_model.fit(X_train, y_train)
    off_imp = off_model.feature_importances_
    
    # Normalize official importances to sum to 1 for fair comparison if not already
    if np.sum(off_imp) > 0:
        off_imp = off_imp / np.sum(off_imp)
    
    print("\nFeature Importances Comparison (Sorted by MyXGBoost Importance):")
    print(f"{'Feature':<20} | {'MyXGBoost':<12} | {'Official':<12} | {'Diff':<12}")
    print("-" * 65)
    
    # Sort by MyXGBoost importance descending
    indices = np.argsort(my_imp)[::-1]
    
    for i in indices:
        name = feature_names[i]
        print(f"{name:<20} | {my_imp[i]:.6f}     | {off_imp[i]:.6f}     | {abs(my_imp[i]-off_imp[i]):.6f}")
        
    correlation = np.corrcoef(my_imp, off_imp)[0, 1]
    print(f"\nCorrelation between importance vectors: {correlation:.4f}")
    
    plot_importance_comparison(
        feature_names, 
        my_imp, 
        off_imp, 
        'Feature Importance Comparison (Regression - California Housing)',
        'importance_comparison_regression.png'
    )

def run_classification_comparison():
    print("\n--- 2. Classification Feature Importance (Breast Cancer) ---")
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MyXGBoost
    print("Training MyXGBoost...")
    my_model = MyXGBClassifier(**COMMON_PARAMS)
    my_model.fit(X_train, y_train)
    my_imp = my_model.feature_importances_
    
    # Official XGBoost
    print("Training Official XGBoost...")
    # Use total_gain to match our implementation
    off_model = XGBClassifier(**COMMON_PARAMS, tree_method='exact', importance_type='total_gain', use_label_encoder=False, eval_metric='logloss')
    off_model.fit(X_train, y_train)
    off_imp = off_model.feature_importances_
    
    # Normalize official importances
    if np.sum(off_imp) > 0:
        off_imp = off_imp / np.sum(off_imp)
        
    print("\nFeature Importances Comparison (Top 10 Sorted by MyXGBoost):")
    print(f"{'Feature':<20} | {'MyXGBoost':<12} | {'Official':<12} | {'Diff':<12}")
    print("-" * 65)
    
    # Sort by MyXGBoost importance to show most relevant
    indices = np.argsort(my_imp)[::-1]
    for i in indices[:10]:
        name = feature_names[i]
        print(f"{name:<20} | {my_imp[i]:.6f}     | {off_imp[i]:.6f}     | {abs(my_imp[i]-off_imp[i]):.6f}")
        
    correlation = np.corrcoef(my_imp, off_imp)[0, 1]
    print(f"\nCorrelation between importance vectors: {correlation:.4f}")

    # Plot top 10 features
    top_indices = indices[:10]
    plot_importance_comparison(
        feature_names[top_indices], 
        my_imp[top_indices], 
        off_imp[top_indices], 
        'Top 10 Feature Importance Comparison (Classification - Breast Cancer)',
        'importance_comparison_classification.png'
    )

if __name__ == "__main__":
    run_regression_comparison()
    run_classification_comparison()