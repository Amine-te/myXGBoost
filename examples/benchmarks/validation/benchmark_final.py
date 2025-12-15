import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

# Add src folder to path to import myXGBoost
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

try:
    from myXGBoost.estimators.classifier import XGBClassifier as MyXGBClassifier
    from myXGBoost.estimators.regressor import XGBRegressor as MyXGBRegressor
except ImportError as e:
    print("Error importing MyXGBoost. Ensure the project structure is correct.")
    print(f"Detail: {e}")
    sys.exit(1)

# Global configuration
PARAMS = {
    'n_estimators': 20,
    'max_depth': 3,
    'learning_rate': 0.1
}

results_data = []

def run_classification_test():
    print("\n--- 1. Classification Test (Breast Cancer) ---")
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ("MyXGBoost", MyXGBClassifier(**PARAMS)),
        ("XGBoost (Official)", XGBClassifier(**PARAMS, eval_metric='logloss', use_label_encoder=False)),
        ("Sklearn GBM", GradientBoostingClassifier(**PARAMS))
    ]

    scores = {}
    
    for name, model in models:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"{name}: Accuracy = {acc:.4f}, Time = {train_time:.4f}s")
        
        results_data.append({
            "Test": "Classification (Cancer)",
            "Model": name,
            "Metric Name": "Accuracy",
            "Metric Value": acc,
            "Time (s)": train_time
        })
        scores[name] = acc
    return scores

def run_regression_test():
    print("\n--- 2. Regression Test (California Housing - Subset 1000) ---")
    # Load and subset the first 1000 samples
    data = fetch_california_housing()
    X, y = data.data[:1000], data.target[:1000]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ("MyXGBoost", MyXGBRegressor(**PARAMS)),
        ("XGBoost (Official)", XGBRegressor(**PARAMS)),
        ("Sklearn GBM", GradientBoostingRegressor(**PARAMS))
    ]

    scores = {}

    for name, model in models:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        
        print(f"{name}: MSE = {mse:.4f}, Time = {train_time:.4f}s")
        
        results_data.append({
            "Test": "Regression (Housing)",
            "Model": name,
            "Metric Name": "MSE",
            "Metric Value": mse,
            "Time (s)": train_time
        })
        scores[name] = mse
    return scores

def run_stress_test():
    print("\n--- 3. Stress Test (Synthetic 5000 samples, 20 features) ---")
    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ("MyXGBoost", MyXGBClassifier(**PARAMS)),
        ("XGBoost (Official)", XGBClassifier(**PARAMS, eval_metric='logloss', use_label_encoder=False)),
        ("Sklearn GBM", GradientBoostingClassifier(**PARAMS))
    ]

    times = {}

    for name, model in models:
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        print(f"{name}: Time = {train_time:.4f}s")
        
        results_data.append({
            "Test": "Stress Test",
            "Model": name,
            "Metric Name": "Time Only",
            "Metric Value": 0, # Not relevant here
            "Time (s)": train_time
        })
        times[name] = train_time
    return times

def generate_dashboard(cls_scores, reg_scores, stress_times):
    print("\n--- Generating Dashboard ---")
    
    # Style and colors
    plt.style.use('ggplot')
    model_colors = {
        "MyXGBoost": '#3498db',        # Blue
        "XGBoost (Official)": '#e67e22', # Orange
        "Sklearn GBM": '#2ecc71'       # Green
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Benchmark: MyXGBoost vs XGBoost vs Sklearn', fontsize=20, fontweight='bold', color='#333333')

    def plot_subplot(ax, data, title, ylabel, is_time=False):
        names = list(data.keys())
        values = list(data.values())
        colors = [model_colors.get(n, '#333333') for n in names]
        
        bars = ax.bar(names, values, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Titles and labels
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Grid
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        
        # Rotate x labels
        ax.tick_params(axis='x', rotation=15)
        
        # Values on bars
        max_val = max(values) if values else 1
        ax.set_ylim(0, max_val * 1.15)
        
        for bar in bars:
            height = bar.get_height()
            label = f"{height:.2f}s" if is_time else f"{height:.4f}"
            ax.text(bar.get_x() + bar.get_width()/2., height + (max_val * 0.02),
                    label, ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

    # Plot 1: Classification Accuracy
    plot_subplot(axes[0], cls_scores, 'Classification Accuracy\n(Higher is better)', 'Accuracy')

    # Plot 2: Regression MSE
    plot_subplot(axes[1], reg_scores, 'Regression MSE\n(Lower is better)', 'Mean Squared Error')

    # Plot 3: Stress Test Time
    plot_subplot(axes[2], stress_times, 'Training Time - Stress Test\n(Lower is better)', 'Seconds', is_time=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig('benchmark_dashboard.png', dpi=300)
    print("Dashboard saved as 'benchmark_dashboard.png'")

if __name__ == "__main__":
    # Run tests
    cls_res = run_classification_test()
    reg_res = run_regression_test()
    stress_res = run_stress_test()

    # Display Table
    df = pd.DataFrame(results_data)
    print("\n=== DETAILED RESULTS ===")
    # Reorder columns for readability
    print(df[['Test', 'Model', 'Metric Name', 'Metric Value', 'Time (s)']].to_string(index=False))

    # Generate Graph
    generate_dashboard(cls_res, reg_res, stress_res)
