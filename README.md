# myXGBoost

myXGBoost is a high-performance, modular implementation of the XGBoost algorithm written entirely in Python. Designed for research and educational purposes, this project strictly adheres to the mathematical principles outlined in the original XGBoost paper, offering a transparent and modifiable codebase without sacrificing correctness.

## Project Overview

The primary goal of this project is to provide a clean, "from-scratch" implementation that bridges the gap between theoretical understanding and practical application. Unlike the official C++ implementation, myXGBoost is written in pure Python using NumPy for aggressive vectorization, making the internal logic accessible for inspection and modification.

The implementation has been rigorously validated against the official XGBoost library, achieving a Pearson correlation of 1.000 and a Mean Absolute Error (MAE) of 0.000 on regression tasks using exact split finding.

## Key Features

- **Mathematical Exactness**: Proven equivalence to the official XGBoost implementation (exact tree method).
- **High Performance**: Utilizes advanced NumPy vectorization techniques for efficient split finding and tree construction.
- **Scikit-Learn API**: Follows standard design patterns (`fit`, `predict`, `score`) for seamless integration with the Python data science ecosystem.
- **Comprehensive Functionality**:
  - Exact and Approximate (Histogram-based) split finding algorithms.
  - Built-in support for Early Stopping to prevent overfitting.
  - Gain-based Feature Importance calculation.
  - Support for custom Loss Functions.
  - L2 Regularization (`reg_lambda`).
  - Weighted training sample support.
  - Multiclass classification via Softmax.

## Installation

To install the package, clone the repository and install it in editable mode using pip:

```bash
pip install -e .
```

## Usage

The library provides `XGBRegressor` and `XGBClassifier` classes that mimic the interface of scikit-learn estimators.

### Regression Example

```python
import numpy as np
from myXGBoost import XGBRegressor

# Generate synthetic data
X = np.random.rand(100, 5)
y = X @ np.array([1.5, -2.0, 3.0, 0.5, -1.0]) + np.random.normal(0, 0.1, 100)

# Initialize and train
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    reg_lambda=1.0,
    verbose=True
)
model.fit(X, y)

# Predict and inspect importance
preds = model.predict(X)
print("Feature Importances:", model.feature_importances_)
```

### Classification Example

```python
from myXGBoost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Initialize and train
clf = XGBClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=4
)
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

## Validation and Benchmarks

This repository includes a comprehensive suite of tests to verify both mathematical correctness and performance stability.

- **Mathematical Validation**: The script `examples/benchmarks/validation/test_correlation_complete.py` performs a direct comparison with the official XGBoost library to ensure prediction equivalence.
- **Performance Benchmarking**: The script `examples/benchmarks/synthetic/run_benchmark.py` evaluates training time and accuracy metrics under various load conditions.

To run the full test suite:

```bash
pytest tests/
```

## Project Structure

The codebase is organized to separate core logic from examples and tests:

- `src/myXGBoost/booster/`: Core Gradient Boosting engine.
- `src/myXGBoost/trees/`: Vectorized tree building and split finding logic.
- `src/myXGBoost/loss/`: Implementation of objective functions (MSE, LogLoss, Softmax).
- `src/myXGBoost/estimators/`: Scikit-learn compatible wrapper classes.
- `examples/benchmarks/`: Validation scripts and performance tests.
- `tests/`: Unit tests covering robustness, serialization, and logic.

## Presentation
https://docs.google.com/presentation/d/1ni_zggvNlOWF5LK8n70kaI92uP1uFQ9bfEPVZN_lVEA/edit?usp=sharing
