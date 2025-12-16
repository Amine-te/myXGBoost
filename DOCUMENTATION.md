# myXGBoost: Gradient Boosting Implementation

**A Python implementation of XGBoost-style gradient boosting for regression and classification following scikit-learn API conventions.**

## Overview

myXGBoost is a from-scratch implementation of the XGBoost (eXtreme Gradient Boosting) algorithm, designed to provide an educational and production-ready gradient boosting framework. The implementation follows scikit-learn's API design patterns, making it familiar and easy to integrate into existing machine learning pipelines.

**When to use myXGBoost:**
- When you need a powerful ensemble learning method for regression or classification tasks
- When your data contains complex non-linear relationships
- When you want fine-grained control over regularization and tree construction
- When you need a scikit-learn-compatible gradient boosting implementation with XGBoost-style features

---

## Mathematical Overview

### Gradient Boosting Fundamentals

Gradient boosting is an ensemble learning technique that builds a predictive model as a sum of weak learners (typically decision trees). The model is constructed in an additive manner:

$$\hat{y}_i = \sum_{k=0}^{K} f_k(x_i)$$

where $f_k$ represents individual trees, and $K$ is the total number of boosting rounds.

### Second-Order Optimization

Unlike traditional gradient boosting that only uses first-order gradients, XGBoost employs **second-order Taylor expansion** of the loss function, incorporating both gradients and Hessians:

**At each boosting iteration:**
1. Compute first-order gradients: $g_i = \frac{\partial L(y_i, \hat{y}_i)}{\partial \hat{y}_i}$
2. Compute second-order gradients (Hessians): $h_i = \frac{\partial^2 L(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$
3. Build a new tree that minimizes the objective function using both $g$ and $h$
4. Update predictions: $\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)$

where $\eta$ is the learning rate (shrinkage parameter).

### Regularization and Overfitting Control

The objective function at each iteration includes regularization terms:

$$\text{Obj}^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)$$

where $\Omega(f)$ penalizes model complexity through:
- **L2 regularization on leaf weights** ($\lambda$)
- **Minimum loss reduction** ($\gamma$) required to make splits
- **Minimum child weight** (minimum sum of Hessians in child nodes)

### Residual Learning

The model learns to correct residuals (errors) from previous iterations:
- Each new tree fits the negative gradients of the loss function
- This effectively learns the "mistakes" of the ensemble so far
- Predictions improve iteratively as more trees are added

---

## Installation

### Requirements

- **Python**: ≥ 3.8
- **NumPy**: ≥ 1.20.0
- **SciPy**: ≥ 1.7.0 (optional, for advanced operations)

### Installation from Source

```bash
# Clone the repository
git clone https://github.com/Amine-te/myXGBoost.git
cd myXGBoost

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```python
import myXGBoost
from myXGBoost.estimators import XGBRegressor, XGBClassifier

print(myXGBoost.__version__)
```

---

## Quick Start

### Regression Example

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from myXGBoost.estimators.regressor import XGBRegressor

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
```

### Classification Example

```python
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, log_loss
from myXGBoost.estimators.classifier import XGBClassifier

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.4f}")
print(f"Log Loss: {logloss:.4f}")
```

---

## API Reference

### XGBRegressor

```python
class myXGBoost.estimators.regressor.XGBRegressor(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=6,
    min_child_weight=1.0,
    gamma=0.0,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=None,
    verbose=False
)
```

**XGBoost-style gradient boosting regressor with second-order optimization.**

#### Parameters

**learning_rate** : *float, default=0.1*
- Boosting learning rate (also known as eta or shrinkage)
- Controls the step size at each boosting iteration
- **Range**: (0, 1]
- **Effect**: 
  - Lower values (0.01-0.1) require more trees but often generalize better
  - Higher values (0.3-1.0) train faster but may overfit
  - Typical values: 0.01, 0.05, 0.1, 0.3
- **Bias/Variance**: Lower learning rate → higher bias, lower variance

**n_estimators** : *int, default=100*
- Number of boosting rounds (trees to build sequentially)
- **Range**: [1, ∞)
- **Effect**:
  - More trees → more complex model, longer training time
  - Too many trees may lead to overfitting without proper regularization
  - Typical values: 50-500 for small datasets, 100-1000 for larger datasets
- **Bias/Variance**: More estimators → lower bias, potential for higher variance

**max_depth** : *int, default=6*
- Maximum depth of each decision tree
- Controls the complexity of individual trees
- **Range**: [1, ∞), typically [3, 10]
- **Effect**:
  - Deeper trees capture more complex interactions
  - Shallow trees (3-4) prevent overfitting, deeper trees (6-10) for complex patterns
  - Very deep trees (>12) often overfit
- **Bias/Variance**: Greater depth → lower bias, higher variance

**min_child_weight** : *float, default=1.0*
- Minimum sum of instance weights (Hessian) needed in a child node
- Acts as a regularization parameter preventing trees from splitting on very small groups
- **Range**: [0, ∞)
- **Effect**:
  - Higher values → more conservative splits, smoother decision boundaries
  - Lower values allow more aggressive splitting
  - Typical values: 1 (default), 3-10 for noisy data
- **Bias/Variance**: Higher values → higher bias, lower variance

**gamma** : *float, default=0.0*
- Minimum loss reduction required to make a further partition on a leaf node
- Also known as minimum split loss
- **Range**: [0, ∞)
- **Effect**:
  - Acts as a regularization parameter
  - Higher values → fewer splits, simpler trees
  - 0 means no constraint (split whenever gain > 0)
  - Typical values: 0 (default), 0.1-1.0 for regularization
- **Bias/Variance**: Higher gamma → higher bias, lower variance

**subsample** : *float, default=1.0*
- Fraction of training samples used for fitting each tree
- Implements stochastic gradient boosting
- **Range**: (0, 1]
- **Effect**:
  - < 1.0 introduces randomness, reduces overfitting, speeds up training
  - 0.5-0.8 often works well for large datasets
  - 1.0 uses all data (no subsampling)
- **Bias/Variance**: Lower subsample → higher bias, lower variance

**colsample_bytree** : *float, default=1.0*
- Fraction of features used when constructing each tree
- Feature subsampling per tree
- **Range**: (0, 1]
- **Effect**:
  - Reduces correlation between trees
  - Prevents overfitting on datasets with many features
  - 0.5-0.9 often effective for high-dimensional data
- **Bias/Variance**: Lower colsample → higher bias, lower variance

**random_state** : *int or None, default=None*
- Random seed for reproducibility
- Controls randomness in:
  - Row subsampling (if subsample < 1.0)
  - Column subsampling (if colsample_bytree < 1.0)
- **Effect**: Set to an integer for reproducible results

**verbose** : *bool, default=False*
- Whether to print progress information during training
- **Effect**: If True, prints iteration information and loss values

#### Attributes (Set After Fitting)

**n_features_in_** : *int*
- Number of features seen during fit

**feature_names_in_** : *ndarray of shape (n_features_in_,)*
- Names of features seen during fit (only if input has feature names)

**booster_** : *GradientBooster*
- The underlying gradient boosting model containing:
  - `trees`: List of trained DecisionTree objects
  - `initial_prediction`: Initial constant prediction
  - `eval_results`: List of evaluation metrics per iteration (if eval_set provided)
  - `best_iteration_`: Best iteration index (if early stopping used)

#### Methods

##### fit(X, y, sample_weight=None, eval_set=None, eval_metric=None, verbose=None)

Fit the gradient boosting model to training data.

**Parameters:**

- **X** : *array-like of shape (n_samples, n_features)*
  - Training input samples

- **y** : *array-like of shape (n_samples,)*
  - Target values (real numbers for regression)

- **sample_weight** : *array-like of shape (n_samples,), default=None*
  - Individual weights for each sample (currently not fully supported)

- **eval_set** : *list of tuples (X_val, y_val), default=None*
  - Validation sets for monitoring performance and early stopping
  - Example: `[(X_val, y_val)]` or `[(X_train, y_train), (X_val, y_val)]`

- **eval_metric** : *str or callable, default=None*
  - Metric to evaluate on validation sets
  - Can be a metric object with `.score()` method or a callable
  - Example: `lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)`

- **verbose** : *bool, default=None*
  - Override instance verbose setting for this fit call
  - If None, uses `self.verbose`

**Returns:**
- **self** : *XGBRegressor*
  - Fitted estimator

**Example:**
```python
from myXGBoost.estimators.regressor import XGBRegressor
from sklearn.metrics import mean_squared_error

model = XGBRegressor(n_estimators=100, random_state=42)

# Basic fit
model.fit(X_train, y_train)

# Fit with validation monitoring
eval_metric = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)], 
          eval_metric=eval_metric,
          verbose=True)
```

##### predict(X)

Predict target values for input samples.

**Parameters:**

- **X** : *array-like of shape (n_samples, n_features)*
  - Input samples for prediction

**Returns:**
- **y_pred** : *ndarray of shape (n_samples,)*
  - Predicted target values

**Raises:**
- **ValueError** : If model is not fitted or feature count mismatch

**Example:**
```python
y_pred = model.predict(X_test)
```

##### score(X, y, sample_weight=None)

Return the coefficient of determination (R²) of the prediction.

**Parameters:**

- **X** : *array-like of shape (n_samples, n_features)*
  - Test samples

- **y** : *array-like of shape (n_samples,)*
  - True values for X

- **sample_weight** : *array-like of shape (n_samples,), default=None*
  - Sample weights

**Returns:**
- **score** : *float*
  - R² score of `self.predict(X)` versus `y`
  - Best possible score is 1.0, can be negative for poor predictions

**Example:**
```python
r2_score = model.score(X_test, y_test)
print(f"R² Score: {r2_score:.4f}")
```

##### get_params(deep=True)

Get parameters for this estimator.

**Parameters:**

- **deep** : *bool, default=True*
  - If True, return parameters for this estimator and contained subobjects

**Returns:**
- **params** : *dict*
  - Parameter names mapped to their values

**Example:**
```python
params = model.get_params()
print(params)
# {'learning_rate': 0.1, 'n_estimators': 100, ...}
```

##### set_params(\*\*params)

Set the parameters of this estimator.

**Parameters:**

- **\*\*params** : *dict*
  - Estimator parameters to set

**Returns:**
- **self** : *XGBRegressor*
  - Estimator instance

**Raises:**
- **ValueError** : If parameter name is invalid

**Example:**
```python
model.set_params(learning_rate=0.05, n_estimators=200)
```

---

### XGBClassifier

```python
class myXGBoost.estimators.classifier.XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=6,
    min_child_weight=1.0,
    gamma=0.0,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=None,
    verbose=False
)
```

**XGBoost-style gradient boosting classifier with second-order optimization.**

Supports both binary and multiclass classification using logistic loss (binary) or softmax loss (multiclass).

#### Parameters

Parameters are identical to `XGBRegressor` (see above).

#### Attributes (Set After Fitting)

**n_features_in_** : *int*
- Number of features seen during fit

**feature_names_in_** : *ndarray of shape (n_features_in_,)*
- Names of features seen during fit (only if input has feature names)

**classes_** : *ndarray of shape (n_classes,)*
- Unique class labels in training data

**n_classes_** : *int*
- Number of classes (2 for binary, >2 for multiclass)

**booster_** : *GradientBooster*
- The underlying gradient boosting model

#### Methods

##### fit(X, y, sample_weight=None, eval_set=None, eval_metric=None, verbose=None)

Fit the gradient boosting classifier to training data.

**Parameters:**

- **X** : *array-like of shape (n_samples, n_features)*
  - Training input samples

- **y** : *array-like of shape (n_samples,)*
  - Target class labels

- **sample_weight** : *array-like of shape (n_samples,), default=None*
  - Individual weights for each sample

- **eval_set** : *list of tuples (X_val, y_val), default=None*
  - Validation sets for monitoring performance

- **eval_metric** : *str or callable, default=None*
  - Metric to evaluate on validation sets

- **verbose** : *bool, default=None*
  - Override instance verbose setting

**Returns:**
- **self** : *XGBClassifier*
  - Fitted estimator

**Example:**
```python
from myXGBoost.estimators.classifier import XGBClassifier

model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

##### predict(X)

Predict class labels for input samples.

**Parameters:**

- **X** : *array-like of shape (n_samples, n_features)*
  - Input samples for prediction

**Returns:**
- **y_pred** : *ndarray of shape (n_samples,)*
  - Predicted class labels

**Example:**
```python
y_pred = model.predict(X_test)
```

##### predict_proba(X)

Predict class probabilities for input samples.

**Parameters:**

- **X** : *array-like of shape (n_samples, n_features)*
  - Input samples for prediction

**Returns:**
- **proba** : *ndarray of shape (n_samples, n_classes)*
  - Class probabilities for each sample
  - For binary classification: `[:, 0]` = P(class 0), `[:, 1]` = P(class 1)
  - For multiclass: softmax probabilities for each class

**Example:**
```python
y_proba = model.predict_proba(X_test)
print(f"Probability of class 1: {y_proba[:5, 1]}")
```

##### score(X, y, sample_weight=None)

Return the mean accuracy on the given test data and labels.

**Parameters:**

- **X** : *array-like of shape (n_samples, n_features)*
  - Test samples

- **y** : *array-like of shape (n_samples,)*
  - True labels for X

- **sample_weight** : *array-like of shape (n_samples,), default=None*
  - Sample weights

**Returns:**
- **score** : *float*
  - Mean accuracy of `self.predict(X)` versus `y`

**Example:**
```python
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Advanced Usage

### Controlling Overfitting

Gradient boosting models can easily overfit if not properly regularized. Use these strategies:

#### 1. Regularization Parameters

```python
# Strong regularization for noisy data
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,      # Smaller learning rate
    max_depth=4,             # Shallower trees
    min_child_weight=5.0,    # Require more samples per leaf
    gamma=1.0,               # Require significant gain to split
    subsample=0.8,           # Row subsampling
    colsample_bytree=0.8,    # Column subsampling
    random_state=42
)
```

#### 2. Learning Rate vs. Number of Trees Trade-off

There's a crucial balance between `learning_rate` and `n_estimators`:

```python
# Strategy 1: Many trees with small learning rate (often best performance)
model_slow = XGBRegressor(n_estimators=500, learning_rate=0.01)

# Strategy 2: Fewer trees with larger learning rate (faster training)
model_fast = XGBRegressor(n_estimators=50, learning_rate=0.3)

# Strategy 3: Moderate approach (good balance)
model_balanced = XGBRegressor(n_estimators=100, learning_rate=0.1)
```

**Rule of thumb**: `learning_rate × n_estimators ≈ constant` for similar final performance

#### 3. Early Stopping

Monitor validation performance and stop when it plateaus:

```python
from sklearn.metrics import mean_squared_error

# Define validation metric (negative MSE for minimization)
eval_metric = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)

# Note: Early stopping support is built into the booster but not fully exposed
# Current usage through eval_set monitoring:
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, verbose=True)
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)], 
          eval_metric=eval_metric)
```

### Effect of Key Hyperparameters

#### Max Depth

```python
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

depths = [2, 4, 6, 8, 10]
train_errors = []
test_errors = []

for depth in depths:
    model = XGBRegressor(n_estimators=100, max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_errors.append(mean_squared_error(y_train, train_pred))
    test_errors.append(mean_squared_error(y_test, test_pred))

plt.plot(depths, train_errors, label='Train Error', marker='o')
plt.plot(depths, test_errors, label='Test Error', marker='s')
plt.xlabel('Max Depth')
plt.ylabel('MSE')
plt.legend()
plt.title('Effect of Max Depth on Model Performance')
plt.show()
```

**Interpretation:**
- Shallow trees (2-3): High bias, low variance, underfitting
- Medium trees (4-6): Good balance, often optimal
- Deep trees (8+): Low bias, high variance, overfitting risk

#### Learning Rate

```python
learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5]
test_scores = []

for lr in learning_rates:
    model = XGBRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    test_scores.append(score)
    print(f"Learning Rate: {lr:.2f}, R² Score: {score:.4f}")
```

**Interpretation:**
- Very low (0.01): Slow convergence, may need more trees
- Low-Medium (0.05-0.1): Good generalization, recommended range
- High (0.3-0.5): Fast convergence but may miss optimal solution

### Subsampling Strategies

```python
# No subsampling (use all data)
model_full = XGBRegressor(subsample=1.0, colsample_bytree=1.0)

# Row subsampling (stochastic gradient boosting)
model_row = XGBRegressor(subsample=0.8, colsample_bytree=1.0)

# Column subsampling (random features per tree)
model_col = XGBRegressor(subsample=1.0, colsample_bytree=0.8)

# Both row and column subsampling (maximum diversity)
model_both = XGBRegressor(subsample=0.8, colsample_bytree=0.8)
```

**Benefits of subsampling:**
- Reduces overfitting
- Speeds up training
- Increases diversity between trees
- Especially helpful for large datasets

---

## Comprehensive Examples

### Example 1: Complete Regression Pipeline

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from myXGBoost.estimators.regressor import XGBRegressor

# Load real-world dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model with optimized hyperparameters
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3.0,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=False
)

# Train model
print("Training model...")
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)
print(f"\nTraining Set:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"  MAE:  {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"  R²:   {r2_score(y_train, y_train_pred):.4f}")

print(f"\nTest Set:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"  MAE:  {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"  R²:   {r2_score(y_test, y_test_pred):.4f}")
```

### Example 2: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Create base model
base_model = XGBRegressor(random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
print("Starting grid search...")
grid_search.fit(X_train, y_train)

# Best parameters
print("\nBest Parameters:")
print(grid_search.best_params_)
print(f"\nBest CV Score (Negative MSE): {grid_search.best_score_:.4f}")

# Evaluate best model
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test R² Score: {test_score:.4f}")
```

### Example 3: Comparison with Scikit-learn

```python
from sklearn.ensemble import GradientBoostingRegressor
from time import time

# Create comparable models
models = {
    'myXGBoost': XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),
    'sklearn GradientBoosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

# Compare performance
results = {}
for name, model in models.items():
    # Time training
    start = time()
    model.fit(X_train, y_train)
    train_time = time() - start
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'train_time': train_time,
        'rmse': rmse,
        'r2': r2
    }

# Print comparison
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Training Time: {metrics['train_time']:.3f}s")
    print(f"  RMSE:          {metrics['rmse']:.4f}")
    print(f"  R² Score:      {metrics['r2']:.4f}")
```

### Example 4: Multiclass Classification

```python
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from myXGBoost.estimators.classifier import XGBClassifier

# Load iris dataset (3 classes)
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Create and train classifier
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nAccuracy: {model.score(X_test, y_test):.4f}")

# Show prediction probabilities for first 5 samples
print("\nPrediction Probabilities (first 5 samples):")
for i in range(5):
    print(f"  Sample {i}: {y_proba[i]}, Predicted: {iris.target_names[y_pred[i]]}")
```

---

## Performance and Limitations

### When myXGBoost Performs Well

✅ **Strengths:**
- **Tabular data**: Excellent for structured/tabular datasets with mixed feature types
- **Non-linear relationships**: Captures complex interactions without manual feature engineering
- **Missing values**: Can handle missing values through split finding (if implemented)
- **Feature importance**: Provides interpretable feature importance measures
- **Regularization**: Built-in regularization prevents overfitting
- **Robust to outliers**: Less sensitive to outliers than linear models

### Known Limitations

⚠️ **Limitations:**
- **High-dimensional sparse data**: Less effective than linear models on very high-dimensional sparse data (e.g., text)
- **Time series**: Not designed for time series; requires careful feature engineering
- **Computational cost**: Slower than linear models, especially with many trees and deep depths
- **Memory usage**: Can be memory-intensive for very large datasets
- **Extrapolation**: Poor at extrapolating beyond training data range
- **Interpretability**: Less interpretable than single decision trees or linear models

### Comparison with Other Implementations

#### vs. Scikit-learn GradientBoostingRegressor

| Feature | myXGBoost | sklearn GradientBoosting |
|---------|-----------|-------------------------|
| Second-order optimization (Hessian) | ✅ Yes | ❌ No (only gradients) |
| Column subsampling | ✅ Yes (`colsample_bytree`) | ❌ No |
| Regularization (gamma) | ✅ Yes | ❌ No |
| Speed | ~Similar | ~Similar |
| API compatibility | ✅ sklearn-compatible | ✅ sklearn native |

**When to use myXGBoost over sklearn:**
- Need column subsampling for feature diversity
- Want second-order optimization for better convergence
- Require XGBoost-style regularization (gamma, min_child_weight)

#### vs. Official XGBoost Library

| Feature | myXGBoost | Official XGBoost |
|---------|-----------|------------------|
| Core algorithm | ✅ Implemented | ✅ Highly optimized |
| Parallel training | ⚠️ Limited | ✅ Full parallelization |
| GPU support | ❌ No | ✅ Yes |
| Distributed training | ❌ No | ✅ Yes (Dask, Spark) |
| Speed | Slower | Much faster |
| Features | Core features | Full feature set |

**When to use official XGBoost:**
- Production environments requiring maximum performance
- Large-scale datasets (>100k samples)
- Need GPU acceleration or distributed training
- Require advanced features (custom objectives, constraints, etc.)

**When to use myXGBoost:**
- Educational purposes (understanding XGBoost internals)
- Small to medium datasets where speed is not critical
- Preference for pure Python implementation
- Need for code customization and experimentation

---

## Reproducibility and Determinism

### Random Seed Behavior

myXGBoost uses random number generation in several places:
- **Row subsampling**: When `subsample < 1.0`
- **Column subsampling**: When `colsample_bytree < 1.0`

To ensure reproducible results:

```python
# Set random_state parameter
model = XGBRegressor(
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42  # Fixed seed
)

# Multiple runs will produce identical results
for i in range(3):
    model_run = XGBRegressor(
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model_run.fit(X_train, y_train)
    pred = model_run.predict(X_test)
    print(f"Run {i+1}: First prediction = {pred[0]:.6f}")
# All runs will print identical values
```

### Deterministic Training

When `random_state` is set and no subsampling is used (`subsample=1.0` and `colsample_bytree=1.0`), training is completely deterministic:

```python
# Fully deterministic setup
model = XGBRegressor(
    n_estimators=100,
    subsample=1.0,          # No row subsampling
    colsample_bytree=1.0,   # No column subsampling
    random_state=42         # Seed (actually not needed here)
)
```

**Note on NumPy Random State:**
The implementation uses `np.random.seed()` internally. For multi-threaded environments, consider using `numpy.random.RandomState` for better isolation (future enhancement).

---

## Algorithm Details

### Split Finding Strategy

myXGBoost implements **hybrid split finding**:

1. **Exact Greedy Algorithm** (for small datasets, n < 10,000):
   - Evaluates all possible split points for each feature
   - Guarantees optimal splits but computationally expensive
   - Time complexity: O(n × features × depth)

2. **Approximate Histogram Algorithm** (for large datasets, n ≥ 10,000):
   - Groups continuous features into discrete bins (histogram)
   - Evaluates only bin boundaries as split candidates
   - Faster with minimal loss in accuracy
   - Time complexity: O(n × features + bins × features × depth)

The algorithm automatically selects the appropriate method based on dataset size (controlled by `exact_threshold=10000` in `GradientBooster`).

### Objective Function

At each boosting iteration, the model minimizes:

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

where:
- $L$ is the loss function (MSE for regression, log loss for classification)
- $f_t$ is the new tree to be learned
- $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$ is the regularization term
- $T$ is the number of leaves
- $w_j$ is the leaf weight

Using second-order Taylor approximation:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$

where $g_i = \frac{\partial L}{\partial \hat{y}_i^{(t-1)}}$ and $h_i = \frac{\partial^2 L}{\partial {\hat{y}_i^{(t-1)}}^2}$

### Leaf Weight Calculation

For each leaf $j$, the optimal weight is:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

where $I_j$ is the set of samples in leaf $j$.

### Split Gain Calculation

The gain from splitting a node into left (L) and right (R) children:

$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

where $G$ and $H$ are the sum of gradients and Hessians respectively.

A split is made only if Gain > 0.

---

## References

### Original Papers

1. **Chen, T., & Guestrin, C. (2016).** XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
   - DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
   - arXiv: [1603.02754](https://arxiv.org/abs/1603.02754)

2. **Friedman, J. H. (2001).** Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189-1232.
   - The foundational paper on gradient boosting

3. **Friedman, J. H. (2002).** Stochastic gradient boosting. *Computational Statistics & Data Analysis*, 38(4), 367-378.
   - Introduces row subsampling for gradient boosting

### Additional Resources

- **Official XGBoost Documentation**: https://xgboost.readthedocs.io/
- **Scikit-learn Ensemble Methods**: https://scikit-learn.org/stable/modules/ensemble.html
- **Understanding XGBoost**: Introduction to Boosted Trees (https://xgboost.readthedocs.io/en/latest/tutorials/model.html)

### Related Algorithms

- **LightGBM**: Microsoft's gradient boosting framework with leaf-wise growth
- **CatBoost**: Yandex's gradient boosting with categorical feature handling
- **AdaBoost**: Adaptive boosting (precursor to gradient boosting)
- **Random Forests**: Parallel ensemble of trees (bagging vs. boosting)

---

## Contributing and Support

### Contributing

Contributions are welcome! Areas for improvement:
- Feature importance calculation
- Early stopping implementation
- Categorical feature handling
- Missing value support
- Parallel tree building
- GPU acceleration

### License

MIT License - See LICENSE file for details

### Authors

- **Amine FARIS**
- **Zouga Mouhcine**
- **Serraji Wiame**
- **El Madani Adam**

### Citation

If you use myXGBoost in academic work, please cite:

```bibtex
@software{myxgboost2024,
  authors = {Faris. Amine, Zouga. Mouhcine, Serraji. Wiame, El Madani. Adam },
  title = {myXGBoost: A Python Implementation of XGBoost},
  year = {2025},
  url = {https://github.com/Amine-te/myXGBoost}
}
```

---

## Appendix: Quick Reference

### Default Hyperparameters

```python
XGBRegressor(
    learning_rate=0.1,          # Learning rate (eta)
    n_estimators=100,            # Number of trees
    max_depth=6,                 # Maximum tree depth
    min_child_weight=1.0,        # Minimum sum of Hessians in child
    gamma=0.0,                   # Minimum split loss
    subsample=1.0,               # Row sampling ratio
    colsample_bytree=1.0,        # Column sampling ratio
    random_state=None,           # Random seed
    verbose=False                # Print progress
)
```

### Common Hyperparameter Ranges

| Parameter | Low Regularization | Medium | High Regularization |
|-----------|-------------------|---------|---------------------|
| `learning_rate` | 0.3 | 0.1 | 0.01 |
| `n_estimators` | 50 | 100 | 500 |
| `max_depth` | 8 | 6 | 3 |
| `min_child_weight` | 0.5 | 1.0 | 5.0 |
| `gamma` | 0.0 | 0.1 | 1.0 |
| `subsample` | 1.0 | 0.8 | 0.5 |
| `colsample_bytree` | 1.0 | 0.8 | 0.5 |

### Typical Use Cases and Configurations

```python
# Fast prototyping (quick training)
model_fast = XGBRegressor(n_estimators=50, learning_rate=0.3, max_depth=4)

# Balanced (good default)
model_balanced = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)

# High accuracy (slower, less overfitting)
model_accurate = XGBRegressor(
    n_estimators=500, 
    learning_rate=0.01, 
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8
)

# Strong regularization (prevent overfitting)
model_regularized = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5.0,
    gamma=0.5,
    subsample=0.7,
    colsample_bytree=0.7
)
```

---

**End of Documentation**

*Last Updated: December 15, 2025*
*Version: 0.1.0*
