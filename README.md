# myXGBoost

A modular XGBoost implementation following sklearn design patterns.

## Installation

```bash
pip install -e .
```

## Usage

```python
from myxgboost import XGBRegressor, XGBClassifier

# Regression
regressor = XGBRegressor()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# Classification
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

