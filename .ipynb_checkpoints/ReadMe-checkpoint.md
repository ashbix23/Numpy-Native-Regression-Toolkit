# Linear Regression from Scratch  

![Predicted vs Actual](notebooks/images/predicted_vs_actual.png)

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![NumPy](https://img.shields.io/badge/NumPy-%3E%3D1.24-orange.svg)](https://numpy.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-red.svg)](https://jupyter.org/)  

---

## Overview
This project builds **Multivariate Linear Regression from scratch** using only **NumPy**, implementing modern enhancements such as **Ridge (L2) Regularization**, **Early Stopping**, and **visual diagnostics**. It benchmarks performance against scikit-learn's `LinearRegression` and `Ridge`, using the **California Housing** dataset.

The purpose: to deeply understand the mechanics of gradient descent, regularization, and model evaluation — not just use libraries, but *build the fundamentals yourself*.


---

## Table of Contents
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results & Visualizations](#results--visualizations)
- [Interpretation](#interpretation)
- [Troubleshooting](#troubleshooting)
- [Extensions](#extensions)
- [Contributing](#contributing)
- [License](#license)
- [New Additions and Enhancements](#new-additions-and-enhancements-post-feedback-implementation)


---

## Key Features
- **Batch Gradient Descent** with configurable hyperparameters  
- **Ridge (L2) Regularization** for better generalization  
- **Early Stopping** (tolerance + patience)  
- **Standardized Train/Validation/Test Split**  
- **Custom Evaluation Metrics:** MSE, MAE, R²  
- **Visual Diagnostics:**
  - Convergence curves
  - Feature importance bar chart
  - Residuals plot
  - Predicted vs Actual scatter
- **Baselines:** Compare against scikit-learn’s `LinearRegression` and `Ridge`

---

## Project Structure
```
Linear-Regression-Scratch/
├── linear_scratch/
│   ├── __init__.py
│   ├── model.py
│   ├── metrics.py
│   ├── plotting.py
│   ├── preprocessing.py        
│   ├── evaluation.py           
│   ├── visualization.py        
│   └── benchmarks.py           
│
├── notebooks/
│   ├── Linear_Regression_Scratch.ipynb
│   ├── End_to_End_Pipeline.ipynb        
│   └── images/
│       ├── feature_importance.png
│       ├── residuals.png
│       └── predicted_vs_actual.png
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Dataset & Preprocessing
- **Dataset:** California Housing dataset from `scikit-learn.datasets`  
- **Target:** Median house value (log-scaled)  
- **Preprocessing Steps:**
  - Missing-value handling via `SimpleImputer`
  - Scaling using `StandardScaler` or `RobustScaler`
  - Random Train/Validation/Test split (seeded)
  - Ridge penalty tuning and early stopping based on validation loss

This ensures stable gradient updates and mitigates feature magnitude bias.

---

## Installation

```
# Clone the repository
git clone https://github.com/AshBeeXD/Linear-Regression-Scratch.git
cd Linear-Regression-Scratch

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\Activate.ps1 # Windows

# Install dependencies
pip install -r requirements.txt

```

---

## Quick Start

Run the Jupyter notebook to reproduce the full workflow:
```
jupyter notebook notebooks/Linear_Regression_Scratch.ipynb

```
Or import and train directly from Python:
```
from linear_scratch import LinearRegressionScratch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# Train custom model
model = LinearRegressionScratch(
    learning_rate=0.01, n_iters=1000, lambda_reg=0.1, tol=1e-4, patience=10
)
model.fit(X_train, y_train, verbose=True)

# Evaluate performance
print(model.evaluate(X_test, y_test))
```

---

## Results & Visualizations

| Metric | Train (Scratch) | Test (Scratch) | Train (SK Ridge) | Test (SK Ridge) |
|:-------|:----------------|:---------------|:-----------------|:----------------|
| MSE    | 0.6534 | 0.6625 | 0.6474 | 0.6589 |
| MAE    | 0.6070 | 0.6112 | 0.5987 | 0.6033 |
| R²     | 0.5112 | 0.4945 | 0.5157 | 0.4972 |

**Visual Diagnostics**

- **Feature Importance**  
  ![Feature Importance](notebooks/images/feature_importance.png)

- **Residuals Distribution**  
  ![Residuals](notebooks/images/residuals.png)

- **Predicted vs Actual**  
  ![Predicted vs Actual](notebooks/images/predicted_vs_actual.png)

---

## Interpretation

The model highlights **Median Income (MedInc)** as the most influential predictor of housing prices, followed by **House Age** and **Average Rooms**.  
Ridge regularization reduces overfitting by penalizing large coefficients, while **early stopping** ensures faster and more stable convergence.

---

## Benchmarks & Performance Parity
`benchmarks.py` demonstrates near-parity (≈96–98%) with scikit-learn’s `Ridge` regression.

**Run:**
```
python -m linear_scratch.benchmarks
```

Benchmark Results:
| Model | MSE| MAE| R2|
|:-------|:----------------|:---------------|:-----------------|
| Scratch  | 0.5826 | 0.5624 | 0.5554| 
| Ridge    | 0.5559 | 0.5332 | 0.5758 | 

Performance parity: 96.46%

These results confirm that the scratch-built model performs comparably to Ridge regression while being trained via gradient descent rather than a closed-form analytical solution.

---

## Advanced Preprocessing Pipeline

`preprocessing.py` introduces a structured and reliable data preprocessing system to ensure consistent, reproducible inputs for model training.

### Key Features
- Handles missing values using `SimpleImputer`
- Scales features using either `StandardScaler` or `RobustScaler`
- Supports column-wise transformations with `ColumnTransformer`
- Integrates seamlessly with the end-to-end pipeline
- Ensures consistent preprocessing between training and testing sets

This approach improves model stability, prevents data leakage, and ensures reproducibility across experiments.

**Usage Example:**
```python
from linear_scratch.preprocessing import preprocess

X_train_prep, X_test_prep, transformer = preprocess(X_train, X_test)
```

---

## Comprehensive Evaluation Suite

The `evaluation.py` module introduces a complete and extensible evaluation system for analyzing model performance through both numerical metrics and visual diagnostics.

---

### Metrics Included
- **MSE** — Mean Squared Error  
- **RMSE** — Root Mean Squared Error  
- **MAE** — Mean Absolute Error  
- **MAPE** — Mean Absolute Percentage Error  
- **R²** — Coefficient of Determination  

Each metric provides a different perspective on model performance — from absolute error magnitude (MAE) to relative error percentage (MAPE) and overall variance explanation (R²).

**Example Usage:**
```python
from linear_scratch.evaluation import evaluate_model, plot_diagnostics

results = evaluate_model(y_test, y_pred)
plot_diagnostics(y_test, y_pred)
```
---

**Outputs:**

- Tabular summary of all performance metrics

- Residual distribution plots for model error analysis

- Predicted vs Actual plots to visualize model fit quality

- Optional MAPE percentage for interpretability on scaled data

- This evaluation suite ensures that both quantitative and visual validation steps are incorporated into the workflow.

## Visualization Module for Model Insights

The `visualization.py` module introduces a comprehensive and unified visualization framework for interpreting model behavior, performance, and convergence patterns.  
It centralizes all major diagnostic plots into a single, easy-to-use interface to help users understand **how** and **why** their linear model performs the way it does.

---

### Key Capabilities

- **Convergence Curve:** Visualizes loss reduction across epochs, confirming whether the model converged smoothly or prematurely.  
- **Feature Importance Chart:** Displays the contribution of each feature to the prediction, derived from learned weights.  
- **Residual Distribution Plot:** Highlights bias and variance in model errors — helps identify underfitting or overfitting.  
- **Predicted vs Actual Plot:** Provides a visual measure of regression accuracy and model alignment with the target variable.  

---

### Example Usage
```python
from linear_scratch.visualization import plot_model_insights

plot_model_insights(model, X_test, y_test, feature_names=feature_names)
```

This single call generates all key diagnostics automatically and saves or displays them as high-resolution figures for reporting or further analysis.

---

## Generated Plots

When the visualization module is executed, it automatically generates and saves several key diagnostic plots that capture the model’s performance and learning behavior.  
These visualizations help assess the regression quality, feature influence, and stability of training.

---

### 1. Convergence Curve
- **Purpose:** Shows how the loss decreases over training epochs.  
- **Interpretation:**  
  - A smooth downward trend indicates stable convergence.  
  - A noisy or flat curve may signal too high a learning rate or early stopping.  
- **Insight:** Confirms that gradient descent optimization is functioning as expected.

---

### 2. Feature Importance Chart
- **Purpose:** Displays the relative weight or importance of each input feature.  
- **Interpretation:**  
  - Features with higher absolute weights contribute more to predictions.  
  - Negative weights indicate an inverse relationship with the target variable.  
- **Insight:** Helps identify the key drivers of housing prices, e.g., **Median Income (MedInc)** typically has the strongest positive influence.

---

### 3. Residual Distribution Plot
- **Purpose:** Shows how prediction errors (residuals) are distributed.  
- **Interpretation:**  
  - A centered and symmetric distribution around zero suggests an unbiased model.  
  - Heavy tails or skewness may indicate systematic prediction bias.  
- **Insight:** Helps evaluate underfitting, overfitting, or model bias.

---

### 4. Predicted vs Actual Plot
- **Purpose:** Compares the model’s predictions with actual target values.  
- **Interpretation:**  
  - Points close to the diagonal line represent accurate predictions.  
  - Dispersion away from the diagonal indicates variance or bias in the model.  
- **Insight:** A dense cluster along the diagonal demonstrates strong performance parity with the true data distribution.

---

## Benefits of Visualization

- Enables **quick debugging** of training behavior through visual cues.  
- Offers **interpretability** — users can understand how the model arrives at predictions.  
- Ensures **reproducibility**, as plots are automatically generated and saved.  
- Serves as a **communication tool** in reports, presentations, or publications.

By combining quantitative metrics with visual analysis, the project bridges the gap between algorithmic performance and human interpretability.

---

## Integration with the Full Pipeline

The visualization module integrates directly into the end-to-end pipeline, running seamlessly after model training and evaluation.

**Example Workflow:**
```python
from linear_scratch.visualization import plot_model_insights
from linear_scratch.evaluation import evaluate_model

# Evaluate model predictions
results = evaluate_model(y_test, y_pred)

# Generate all visualizations in one call
plot_model_insights(model, X_test, y_test, feature_names=feature_names)
```

---
**Typical Execution Order:**

- Train model with LinearRegressionScratch

- Evaluate metrics with evaluation.py

- Generate plots using visualization.py

- Save all outputs under /notebooks/images/

- This ensures a unified, reproducible workflow — where numerical validation and visualization are tightly coupled.

---

## Troubleshooting
| Issue | Possible Cause | Fix |
|:------|:----------------|:----|
| Diverging loss | Too high learning rate | Lower `learning_rate` (e.g., 0.001) |
| No improvement in validation | Over-regularization | Reduce `lambda_reg` |
| NaN or inf values | Data not standardized | Use `StandardScaler` before training |


---

## Contributing
Contributions, bug reports, and improvements are welcome!  
Fork the repo, create a feature branch, and submit a pull request.

```
git checkout -b feature/new-feature
git commit -m "Add new feature"
git push origin feature/new-feature
```

---

## License
Released under the **MIT License**. See [LICENSE](LICENSE) for details.
