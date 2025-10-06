# Linear Regression from Scratch (Enhanced)

This project implements **multiple linear regression** entirely from scratch using **NumPy**, with **Ridge (L2) regularization**, **early stopping**, and thorough **evaluation and visualization**.  
It uses the **California Housing** dataset and benchmarks performance against scikit-learn’s `LinearRegression` and `Ridge`.

---

## Features

- **Batch Gradient Descent** with configurable learning rate and epochs  
- **Multiple features** (true multivariate regression)  
- **Ridge (L2) Regularization** to improve generalization  
- **Early Stopping** using a validation set (`tol`, `patience`)  
- Proper **Train/Validation/Test split** and **Standardization**  
- **Metrics:** MSE, MAE, R² on train/test  
- **Visualizations:** convergence curve, coefficient bar chart, residuals plot  
- **Baselines:** scikit-learn `LinearRegression` and `Ridge` on the same data

---

## Project Structure

    Linear-Regression-Scratch/
    ├── linear_scratch/
    │   ├── __init__.py
    │   ├── model.py           # NumPy implementation (GD + L2 + early stopping)
    │   ├── metrics.py         # Custom MSE, MAE, R²
    │   └── plotting.py        # Convergence, coefficients, residuals
    ├── notebooks/
    │   └── Linear_Regression_Scratch_Enhanced.ipynb
    ├── requirements.txt
    └── README.md

---

## Installation

    # Clone this repository
    git clone https://github.com/AshBeeXD/Linear-Regression-Scratch.git
    cd Linear-Regression-Scratch

    # (Optional) Create and activate a virtual environment
    python -m venv .venv
    # macOS/Linux:
    source .venv/bin/activate
    # Windows (PowerShell):
    .venv\Scripts\Activate.ps1

    # Install dependencies
    pip install -r requirements.txt

---

## Usage

    # Open the notebook
    jupyter notebook notebooks/Linear_Regression_Scratch_Enhanced.ipynb

Then execute all cells in order to:
1. Load and preprocess the dataset  
2. Train the custom Linear Regression model (NumPy)  
3. Visualize convergence and interpret coefficients  
4. Compare against scikit-learn baselines

---

## Example Results

| Metric | Train (Scratch) | Test (Scratch) | Train (SK Ridge) | Test (SK Ridge) |
|:------:|:----------------:|:--------------:|:----------------:|:---------------:|
|   MSE  |       0.53       |      0.56      |       0.53       |       0.56      |
|   MAE  |       0.56       |      0.59      |       0.56       |       0.59      |
|   R²   |       0.60       |      0.57      |       0.60       |       0.57      |

*Values will vary slightly by seed and split. Typical R² on this setup is ~0.55–0.60.*

---

## Key Learnings

- **Standardization** greatly stabilizes and accelerates gradient descent  
- **Ridge** reduces overfitting by shrinking coefficients toward zero  
- **Early stopping** halts training when validation loss plateaus  
- Building models from scratch deepens understanding of optimization and bias–variance

---

## Extensions

- Add features (e.g., `AveBedrms`, `Population`, `AveOccup`) and re-tune hyperparameters  
- Implement **L1 (Lasso)** or **Elastic Net**  
- Try **SGD**, momentum, or learning-rate schedules  
- Use **k-fold cross-validation** for robust evaluation  
- Add a small **hyperparameter sweep** for `lambda_reg` and learning rate

---

## License

This project is released under the **MIT License** (see `LICENSE`).
