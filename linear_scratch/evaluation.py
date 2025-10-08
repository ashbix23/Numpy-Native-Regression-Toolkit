import numpy as np
import matplotlib.pyplot as plt
from linear_scratch.metrics import mse, mae, r2_score

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100)

def evaluate_model(y_true, y_pred):
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def plot_diagnostics(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, s=10)
    plt.axhline(0, color="r", linestyle="--")
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")

    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, s=10)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], "r--")
    plt.title("Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.show()

