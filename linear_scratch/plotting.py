from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence

def plot_training_history(train_costs, val_costs=None, title="Convergence"):
    plt.figure()
    plt.plot(train_costs, label="Train")
    if val_costs is not None and len(val_costs) > 0:
        plt.plot(val_costs, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (+ L2)")
    plt.title(title)
    plt.legend()
    plt.show()

def bar_feature_importance(coefs, feature_names=None, title="Feature Importance"):
    plt.figure()
    idx = np.arange(len(coefs))
    plt.bar(idx, coefs)
    if feature_names is not None:
        plt.xticks(idx, feature_names, rotation=45, ha="right")
    plt.ylabel("Coefficient")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def residuals_plot(y_true, y_pred, title="Residuals"):
    plt.figure()
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, s=12)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.show()

