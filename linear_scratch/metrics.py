from __future__ import annotations
import numpy as np

Array = np.ndarray

def mse(y_true: Array, y_pred: Array) -> float:
    """
    Mean Squared Error.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("mse: y_true and y_pred must have the same shape.")
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true: Array, y_pred: Array) -> float:
    """
    Mean Absolute Error.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("mae: y_true and y_pred must have the same shape.")
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true: Array, y_pred: Array) -> float:
    """
    Coefficient of determination R^2.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("r2_score: y_true and y_pred must have the same shape.")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

