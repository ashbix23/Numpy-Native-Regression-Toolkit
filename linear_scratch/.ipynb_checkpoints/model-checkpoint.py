from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from .metrics import mse, mae, r2_score

Array = np.ndarray

@dataclass
class LinearRegressionScratch:
    """
    Multiple Linear Regression using batch Gradient Descent with optional
    Ridge (L2) regularization and early stopping.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent.
    n_iters : int, default=1000
        Maximum training iterations (epochs).
    lambda_reg : float, default=0.0
        Ridge regularization strength (L2). Set >0 to shrink weights.
    tol : float, default=1e-4
        Minimum absolute improvement in validation loss to count as progress.
    patience : int, default=20
        Number of epochs with < tol improvement allowed before stopping early.
    random_state : Optional[int], default=None
        If set, seeds NumPy RNG for reproducibility.

    Attributes
    ----------
    weight : (n_features,) ndarray
        Learned weights.
    bias : float
        Intercept term.
    cost_history : List[float]
        Training MSE history (with L2 term included in cost).
    val_cost_history : List[float]
        Validation MSE history when validation data is provided.
    """
    learning_rate: float = 0.01
    n_iters: int = 1000
    lambda_reg: float = 0.0
    tol: float = 1e-4
    patience: int = 20
    random_state: Optional[int] = None

    weight: Optional[Array] = field(init=False, default=None)
    bias: float = field(init=False, default=0.0)
    cost_history: List[float] = field(init=False, default_factory=list)
    val_cost_history: List[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.random_state is not None:
            np.random.seed(self.random_state)

    # ----------- core API -----------
    def fit(
        self,
        X: Array,
        y: Array,
        X_val: Optional[Array] = None,
        y_val: Optional[Array] = None,
        verbose: bool = False,
    ) -> "LinearRegressionScratch":
        X, y = self._validate_X_y(X, y)
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history.clear()
        self.val_cost_history.clear()

        best_val = np.inf
        epochs_since_improve = 0

        for it in range(self.n_iters):
            # forward
            y_pred = X @ self.weight + self.bias

            # gradients (MSE + L2, averaged)
            error = y_pred - y
            dw = (2.0 / n_samples) * (X.T @ error) + 2.0 * self.lambda_reg * self.weight
            db = (2.0 / n_samples) * np.sum(error)

            # step
            self.weight -= self.learning_rate * dw
            self.bias   -= self.learning_rate * db

            # training objective: MSE + L2 term (without regularizing bias)
            train_cost = mse(y, y_pred) + (self.lambda_reg / n_samples) * float(np.dot(self.weight, self.weight))
            self.cost_history.append(train_cost)

# validation early stopping
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_cost = mse(y_val, y_val_pred) + (self.lambda_reg / n_samples) * float(np.dot(self.weight, self.weight))
                self.val_cost_history.append(val_cost)

                improvement = best_val - val_cost
                if improvement > self.tol:
                    best_val = val_cost
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= self.patience:
                        if verbose:
                            print(f"[EarlyStopping] epoch={it+1}, best_val={best_val:.6f}")
                        break

        return self

    def predict(self, X: Array) -> Array:
        X = self._validate_X(X)
        if self.weight is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return X @ self.weight + self.bias

    def evaluate(self, X: Array, y: Array) -> Dict[str, float]:
        y_pred = self.predict(X)
        return {
            "MSE": mse(y, y_pred),
            "MAE": mae(y, y_pred),
            "R2":  r2_score(y, y_pred),
        }

    # ----------- utilities -----------
    @staticmethod
    def _validate_X(X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 2D array-like [n_samples, n_features].")
        return X

    @staticmethod
    def _validate_X_y(X: Array, y: Array) -> Tuple[Array, Array]:
        X = LinearRegressionScratch._validate_X(X)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        return X, y

