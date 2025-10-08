from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_scratch import LinearRegressionScratch
from linear_scratch.metrics import mse, mae, r2_score
import numpy as np

def benchmark(parity_threshold=0.98):
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

    scratch = LinearRegressionScratch(
        learning_rate=0.01, n_iters=1500, lambda_reg=0.1, tol=1e-5,
        patience=20, random_state=42
    )
    scratch.fit(X_train, y_train, verbose=False)
    y_pred_scratch = scratch.predict(X_test)

    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    metrics = {
        "Scratch": {
            "MSE": mse(y_test, y_pred_scratch),
            "MAE": mae(y_test, y_pred_scratch),
            "R2": r2_score(y_test, y_pred_scratch)
        },
        "Ridge": {
            "MSE": mse(y_test, y_pred_ridge),
            "MAE": mae(y_test, y_pred_ridge),
            "R2": r2_score(y_test, y_pred_ridge)
        }
    }

    parity = 1 - abs(metrics["Scratch"]["R2"] - metrics["Ridge"]["R2"]) / abs(metrics["Ridge"]["R2"])

    print("\nBenchmark Results:")
    for model_name in metrics:
        m = metrics[model_name]
        print(f"{model_name:>8}: MSE={m['MSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
    print(f"\nPerformance parity: {parity*100:.2f}%")

    if parity < parity_threshold:
        print("Warning: Parity below expected threshold.")

if __name__ == "__main__":
    benchmark()

