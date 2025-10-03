# Linear Regression from Scratch

This project demonstrates how to implement **Linear Regression** completely from scratch using only **NumPy** (no scikit-learn).  
The implementation includes gradient descent optimization, prediction, error calculation, and visualization.

------------------------------------------------------------

## Features
- Initializes weights and bias
- Fits the model using **gradient descent**
- Makes predictions on new data
- Calculates **Mean Squared Error (MSE)**
- Visualizes the best-fit line against raw data

------------------------------------------------------------

## Project Structure
```

├── linear_regression_scratch.ipynb   # Jupyter Notebook implementation
├── README.md                         # Project documentation

```
------------------------------------------------------------

## Installation

Clone the repository and install dependencies:

	$ git clone https://github.com/AshBeeXD/Linear-Regression-Scratch.git

	$ cd Linear-Regression

Dependencies:
- numpy
- matplotlib
- jupyter (if running notebook)

------------------------------------------------------------

## Usage

Run the notebook:

	$ jupyter notebook Linear-Regression.ipynb

------------------------------------------------------------

## Example

We generate synthetic data:
    y = 2x + 3 + ε

where ε is Gaussian noise.  

The model learns parameters close to weight ≈ 2 and bias ≈ 3.

Python example:

>>> X = np.linspace(0, 10, 100).reshape(-1, 1)
>>> y = 2 * X.flatten() + 3 + np.random.randn(100)
>>> 
>>> model = LinearRegressionScratch(learning_rate=0.01, n_iters=1000)
>>> model.fit(X, y)
>>> predictions = model.predict(X)
>>> 
>>> print("Learned weights:", model.weight)
>>> print("Learned bias:", model.bias)
>>> print("MSE:", model.mse(y, predictions))

Sample Output:
    
    Learned weights: [1.9999382]
    
    Learned bias: 3.178489968762191
    
    MSE: 0.8454541970903011

------------------------------------------------------------

## Visualization

The notebook plots the raw data points (blue) and the best fit line (red):

>>> plt.scatter(X, y, color="blue", label="Data")
>>> plt.plot(X, predictions, color="red", label="Best Fit Line")
>>> plt.legend()
>>> plt.show()

Result:
- Blue = raw data
- Red = best fit line

------------------------------------------------------------

## Key Learnings
- How gradient descent works in linear regression
- Implementing ML algorithms from scratch without libraries
- Importance of learning rate and number of iterations

------------------------------------------------------------

## Future Improvements
- Add support for polynomial regression
- Implement stochastic gradient descent (SGD)
- Compare results with scikit-learn’s LinearRegression

------------------------------------------------------------

## License
This project is licensed under the MIT License.
