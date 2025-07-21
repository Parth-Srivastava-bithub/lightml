### 📘 **Linear Regression – Normal Equation Implementation**

This `fit` function fits a **Linear Regression** model using the **Normal Equation**, which directly calculates the optimal weights (coefficients) without any iteration.

#### 💡 Code Explanation:

```python
def fit(self, X_train, y_train):
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)  # Ensure 2D input for single feature

    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Add bias term
    self.coef_ = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))  # Normal Equation
```

#### 🧠 Logic Behind It:

We solve the following equation directly:

$$
\theta = (X^T X)^{-1} X^T y
$$

Where:

* $X$ is the design matrix (with a column of ones for bias)
* $y$ is the target vector
* $\theta$ is the vector of weights (including bias)

#### 🔢 Steps:

1. **Bias Term Addition**
   Add a column of ones to handle the intercept:

   $$
   X = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}
   $$

2. **Apply Normal Equation**
   Calculate weights:

   $$
   \theta = (X^T X)^{-1} X^T y
   $$

3. **Result**
   `self.coef_` stores the learned parameters:

   * First value = bias
   * Rest = feature coefficients

---

### 🧲 **Linear Regression – L1 Regularization (Lasso) using BGD**

This version of linear regression includes **L1 regularization**, which encourages **sparse weights** (many coefficients become zero).

#### 🔧 Code Overview:

```python
def fit(self, X_train, y_train):
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if y_train.ndim > 1:
        y_train = y_train.ravel()

    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Add bias
    self.coef_ = np.zeros(X_train.shape[1])  # Init weights
    n = X_train.shape[0]

    for _ in range(self.epochs):
        error = y_train - np.dot(X_train, self.coef_)
        slope = (-2 / n) * np.dot(X_train.T, error) + self.thisLambda * np.sign(self.coef_)
        self.coef_ -= self.learning_rate * slope
```

#### 🔍 What's New: L1 Penalty

The gradient update includes **L1 term**:

$$
\theta := \theta - \alpha \left[ \frac{-2}{n} X^T (y - X\theta) + \lambda \cdot \text{sign}(\theta) \right]
$$

Where:

* $\lambda$: regularization strength (`self.thisLambda`)
* $\text{sign}(\theta)$: element-wise sign function
* L1 penalty pushes weights toward **zero**, aiding in feature selection

#### 🎯 Result:

After training, `self.coef_` contains:

* $\theta_0$: bias term
* $\theta_i$: possibly zeroed coefficients due to L1

---

