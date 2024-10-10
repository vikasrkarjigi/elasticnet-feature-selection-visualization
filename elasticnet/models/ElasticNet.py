import numpy as np
import matplotlib.pyplot as plt

class ElasticNetModel:

    """
    Initializes the model with alpha (regularization strength),
    l1_ratio (mix of L1/L2 regularization), max_iter, and tolerance for convergence.
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None
        self.intercept = None
        self.mean = None
        self.std = None

    """
    Converts input data to float and replaces empty values with 0,
    ensuring data is in the correct format for processing.
    """
    def clean_data(self, X, y):
        X = np.array([[float(v) if v != '' else 0 for v in row] for row in X])
        y = np.array([float(v) if v != '' else 0 for v in y])
        return X, y.ravel()

    """
    Fits the ElasticNet model using coordinate descent,
    normalizing the data and updating coefficients iteratively.
    """
    def fit(self, X, y):
        X, y = self.clean_data(X, y)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1
        X_normalized = (X - self.mean) / self.std

        n_samples, n_features = X_normalized.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = 0

        for _ in range(self.max_iter):
            coef_old = self.coefficients.copy()
            self.intercept = np.mean(y - X_normalized.dot(self.coefficients))

            for j in range(n_features):
                X_j = X_normalized[:, j]
                y_pred = X_normalized.dot(self.coefficients) + self.intercept
                r_j = y - y_pred + self.coefficients[j] * X_j
                z_j = X_j.dot(r_j)

                if self.l1_ratio == 1:
                    self.coefficients[j] = self._soft_threshold(z_j, self.alpha) / n_samples
                elif self.l1_ratio == 0:
                    self.coefficients[j] = z_j / (n_samples + self.alpha)
                else:
                    self.coefficients[j] = self._soft_threshold(z_j, self.alpha * self.l1_ratio) / (
                                n_samples + self.alpha * (1 - self.l1_ratio))

            if np.sum(np.abs(self.coefficients - coef_old)) < self.tol:
                break

        return self

    """
    Applies the soft-thresholding function for L1 regularization
    to shrink coefficients towards zero.
    """
    def _soft_threshold(self, z, threshold):
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

    """
    Predicts the target values for new input data
    using the trained model by applying the learned coefficients.
    """
    def predict(self, X):
        X, _ = self.clean_data(X, [0] * len(X))
        X_normalized = (X - self.mean) / self.std
        return X_normalized.dot(self.coefficients) + self.intercept

    """
    Computes the mean squared error (MSE)
    between actual and predicted values.
    """
    def mean_squared_error(self, y_actual, y_pred):
        return np.mean((y_actual - y_pred) ** 2)

    """
    Computes the mean absolute error (MAE)
    between actual and predicted values.
    """
    def mean_absolute_error(self, y_actual, y_pred):
        return np.mean(np.abs(y_actual - y_pred))

    """
    Computes the R-squared (coefficient of determination)
    to measure how well the model explains the variance in the target.
    """
    def r_squared(self, y_actual, y_pred):
        ss_total = np.sum((y_actual - np.mean(y_actual)) ** 2)
        ss_residual = np.sum((y_actual - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    """
    Evaluates the model's performance by calculating MSE, MAE, R-squared,
    and plotting the results.
    """
    def evaluate(self, X, y, plot_name):
        y_pred = self.predict(X)
        mse = self.mean_squared_error(y, y_pred)
        mae = self.mean_absolute_error(y, y_pred)
        r2 = self.r_squared(y, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared: {r2}")
        self.plot_results(y, y_pred, plot_name)

    """
    Generates and displays scatter and residual plots
    to visually assess model performance.
    """
    def plot_results(self, y_actual, y_pred, plot_name):
        plt.figure(figsize=(14, 6))
        plt.suptitle(plot_name, fontsize=16)  # Set the figure title

        # Subplot 1: Scatter plot (Actual vs Predicted)
        plt.subplot(1, 2, 1)
        plt.scatter(y_actual, y_pred, color='blue', alpha=0.7, label='- Predicted vs Actual')
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2, label='- Perfect Fit')  # Diagonal line
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.legend()  # Add legend to the scatter plot

        # Subplot 2: Residual plot (Residuals vs Predicted)
        residuals = y_actual - y_pred
        plt.subplot(1, 2, 2)
        plt.scatter(y_pred, residuals, color='purple', alpha=0.7, label='- Residuals')
        plt.axhline(0, color='red', linestyle='--', label='- Zero Residual')  # Horizontal line at 0
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Values")
        plt.legend()  # Add legend to the residual plot

        # Display both plots
        plt.tight_layout()
        plt.show()
