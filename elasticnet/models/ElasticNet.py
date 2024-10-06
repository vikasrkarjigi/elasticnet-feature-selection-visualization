import numpy as np

class ElasticNetModel:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None
        self.intercept = None
        self.mean = None
        self.std = None

    def clean_data(self, X, y):
        # Convert to numpy arrays and handle null values
        X = np.array([[float(v) if v != '' else 0 for v in row] for row in X])
        y = np.array([float(v) if v != '' else 0 for v in y])
        return X, y.ravel()

    def fit(self, X, y):
        # Clean and prepare the data
        X, y = self.clean_data(X, y)

        # Normalize features
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1  # Avoid division by zero
        X_normalized = (X - self.mean) / self.std

        n_samples, n_features = X_normalized.shape

        # Initialize coefficients
        self.coefficients = np.zeros(n_features)
        self.intercept = 0

        for _ in range(self.max_iter):
            coef_old = self.coefficients.copy()

            # Update intercept
            self.intercept = np.mean(y - X_normalized.dot(self.coefficients))

            # Update coefficients using coordinate descent
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
                    self.coefficients[j] = self._soft_threshold(z_j, self.alpha * self.l1_ratio) / (n_samples + self.alpha * (1 - self.l1_ratio))

            # Check for convergence
            if np.sum(np.abs(self.coefficients - coef_old)) < self.tol:
                break

        return self

    def _soft_threshold(self, z, threshold):
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

    def predict(self, X):
        X, _ = self.clean_data(X, [0] * len(X))  # Clean X data
        X_normalized = (X - self.mean) / self.std
        return X_normalized.dot(self.coefficients) + self.intercept