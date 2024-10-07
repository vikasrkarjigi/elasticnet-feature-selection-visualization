import csv

import numpy

from elasticnet.models.ElasticNet import ElasticNetModel

def test_predict():
    model = ElasticNetModel()
    data = []
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = numpy.array([[v for k,v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([[v for k,v in datum.items() if k=='y'] for datum in data])
    results = model.fit(X,y)
    preds = results.predict(X)
    # assert preds == 0.5

    # Check if predictions are close to the actual values
    y_float = numpy.array([float(val) for val in y])
    mse = numpy.mean((preds - y_float) ** 2)
    print(f"Mean Squared Error: {mse}")

    # Check if predictions are not all the same
    assert not numpy.allclose(preds, preds[0]), "All predictions are the same"

    # Check if predictions are finite
    assert numpy.all(numpy.isfinite(preds)), "Predictions contain non-finite values"

    # Check if there's some correlation between predictions and actual values
    correlation = numpy.corrcoef(preds, y_float)[0, 1]
    print(f"Correlation between predictions and actual values: {correlation}")
    assert correlation > 0, "No positive correlation between predictions and actual values"

    model.evaluate(X,y_float,"small_test data set")

    print("All tests passed!")

def test_zero_variance_features():
    # Create dataset with one feature having zero variance
    X = numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = numpy.array([1, 2, 3, 4])

    model = ElasticNetModel()
    model.fit(X, y)
    preds = model.predict(X)

    # Ensure predictions are finite and reasonable
    assert numpy.all(numpy.isfinite(preds)), "Predictions contain non-finite values"
    model.evaluate(X, y, "Zero variance data set")
    print("Zero variance feature test passed!")

def test_highly_correlated_features():
    # Create highly correlated features
    X = numpy.array([[1, 2, 4], [2, 4, 8], [3, 6, 12], [4, 8, 16]])
    y = numpy.array([2, 4, 6, 8])

    model = ElasticNetModel()
    model.fit(X, y)
    preds = model.predict(X)

    # Ensure the predictions are finite and reasonable
    assert numpy.all(numpy.isfinite(preds)), "Predictions contain non-finite values"
    model.evaluate(X, y, "High correlated data set")
    print("Highly correlated features test passed!")

def test_sparse_data():
    # Create dataset with many zero features
    X = numpy.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
    y = numpy.array([1, 2, 3, 4])

    model = ElasticNetModel()
    model.fit(X, y)
    preds = model.predict(X)

    # Ensure the predictions are finite and reasonable
    assert numpy.all(numpy.isfinite(preds)), "Predictions contain non-finite values"
    model.evaluate(X, y, "sparse data set")
    print("Sparse data test passed!")

def test_with_outliers():
    # Create a dataset with outliers
    X = numpy.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [100, 200, 300]])  # Outlier in last row
    y = numpy.array([1, 2, 3, 4, 100])

    model = ElasticNetModel()
    model.fit(X, y)
    preds = model.predict(X)

    # Ensure the predictions are finite and reasonable
    assert numpy.all(numpy.isfinite(preds)), "Predictions contain non-finite values"
    model.evaluate(X, y, "Data set with outliers")
    print("Outliers test passed!")

def test_large_dataset():
    # Create a large random dataset
    numpy.random.seed(42)
    X = numpy.random.rand(10000, 10)
    y = numpy.random.rand(10000)

    model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)
    model.fit(X, y)
    preds = model.predict(X)

    # Ensure predictions are finite
    assert numpy.all(numpy.isfinite(preds)), "Predictions contain non-finite values"
    model.evaluate(X, y, "Large Data set")
    print("Large dataset test passed!")

def test_different_alpha_l1_ratios():
    X, y = create_test_data(n_samples=100, n_features=3)

    # Different alpha and l1_ratio settings
    for alpha, l1_ratio in [(0.01, 0.1), (1.0, 0.5), (10.0, 0.9)]:
        model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X, y)
        preds = model.predict(X)

        # Ensure predictions are finite and reasonable
        assert numpy.all(numpy.isfinite(preds)), "Predictions contain non-finite values"
        model.evaluate(X, y, "Data set with differnt alpha & l1 ratios")
        print(f"Alpha {alpha}, L1_ratio {l1_ratio} test passed!")

def test_non_normalized_data():
    # Dataset without normalization
    X = numpy.array([[10, 20, 30], [20, 40, 60], [30, 60, 90], [40, 80, 120]])
    y = numpy.array([10, 20, 30, 40])

    model = ElasticNetModel()
    model.fit(X, y)
    preds = model.predict(X)

    # Ensure predictions are finite
    assert numpy.all(numpy.isfinite(preds)), "Predictions contain non-finite values"
    model.evaluate(X, y, "Data set without normalized")
    print("Non-normalized data test passed!")

def create_test_data(n_samples=100, n_features=3):
    """
    Creates synthetic data for testing the ElasticNet model.

    Parameters:
        n_samples (int): Number of data points.
        n_features (int): Number of features.

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target vector of shape (n_samples,).
    """
    numpy.random.seed(42)  # For reproducibility
    X = numpy.random.randn(n_samples, n_features)  # Random features
    y = X @ numpy.random.randn(n_features) + numpy.random.normal(scale=0.5, size=n_samples)  # y = Xβ + noise
    return X, y





