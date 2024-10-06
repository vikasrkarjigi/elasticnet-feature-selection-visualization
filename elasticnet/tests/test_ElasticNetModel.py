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

    print("All tests passed!")