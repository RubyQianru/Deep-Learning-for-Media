import numpy as np
from utils import evaluate_model

import numpy as np
from unittest.mock import Mock

def test_evaluate_model():
    # Create a mock model that always predicts 0
    mock_model = Mock()
    mock_model.predict.return_value = np.zeros((10, 1))

    # Generate a random test set with half 0s and half 1s
    X_test = np.random.rand(10, 3)
    y_test = np.concatenate([np.zeros(5), np.ones(5)])

    # Evaluate the model
    accuracy = evaluate_model(mock_model, X_test, y_test)

    # Check that the accuracy is approximately 50%
    assert isinstance(accuracy, float)
    assert np.isclose(accuracy, 0.5, rtol=1e-3)


