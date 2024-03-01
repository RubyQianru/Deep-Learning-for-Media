import numpy as np
from tensorflow import keras
from utils import classify_mushrooms

from unittest.mock import patch

def test_classify_mushrooms():
    # Create mock model
    mock_model = keras.models.Sequential()
    mock_model.add(keras.layers.Dense(1, input_dim=1, activation='sigmoid'))

    # Create mock features
    mock_features = np.array([[1], [2], [3]])

    # Mock the predict method of the model to return some fixed values
    with patch.object(mock_model, 'predict') as mock_predict:
        mock_predict.return_value = np.array([[0.1], [0.9], [0.4]])

        # Call the function to be tested
        result = classify_mushrooms(mock_model, mock_features)

    # Check the output
    np.testing.assert_array_equal(result, np.array([0, 1, 0]))

