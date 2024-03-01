from unittest.mock import Mock
import numpy as np
from tensorflow import keras
from utils import train_model

def test_train_model():
    # create a mock model and history object
    mock_model = Mock(spec=keras.models.Model)
    mock_history = Mock(spec=keras.callbacks.History)

    # create mock training and validation data
    X_train = np.zeros((100, 10))
    y_train = np.zeros((100,))
    X_val = np.zeros((50, 10))
    y_val = np.zeros((50,))

    # train the mock model using the mock data
    train_model(mock_model, X_train, y_train, epochs=5, batch_size=32,
                class_weights=None, X_val=X_val, y_val=y_val)

    # assert that the mock model's `fit` method was called with the expected arguments
    mock_model.fit.assert_called_once_with(X_train, y_train,
                                           validation_data=(X_val, y_val),
                                           epochs=5, batch_size=32,
                                           class_weight=None)

    # assert that the mock model and history object were returned
    model, history = train_model(mock_model, X_train, y_train, epochs=5, batch_size=32,
                                 class_weights=None, X_val=X_val, y_val=y_val)
    assert model == mock_model
    assert history == mock_model.fit.return_value

