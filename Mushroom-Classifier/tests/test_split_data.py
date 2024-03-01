import numpy as np
from numpy.testing import assert_array_equal
from pytest import approx
from utils import split_data


def test_split_data():
    # Test case 1
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 1, 1, 1, 0])
    X_train, X_val, y_train, y_val = split_data(X, y, train_size=0.8, val_size=0.2)
    assert len(X_train) == 4
    assert len(X_val) == 1
    assert len(y_train) == 4
    assert len(y_val) == 1

    # Test case 2
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 0, 1, 1, 1])
    X_train, X_val, y_train, y_val = split_data(X, y, train_size=0.7, val_size=0.3)
    assert len(X_train) == 3
    assert len(X_val) == 2
    assert len(y_train) == 3
    assert len(y_val) == 2

    # Test case 3
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 0, 1, 1, 1])
    X_train, X_val, y_train, y_val = split_data(X, y, train_size=0.85, val_size=0.15)
    assert len(X_train) == 4
    assert len(X_val) == 1
    assert len(y_train) == 4
    assert len(y_val) == 1
