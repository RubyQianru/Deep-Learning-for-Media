import numpy as np
from utils import load_data

def test_load_data():
  X_train, X_test, y_train, y_test = load_data()

  assert isinstance(X_train, np.ndarray) #, "X_train should be a numpy array"
  assert isinstance(X_test, np.ndarray) #, "X_test should be a numpy array"
  assert isinstance(y_train, np.ndarray) #, "y_train should be a numpy array"
  assert isinstance(y_test, np.ndarray)#, "y_test should be a numpy array"

  assert X_train.shape[0] == y_train.shape[0]#, "X_train and y_train should have the same number of samples"
  assert X_test.shape[0] == y_test.shape[0]#, "X_test and y_test should have the same number of samples"

  assert len(np.unique(y_train)) == 2#, "y_train should contain only 2 unique classes"
  assert len(np.unique(y_test)) == 2#, "y_test should contain only 2 unique classes"
  assert np.sum(y_train == 0) + np.sum(y_train == 1) == y_train.shape[0]#, "y_train should contain only 0s and 1s"
  assert np.sum(y_test == 0) + np.sum(y_test == 1) == y_test.shape[0]#, "y_test should contain only 0s and 1s"
