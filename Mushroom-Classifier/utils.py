import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from keras.models import Sequential


def load_data():
    """
    Load the mashroom dataset from disk.

    Returns
    -------
    X_train : np.ndarray
        A numpy array containing the features of the trainin set of the 
        mashroom dataset.
    X_test : np.ndarray
        A numpy array containing the features of the test set of the 
        mashroom dataset.
    y_train : np.ndarray
        A numpy array containing the training labels of the mashroom dataset.
    y_test : np.ndarray
        A numpy array containing the test labels of the mashroom dataset.
    """
    # YOUR CODE HERE
    x_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    x_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    return x_train, x_test, y_train, y_test
    # Hint: use np.load
    pass

def split_data(X, y, train_size=0.9, val_size=0.1):
    """
    Splits the input data into training and validation according
    to the given percentages. 

    Parameters
    ----------
    X : np.ndarray
        The input features to be split.
    y : np.ndarray
        The input labels to be split.
    train_size : float, optional
        The proportion of data to be used for training, by default 0.9.
    val_size : float, optional
        The proportion of data to be used for validation, by default 0.1.

    Returns
    -------
    X_train : np.ndarray
        The input features to be used for training.
    X_val : np.ndarray
        The input features to be used for validation.
    y_train : np.ndarray
        The input labels to be used for training.
    y_val : np.ndarray
        The input labels to be used for validation.
    """
    # YOUR CODE HERE
    # Calculate the size of training and validation sets given the percentages
    train_size = int(len(X) * train_size)
    val_size = int(len(X) * val_size)

    x_train = X
    y_train = y

    x_train, x_val = x_train[:train_size], x_train[-val_size:]
    y_train, y_val = y_train[:train_size], y_train[-val_size:]


    return x_train, x_val, y_train, y_val
    pass


def build_baseline(input_shape):
  """
  Build a binary classification baseline using Dense layers.

  Parameters
  ----------
  input_shape : int
      The shape of the input data. This should be an integer
      specifying the number of features in the input data (e.g. 200).

  Returns
  -------
  baseline : keras.models.Sequential
      A Keras sequential model object representing the built model.
      The model architecture should consist of at least two Dense layers with
      appropriate sizes and activations, followed by an output layer with a 
      single unit and sigmoid activation function. The model should be
      compiled with the binary cross-entropy loss function, an adam optimizer, 
      and the accuracy metric.
  """
  # YOUR CODE HERE
  model = keras.models.Sequential([
    layers.Dense(input_shape//4, activation="relu"),
    layers.Dense(1, activation="sigmoid")
  ])

  model.compile(optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"])
  # Build the model

  return model
  pass


def train_model(model, X_train, y_train, epochs=10, batch_size=32, \
                class_weights=None, X_val=None, y_val=None):
    """
    Train a given Keras model using the provided training data. Use the 
    validation data to monitor the model during training.

    Parameters
    ----------
    model : keras.Sequential
        The Keras model to be trained.
    X_train : np.ndarray
        The input features for training the model.
    y_train : np.ndarray
        The target values for training the model.
    epochs : int, optional
        The number of epochs to train the model for. Default is 10.
    batch_size : int, optional
        The batch size to use during training. Default is 32.
    class_weights : dict, optional
        A dictionary containing class weights to be applied to the loss function
        during training to balance the data. Default is None.
    X_val : np.ndarray, optional
        The input features for validating the model.
    y_val : np.ndarray, optional
        The target values for validating the model.

    Returns
    -------
    model : keras.Sequential
        The trained Keras model.
    history : keras.callbacks.History
        The training history of the model.
    """
    # YOUR CODE HERE

    history = model.fit(X_train,y_train, 
                        epochs=10,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        class_weight=class_weights)
    
    return model, history
    # Train the model
    pass

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on a test set and return the accuracy of the model.

    Parameters
    ----------
    model : keras.models.Model
        The trained Keras model to evaluate.
    X_test : numpy.ndarray
        The test set features as a numpy array.
    y_test : numpy.ndarray
        The test set labels as a numpy array.

    Returns
    -------
    accuracy : float
        The accuracy of the model on the test set as a percentage.
    """
    # YOUR CODE HERE
    test_loss, test_acc = model.evaluate(X_test, y_test)
    # Evaluate model

    return test_acc
    pass



def classify_mushrooms(model, mushroom_features):
  """
  Classify mushrooms as poisonous or edible based on the given features. This 
  function is meant to be used in individual or few samples in practice.

  Parameters
  ----------
  model : keras.engine.training.Model
      The trained model to use for classification.
  mushroom_features : np.ndarray
      A numpy array containing the features of the mushrooms to classify.

  Returns
  -------
  np.ndarray
      A numpy array containing the predicted labels (0 for edible, 
      1 for poisonous) for the given mushrooms.
  """
  # YOUR CODE HERE
  # Classify mushrooms
  # Hint: same idea as the evaluation function but you return the predictions
  predictions = model.predict(mushroom_features)
  y_pred = np.argmax(predictions, axis=1)

  return y_pred
  pass

def build_model(input_shape):
  """
  Build a binary classification model using Dense layers. This model should 
  achieve over 90% recall in poisonous mushrooms, and over 90% of precision in 
  edible mushrooms.

  Parameters
  ----------
  input_shape : tuple
      The shape of the input data. This should be a tuple of integers
      specifying the number of features in the input data (e.g. (200,)).

  Returns
  -------
  model : keras.models.Sequential
      A Keras sequential model object representing the built model.
      The model architecture should consist of at least two Dense layers with
      appropriate sizes and activations, followed by an output layer with a 
      single unit and sigmoid activation function. The model should be
      compiled with the binary cross-entropy loss function, an adam optimizer, 
      and the accuracy metric.
  """
  # YOUR CODE HERE
    # YOUR CODE HERE
  model = keras.models.Sequential([
    layers.Dense (input_shape//4, activation="relu"),
    layers.Dense (10, activation="relu"),
    layers.Dense (1, activation="sigmoid")
  ])

  model.compile(optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"])
  # Build the model

  return model
  # Build the model
  pass



# Auxiliary functions below =====================================

def explore_data(X_train, y_train, y_test, y_val):
  """
  Plots the distribution of classes in the training, validation, and test sets.

  Parameters
  ----------
  X_train : np.ndarray
      A numpy array containing the features of the training set.
  y_train : np.ndarray
      A numpy array containing the labels of the training set.
  y_test : np.ndarray
      A numpy array containing the labels of the test set.
  y_val : np.ndarray
      A numpy array containing the labels of the validation set.

  Returns
  -------
  None
  """

  # Class names
  class_names = ['edible', 'poisonous']

  # Plot the distribution of classes in the training, validation, and test sets
  fig, ax = plt.subplots(1, 3, figsize=(10, 5))

  # Plot the distribution of classes in the training set
  train_class_counts = np.bincount(y_train)
  ax[0].bar(range(len(class_names)), train_class_counts)
  ax[0].set_xticks(range(len(class_names)))
  ax[0].set_xticklabels(class_names, rotation=45)
  ax[0].set_title('Training set')

  # Plot the distribution of classes in the test set
  test_class_counts = np.bincount(y_val)
  ax[1].bar(range(len(class_names)), test_class_counts)
  ax[1].set_xticks(range(len(class_names)))
  ax[1].set_xticklabels(class_names, rotation=45)
  ax[1].set_title('Val set')

  # Plot the distribution of classes in the test set
  test_class_counts = np.bincount(y_test)
  ax[2].bar(range(len(class_names)), test_class_counts)
  ax[2].set_xticks(range(len(class_names)))
  ax[2].set_xticklabels(class_names, rotation=45)
  ax[2].set_title('Test set')

  plt.show()


def plot_loss(history):
  """
  Plot the training and validation loss and accuracy.

  Parameters
  ----------
  history : keras.callbacks.History
      The history object returned by the `fit` method of a Keras model.

  Returns
  -------
  None
  """

  # Plot the training and validation loss side by side
  fig, ax = plt.subplots(1, 2, figsize=(10, 5))

  # Plot the training and validation loss
  ax[0].plot(history.history['loss'], label='train')
  ax[0].plot(history.history['val_loss'], label='val')
  ax[0].set_xlabel('Epoch')
  ax[0].set_ylabel('Loss')
  ax[0].legend()

  # Plot the training and validation accuracy
  ax[1].plot(history.history['accuracy'], label='train')
  ax[1].plot(history.history['val_accuracy'], label='val')
  ax[1].set_xlabel('Epoch')
  ax[1].set_ylabel('Accuracy')
  ax[1].legend()



