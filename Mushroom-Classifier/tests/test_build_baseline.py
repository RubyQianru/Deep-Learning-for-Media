from utils import build_baseline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam

def test_build_baseline():
    input_shape = 200
    model = build_baseline(input_shape)

    assert isinstance(model, Sequential)
    assert len(model.layers) >= 3
    assert isinstance(model.layers[0], Dense)
    assert model.layers[0].input_shape == (None, input_shape)
    assert isinstance(model.layers[-1], Dense)
    assert model.layers[-1].units == 1
    assert model.layers[-1].activation == sigmoid
    assert model.loss == 'binary_crossentropy'
    assert isinstance(model.optimizer, Adam)
