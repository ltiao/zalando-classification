"""Models module."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization


def build_model(output_dim, num_layers=2, num_units=64, activation="relu",
                batch_norm=False, *args, **kwargs):
    """
    Instantiate a multi-layer perceptron with rectangular-shaped hidden
    layers.
    """
    model = Sequential()

    for l in range(num_layers):

        model.add(Dense(num_units, activation=activation, *args, **kwargs))

        if batch_norm:
            model.add(BatchNormalization())

    model.add(Dense(output_dim, activation="sigmoid"))

    return model
