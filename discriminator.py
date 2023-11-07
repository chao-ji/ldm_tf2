"""Discriminator for training autoencoders."""
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, BatchNormalization


class Discriminator(tf.keras.layers.Layer):
  def __init__(self, channels=64, num_layers=3):
    super(Discriminator, self).__init__()
    self._channels = channels
    self._num_layers = num_layers

    pad_fn = lambda inputs: tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
    activation_fn = lambda inputs: tf.nn.leaky_relu(inputs, 0.2)

    layers = [pad_fn, Conv2D(channels, 4, 2, "VALID"), activation_fn]
    for n in range(1, num_layers):
      layers.extend([
          pad_fn,
          Conv2D(min(2 ** n, 8) * channels, 4, 2, "VALID", use_bias=False),
          BatchNormalization(epsilon=1e-5, momentum=0.9),
          activation_fn])

    multiplier = min(2 ** num_layers, 8)
    layers.extend([
        pad_fn,
        Conv2D(multiplier * channels, 4, 1, "VALID", use_bias=False),
        BatchNormalization(epsilon=1e-5, momentum=0.9),
        activation_fn])
    layers.extend([pad_fn, Conv2D(1, 4, strides=1, padding="VALID")])
    self._layers = layers 

  def call(self, inputs):
    outputs = inputs
    for layer in self._layers:
      outputs = layer(outputs)
    return outputs
