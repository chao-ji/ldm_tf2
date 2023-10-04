import tensorflow as tf
import numpy as np


class NLayerDiscriminator(tf.keras.layers.Layer):
  def __init__(self, channels=64, num_layers=3):
    super(NLayerDiscriminator, self).__init__()

    use_bias = False
    pad_fn = lambda inputs: tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]]) 
    activation_fn = lambda inputs: tf.nn.leaky_relu(inputs, 0.2)

    layers = [
      pad_fn,
      tf.keras.layers.Conv2D(channels, kernel_size=4, strides=2, padding="VALID"),
      activation_fn, 
    ]

    for n in range(1, num_layers):
      layers.extend([
        pad_fn,
        tf.keras.layers.Conv2D(min(2 ** n, 8) * channels, kernel_size=4, strides=2, padding="VALID", use_bias=use_bias),
        tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9),
        activation_fn, 
      ]
    )

    nf_mult = min(2 ** num_layers, 8)

    layers.extend([
        pad_fn,
        #tf.keras.layers.Conv2D(min(2 ** nf_mult, 8) * channels, kernel_size=4, strides=1, padding="VALID", use_bias=use_bias),
        tf.keras.layers.Conv2D(nf_mult * channels, kernel_size=4, strides=1, padding="VALID", use_bias=use_bias),
        tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9),
        activation_fn, 
      ]
    )
    layers.extend([pad_fn,
        tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding="VALID")]
)
    self._layers = layers 

  def call(self, inputs):
    outputs = inputs
    for layer in self._layers:
      outputs = layer(outputs)

    return outputs

