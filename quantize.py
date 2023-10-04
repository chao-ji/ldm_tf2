import tensorflow as tf
import numpy as np


class VectorQuantizer(tf.keras.layers.Layer):
  def __init__(self, vocab_size, hidden_size, beta):
    super(VectorQuantizer, self).__init__()
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size
    self._beta = beta


  def build(self, inputs_shape):
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor, the last element
        corresponds to the depth.
    """
    self.add_weight(name='kernel',
                    shape=[self._vocab_size, self._hidden_size],
                    #initializer=self._kernel_initializer,
                    dtype='float32',
                    trainable=True)
    super(VectorQuantizer, self).build(inputs_shape)

  def call(self, inputs):
    latent_size = inputs.shape[-1]
    outputs = tf.reshape(inputs, (-1, latent_size))
    #print(outputs.shape)

    outputs = tf.reduce_sum(outputs ** 2, axis=1, keepdims=True) + \
      tf.reduce_sum(self.trainable_weights[0] ** 2, axis=1) - \
      2 * tf.matmul(outputs, tf.transpose(self.trainable_weights[0]))

    #print(outputs.shape)
    min_encoding_indices = tf.argmin(outputs, axis=1)
    #print(min_encoding_indices.shape)
    z_q = tf.reshape(tf.gather(self.trainable_weights[0], min_encoding_indices), inputs.shape)
   
    loss = tf.reduce_mean((tf.stop_gradient(z_q)-inputs)**2) + self._beta * \
                   tf.reduce_mean((z_q - tf.stop_gradient(inputs)) ** 2)

    z_q = inputs + tf.stop_gradient(z_q - inputs)

    perplexity = None
    min_encodings = None

    return z_q, loss, (perplexity, min_encodings, min_encoding_indices) 




if __name__ == "__main__":

  vocab_size = 16384
  hidden_size = 4
  beta = 0.25
  inputs = tf.constant(np.random.uniform(-1, 1, (4, 32, 32, 4)).astype("float32"))
  quantizer = VectorQuantizer(vocab_size, hidden_size, beta) 
  quantizer(inputs)


