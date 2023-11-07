import numpy as np

import tensorflow as tf


class DiagonalGaussian(object):
  def __init__(self, mean, logvar, deterministic=False):
    """
    Args:
      mean (Tensor of shape [batch_size, z_height, z_width, z_channels]): the
        predicted mean of the latent variable.
      logvar (Tensor of shape [batch_size, z_height, z_width, z_channels]): the
        predicted logvar of the latent variable.
    """
    self._mean = mean
    self._logvar = tf.clip_by_value(logvar, -30.0, 20.0)
    self._deterministic = deterministic
    self._std = tf.exp(0.5 * logvar)
    self._var = tf.exp(logvar)
    if deterministic:
      self._var = self._std = tf.zeros_like(self._mean)

  def sample(self):
    outputs = self._mean + self._std * tf.random.normal(self._std.shape)
    return outputs

  def kl(self, other=None):
    if self._deterministic:
      return tf.constant([0.], dtype="float32") 
    else:
      if other is None:
        return 0.5 * tf.reduce_sum(tf.pow(self._mean, 2)
                                       + self._var - 1.0 - self._logvar,
                                       axis=[1, 2, 3])
      else:
        return 0.5 * tf.reduce_sum(
                    tf.pow(self._mean - other._mean, 2) / other._var
                    + self._var / other._var - 1.0 - self._logvar + other._logvar,
                    axis=[1, 2, 3])  

  def nll(self, sample, axis=[1, 2, 3]):
    if self._deterministic:
      return tf.constant([0.], dtype="float32")
 
    logtwopi = np.log(2.0 * np.pi)
    return 0.5 * tf.reduce_sum(
            logtwopi + self._logvar + tf.pow(sample - self._mean, 2) / self._var,
            axis=axis)

  def mode(self):
    return self._mean
