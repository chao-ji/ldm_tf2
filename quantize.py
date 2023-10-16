"""Quantize layer for VQ-VAE or VQ-GAN."""
import tensorflow as tf


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

  def call(self, latents):
    """Compute the codebook loss and return the quantized latent variable.

    Args:
      latents (Tensor of shape [batch_size, z_height, z_width, hidden_size]):
        the latent variable.

    Returns:
      quantized_latents (Tensor of shape [batch_size, z_height, z_width,
          hidden_size]): quantized latent variable.
      codebook_loss (scalar Tensor): the L2 loss between encoder outputs (
          `latents`) and quantized latent variable (`quantized_latents`).
      min_encoding_indices (Tensor of shape [batch_size * z_height * z_width]):
          the index of each quantized latent variable in the codebook.
    """
    # [batch_size * z_height * z_width, hidden_size]
    outputs = tf.reshape(latents, (-1, self._hidden_size))

    # squared pairwise Euclidean distances between
    # `outputs`: [batch_size * z_height * z_width, hidden_size]
    # `self.trainable_weights[0]`: [vocab_size, hidden_size]
    #
    # [batch_size * z_height * z_width, vocab_size]
    outputs = (
        tf.reduce_sum(outputs ** 2, axis=1, keepdims=True) +
        tf.reduce_sum(self.trainable_weights[0] ** 2, axis=1) -
        2 * tf.matmul(outputs, tf.transpose(self.trainable_weights[0]))
    )

    # [batch_size * z_height * z_width]
    min_encoding_indices = tf.argmin(outputs, axis=1)

    # [batch_size, z_height, z_width, hidden_size]
    quantized_latents= tf.reshape(
        tf.gather(self.trainable_weights[0], min_encoding_indices),
        latents.shape,
    )

    codebook_loss = (
        tf.reduce_mean((tf.stop_gradient(quantized_latents) - latents) ** 2 ) +
        self._beta *
            tf.reduce_mean((quantized_latents - tf.stop_gradient(latents)) ** 2
        )
    )

    # passed gradients from decoder unchanged back to incoder
    quantized_latents = latents + tf.stop_gradient(quantized_latents - latents)

    return quantized_latents, codebook_loss, min_encoding_indices
