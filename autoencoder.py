"""Autoencoders with KL Divergence regularization or Vector Quantization
 regularization.
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GroupNormalization

from distribution import DiagonalGaussian
from quantize import VectorQuantizer


GROUP_NORM_EPS = 1e-6

class ResidualBlock(tf.keras.layers.Layer):
  def __init__(
      self,
      channels,
      num_groups=32,
      dropout_rate=0.,
      conv_shortcut=False,
      activation=tf.nn.swish,
    ):
    """
    """
    super(ResidualBlock, self).__init__()
    self._channels = channels
    self._num_groups = num_groups
    self._dropout_rate = dropout_rate
    self._conv_shortcut = conv_shortcut
    self._activation = activation

    self._group_norm1 = GroupNormalization(num_groups, epsilon=GROUP_NORM_EPS)
    self._conv1 = Conv2D(channels, kernel_size=3, padding="same")
    self._group_norm2 = GroupNormalization(num_groups, epsilon=GROUP_NORM_EPS)
    self._dropout = Dropout(dropout_rate)
    self._conv2 = Conv2D(channels, kernel_size=3, padding="same")
    self._shortcut = Dense(channels)
    if conv_shortcut:
      self._conv_time = Conv2D(channels, kernel_size=3, padding="same")
    else:
      self._dense_time = Dense(channels)

  def call(self, inputs, time=None, training=False):
    outputs = self._conv1(self._activation(self._group_norm1(inputs)))

    if time is not None:
      outputs += self._dense_time(
          self._activation(time))[:, tf.newaxis, tf.newaxis]

    outputs = self._conv2(
        self._dropout(
          self._activation(self._group_norm2(outputs)), training=training))

    if inputs.shape[-1] != self._channels:
      inputs = self._shortcut(inputs)

    assert inputs.shape == outputs.shape
    outputs = outputs + inputs
    return outputs


class AttentionBlock(tf.keras.layers.Layer):
  """Attention block."""
  def __init__(self, channels, num_groups=32):
    super(AttentionBlock, self).__init__()
    self._channels = channels
    self._num_groups = num_groups

    self._group_norm = GroupNormalization(num_groups, epsilon=GROUP_NORM_EPS)
    self._dense_query = Dense(channels)
    self._dense_key = Dense(channels)
    self._dense_value = Dense(channels)
    self._dense_output = Dense(channels)

  def call(self, inputs):
    batch_size, height, width, channels = inputs.shape
    outputs = self._group_norm(inputs)

    # [batch_size, height, width, channels]
    q = self._dense_query(outputs)
    # [batch_size, height, width, channels]
    k = self._dense_key(outputs)
    # [batch_size, height, width, channels]
    v = self._dense_value(outputs)

    # [batch_size, height, width, height, width]
    attention_weights = tf.einsum(
        "bhwc,bHWc->bhwHW", q, k) * self._channels ** -0.5
    attention_weights = tf.reshape(
        attention_weights, [batch_size, height, width, height * width])
    attention_weights = tf.nn.softmax(attention_weights, axis=-1)
    attention_weights = tf.reshape(
        attention_weights, [batch_size, height, width, height, width])
    outputs = tf.einsum("bhwHW,bHWc->bhwc", attention_weights, v)
    outputs = self._dense_output(outputs)
    assert inputs.shape == outputs.shape
    outputs = outputs + inputs
    return outputs


class DownBlock(tf.keras.layers.Layer):
  def __init__(
      self,
      channels,
      attention_resolutions,
      dropout_rate=0.,
    ):
    super(DownBlock, self).__init__()
    self._channels = channels
    self._attention_resolutions = attention_resolutions
    self._dropout_rate = dropout_rate

    self._residual = ResidualBlock(channels, dropout_rate=dropout_rate)
    self._attention = AttentionBlock(channels)

  def call(self, inputs, time=None, training=False):
    outputs = self._residual(inputs, time, training=training)
    if outputs.shape[1] in self._attention_resolutions:
      outputs = self._attention(outputs)
    return outputs


class Downsample(tf.keras.layers.Layer):
  def __init__(self, channels, resample_with_conv=True):
    super(Downsample, self).__init__()
    self._channels = channels
    self._resample_with_conv = resample_with_conv

    if resample_with_conv:
      self._conv = Conv2D(channels, kernel_size=3, strides=2, padding="VALID")

  def call(self, inputs, training=False):
    if self._resample_with_conv:
      outputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]])
      #outputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
      outputs = self._conv(outputs)
    else:
      outputs = tf.nn.avg_pool2d(inputs, ksize=2, strides=2, padding="VALID")
    return outputs


class Upsample(tf.keras.layers.Layer):
  def __init__(self, channels, resample_with_conv=True):
    super(Upsample, self).__init__()
    self._channels = channels
    self._resample_with_conv = resample_with_conv

    if resample_with_conv:
      self._conv = Conv2D(channels, kernel_size=3, strides=1, padding="SAME")

  def call(self, inputs, time=None, training=False):
    _, height, width, _ = inputs.shape
    outputs = tf.raw_ops.ResizeNearestNeighbor(
        images=inputs, size=[height * 2, width *2], align_corners=False)
    if self._resample_with_conv:
      outputs = self._conv(outputs)
    return outputs


class UpBlock(tf.keras.layers.Layer):
  def __init__(
      self,
      channels,
      attention_resolutions,
      dropout_rate=0.,
    ):
    super(UpBlock, self).__init__()
    self._channels = channels
    self._attention_resolutions = attention_resolutions
    self._dropout_rate = dropout_rate

    self._residual = ResidualBlock(channels, dropout_rate=dropout_rate)
    self._attention = AttentionBlock(channels)

  def call(self, inputs, time=None, training=False):
    outputs = self._residual(inputs, time, training=training)
    if outputs.shape[1] in self._attention_resolutions:
      outputs = self._attention(outputs)
    return outputs


class MiddleBlock(tf.keras.layers.Layer):
  def __init__(self, channels, dropout_rate=0.):
    super(MiddleBlock, self).__init__()
    self._channels = channels
    self._dropout_rate = dropout_rate

    self._residual1 = ResidualBlock(channels, dropout_rate=dropout_rate)
    self._attention = AttentionBlock(channels)
    self._residual2 = ResidualBlock(channels, dropout_rate=dropout_rate)

  def call(self, inputs, time=None, training=False):
    outputs = self._residual1(inputs, time, training=training)
    outputs = self._attention(outputs)
    outputs = self._residual2(outputs, time, training=training)
    return outputs


class Encoder(tf.keras.layers.Layer):
  def __init__(
      self,
      channels,
      num_blocks=2,
      latent_channels=4,
      attention_resolutions=(),
      dropout_rate=0.0,
      multipliers=(1, 2, 4, 8),
      resample_with_conv=True,
      activation=tf.nn.swish,
    ):
    super(Encoder, self).__init__()
    self._channels = channels
    self._num_blocks = num_blocks
    self._latent_channels = latent_channels
    self._attention_resolutions = attention_resolutions
    self._dropout_rate = dropout_rate
    self._multipliers = multipliers
    self._resample_with_conv = resample_with_conv
    self._activation = activation

    self._conv_in = tf.keras.layers.Conv2D(
        channels, kernel_size=3, padding="SAME")

    num_resolutions = len(multipliers)
    channels_list = [channels * mul for mul in multipliers]

    down = []
    for i in range(num_resolutions):
      for j in range(num_blocks):
        down.append(
            DownBlock(channels_list[i], attention_resolutions, dropout_rate))
      if i < num_resolutions - 1:
        down.append(Downsample(channels_list[i], resample_with_conv))

    self._down = down
    self._middle = MiddleBlock(channels_list[-1])

    self._group_norm = GroupNormalization(32, epsilon=GROUP_NORM_EPS)
    self._conv_out = Conv2D(latent_channels, kernel_size=3, padding="SAME")

  def call(self, inputs, training=False):
    outputs = self._conv_in(inputs)
    hiddens = [outputs]

    for m in self._down:
      outputs = m(outputs, training=training)
      hiddens.append(outputs)
    outputs = self._middle(outputs, training=training)
    outputs = self._conv_out(self._activation(self._group_norm(outputs)))
    return outputs


class Decoder(tf.keras.layers.Layer):
  def __init__(
      self,
      channels,
      out_channels=3,
      num_blocks=2,
      dropout_rate=0.0,
      resample_with_conv=True,
      attention_resolutions=(),
      multipliers=(1, 2, 4, 8),
      activation=tf.nn.swish,
    ):
    super(Decoder, self).__init__()
    self._channels = channels
    self._out_channels = out_channels
    self._num_blocks = num_blocks
    self._dropout_rate = dropout_rate
    self._resample_with_conv = resample_with_conv
    self._attention_resolutions = attention_resolutions
    self._multipliers = multipliers
    self._activation = activation

    channels_list = [channels * mul for mul in multipliers]
    self._conv_in = Conv2D(channels_list[-1], kernel_size=3, padding="SAME")
    self._middle = MiddleBlock(channels_list[-1])

    up = []
    num_resolutions = len(multipliers)
    for i in reversed(range(num_resolutions)):
      for j in range(num_blocks + 1):
        up.append(
            UpBlock(channels_list[i], attention_resolutions, dropout_rate))
      if i > 0:
        up.append(Upsample(channels_list[i], resample_with_conv))
    self._up = up

    self._group_norm = GroupNormalization(32, epsilon=GROUP_NORM_EPS)
    self._conv_out = Conv2D(out_channels, kernel_size=3, padding="SAME")

  def call(self, inputs, training=False):
    outputs = self._conv_in(inputs)
    outputs = self._middle(outputs, training=training)

    for m in self._up:
      outputs = m(outputs, training=training)
    outputs = self._conv_out(self._activation(self._group_norm(outputs)))
    return outputs


class AutoencoderKL(tf.keras.layers.Layer):
  def __init__(
      self,
      latent_channels=4,
      channels=128,
      num_blocks=2,
      attention_resolutions=(),
      dropout_rate=0.,
      multipliers=(1, 2, 4, 4),
      resample_with_conv=True, 
    ): 
    super(AutoencoderKL, self).__init__()
    self._latent_channels = latent_channels
    self._channels = channels
    self._num_blocks = num_blocks
    self._attention_resolutions = attention_resolutions
    self._dropout_rate = dropout_rate
    self._multipliers = multipliers
    self._resample_with_conv = resample_with_conv

    self._encoder = Encoder(
        channels=channels,
        num_blocks=num_blocks,
        latent_channels=latent_channels * 2,
        attention_resolutions=(),
        dropout_rate=dropout_rate,
        multipliers=multipliers,
        resample_with_conv=resample_with_conv,
        activation=tf.nn.swish,
    )
    self._quant_conv = Dense(latent_channels * 2)
    self._post_quant_conv = Dense(latent_channels)
    self._decoder = Decoder(
        channels=channels,
        out_channels=3,
        num_blocks=num_blocks,
        multipliers=multipliers,
        resample_with_conv=resample_with_conv,
        attention_resolutions=(),
        dropout_rate=dropout_rate,
        activation=tf.nn.swish,
    )

  def call(self, inputs, sample_posterior=True, training=False):
    posterior = self.encode(inputs, training=training)
    if sample_posterior:
      latents = posterior.sample()
    else:
      latents = posterior.mode()

    outputs = self.decode(latents, training=training)
    return outputs, posterior

  def encode(self, inputs, training=False):
    outputs = self._encoder(inputs, training=training)
    outputs = self._quant_conv(outputs)
    mean, logvar = tf.split(outputs, 2, axis=-1)
    posterior = DiagonalGaussian(mean, logvar)
    return posterior

  def decode(self, inputs, training=False):
    outputs = self._post_quant_conv(inputs)
    outputs = self._decoder(outputs, training=training)
    return outputs

  def get_last_layer(self):
    return self._decoder._conv_out.weights[0]


class AutoencoderVQ(tf.keras.layers.Layer):
  def __init__(
      self,
      latent_channels=4,
      channels=128,
      num_blocks=2,
      dropout_rate=0,
      multipliers=(1, 2, 2, 4),
      resample_with_conv=True,
      attention_resolutions=(32,),
      vocab_size=16384,
      beta=0.25,
    ):
    """
    """
    super(AutoencoderVQ, self).__init__()
    self._latent_channels = latent_channels
    self._channels = channels
    self._num_blocks = num_blocks
    self._dropout_rate = dropout_rate
    self._multipliers = multipliers
    self._resample_with_conv = resample_with_conv
    self._attention_resolutions = attention_resolutions
    self._vocab_size = vocab_size
    self._beta = beta

    self._encoder = Encoder(
        channels=channels,
        num_blocks=num_blocks,
        latent_channels=latent_channels,
        attention_resolutions=attention_resolutions,
        dropout_rate=dropout_rate,
        multipliers=multipliers,
        resample_with_conv=resample_with_conv,
        activation=tf.nn.swish,
    )
    self._quant_conv = Dense(latent_channels)
    self._quantize = VectorQuantizer(
        vocab_size=vocab_size, hidden_size=latent_channels, beta=beta)
    self._post_quant_conv = Dense(latent_channels)
    self._decoder = Decoder(
        channels=channels,
        out_channels=3,
        num_blocks=num_blocks,
        multipliers=multipliers,
        resample_with_conv=resample_with_conv,
        attention_resolutions=attention_resolutions,
        dropout_rate=dropout_rate,
        activation=tf.nn.swish,
    )

  def encode(self, inputs, only_encode=False, training=False):
    latents = self._encoder(inputs, training=training)
    latents = self._quant_conv(latents)
    if only_encode:
      return latents
    else:
      latents, codebook_loss, indices = self._quantize(latents)
      return latents, codebook_loss, indices

  def decode(self, latents, force_quantize=False, training=False):
    if force_quantize:
      latents = self._quantize(latents)

    latents = self._post_quant_conv(latents)
    outputs = self._decoder(latents, training=training)
    return outputs

  def call(self, inputs, return_indices=False, training=False):
    latents, codebook_loss, indices = self.encode(inputs, training=training)
    outputs = self.decode(latents, training=training)
    if return_indices:
      return outputs, codebook_loss, indices
    else:
      return outputs, codebook_loss

  def get_last_layer(self):
    return self._decoder._conv_out.weights[0]
