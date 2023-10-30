import numpy as np
import tensorflow as tf

from distribution import DiagonalGaussian
from quantize import VectorQuantizer

GROUP_NORM_EPSILON = 1e-6


class ResidualBlock(tf.keras.layers.Layer):
  def __init__(self, channels, num_groups=32, dropout_rate=0., conv_shortcut=False,
      activation=tf.nn.swish, name=None):
    """
    """
    super(ResidualBlock, self).__init__(name=name)

    self._channels = channels
    self._num_groups = num_groups
    self._dropout_rate = dropout_rate
    self._conv_shortcut = conv_shortcut
    self._activation = activation

    self._group_norm1 = tf.keras.layers.GroupNormalization(
        num_groups, epsilon=GROUP_NORM_EPSILON, name="group_norm1"
    )
    self._conv1 = tf.keras.layers.Conv2D(
        channels, kernel_size=3, padding="same", name="conv1"
    )
    self._group_norm2 = tf.keras.layers.GroupNormalization(
        num_groups, epsilon=GROUP_NORM_EPSILON, name="group_norm2"
    )
    self._dropout = tf.keras.layers.Dropout(dropout_rate)
    self._conv2 = tf.keras.layers.Conv2D(
        channels, kernel_size=3, padding="same", name="conv2"
    )
    self._shortcut = tf.keras.layers.Dense(channels, name="shortcut")
    if conv_shortcut:
      self._conv_time = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same", name="conv_time")
    else:
      self._dense_time = tf.keras.layers.Dense(channels, name="dense_time")

  def call(self, inputs, time=None, training=False):
    outputs = self._conv1(self._activation(self._group_norm1(inputs)))

    if time is not None:
      outputs += self._dense_time(
          self._activation(time)
      )[:, tf.newaxis, tf.newaxis]

    outputs = self._conv2(
      self._dropout(
        self._activation(self._group_norm2(outputs)),
        training=training
      )
    )

    if inputs.shape[-1] != self._channels:
      inputs = self._shortcut(inputs)

    assert inputs.shape == outputs.shape
    outputs = outputs + inputs
    return outputs



class AttentionBlock(tf.keras.layers.Layer):
  """Attention block."""
  def __init__(self, channels, num_groups=32, name=None):
    super(AttentionBlock, self).__init__(name=name)
    self._channels = channels
    self._num_groups = num_groups

    self._group_norm = tf.keras.layers.GroupNormalization(
        num_groups, epsilon=GROUP_NORM_EPSILON, name="group_norm"
    )
    self._dense_query = tf.keras.layers.Dense(channels, name="dense_query")
    self._dense_key = tf.keras.layers.Dense(channels, name="dense_key")
    self._dense_value = tf.keras.layers.Dense(channels, name="dense_value")
    self._dense_output = tf.keras.layers.Dense(channels, name="dense_output")

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
        attention_weights, [batch_size, height, width, height * width]
    )
    attention_weights = tf.nn.softmax(attention_weights, axis=-1)
    attention_weights = tf.reshape(
        attention_weights, [batch_size, height, width, height, width]
    )
    outputs = tf.einsum("bhwHW,bHWc->bhwc", attention_weights, v)
    outputs = self._dense_output(outputs)
    assert inputs.shape == outputs.shape
    outputs = outputs + inputs
    return outputs



class DownBlock(tf.keras.layers.Layer):
  def __init__(self, channels, attention_resolutions, attention_type="vanilla", name=None):
    super(DownBlock, self).__init__(name=name)
    self._channels = channels
    self._attention_resolutions = attention_resolutions
    self._attention_type = attention_type

    self._residual = ResidualBlock(channels, name="residual_block")
    if attention_type == "vanilla":
      self._attention = AttentionBlock(channels, name="attention_block")

  def call(self, inputs, time=None, training=False):
    outputs = self._residual(inputs, time, training=training)
    if outputs.shape[1] in self._attention_resolutions:
      outputs = self._attention(outputs)
    return outputs



class Downsample(tf.keras.layers.Layer):
  def __init__(self, channels, resample_with_conv=True, name=None):
    super(Downsample, self).__init__(name=name)
    self._channels = channels
    self._resample_with_conv = resample_with_conv

    if resample_with_conv:
      self._conv = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=2, padding="VALID", name="conv")

  def call(self, inputs, training=False):
    if self._resample_with_conv:
      outputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]])
      #outputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
      outputs = self._conv(outputs)
    else:
      outputs = tf.nn.avg_pool2d(inputs, ksize=2, strides=2, padding="VALID")
    return outputs


class Upsample(tf.keras.layers.Layer):
  def __init__(self, channels, resample_with_conv=True, name=None):
    super(Upsample, self).__init__(name=name)
    self._channels = channels
    self._resample_with_conv = resample_with_conv

    if resample_with_conv:
      self._conv = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding="SAME", name="conv")

  def call(self, inputs, time=None, training=False):
    _, height, width, _ = inputs.shape
    outputs = tf.raw_ops.ResizeNearestNeighbor(images=inputs, size=[height * 2, width *2], align_corners=False)
    if self._resample_with_conv:
      outputs = self._conv(outputs)
    return outputs



class UpBlock(tf.keras.layers.Layer):
  def __init__(self, channels, attention_resolutions, attention_type="vanilla", name=None):
    super(UpBlock, self).__init__(name=name)
    self._channels = channels
    self._attention_resolutions = attention_resolutions
    self._attention_type = attention_type

    self._residual = ResidualBlock(channels, name="residual_block")
    if attention_type == "vanilla":
      self._attention = AttentionBlock(channels, name="attention_block")

  def call(self, inputs, time=None, training=False):
    outputs = self._residual(inputs, time, training=training)
    if outputs.shape[1] in self._attention_resolutions:
      outputs = self._attention(outputs)
    return outputs


class MiddleBlock(tf.keras.layers.Layer):
  def __init__(self, channels, name=None):
    super(MiddleBlock, self).__init__(name=name)
    self._residual1 = ResidualBlock(channels, name="residual_block1")
    self._attention = AttentionBlock(channels, name="attention_block")
    self._residual2 = ResidualBlock(channels, name="residual_block2")

  def call(self, inputs, time=None, training=False):
    outputs = self._residual1(inputs, time)
    outputs = self._attention(outputs)
    outputs = self._residual2(outputs, time)
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
      double_latent_channels=True,
      use_linear_attention=False,
      attention_type="vanilla",
      activation=tf.nn.swish,
    ):
    super(Encoder, self).__init__()

    if use_linear_attention:
      attention_type = "linear"

    self._channels = channels
    self._num_blocks = num_blocks
    self._latent_channels = latent_channels
    self._attention_resolutions = attention_resolutions
    self._dropout_rate = dropout_rate
    self._multipliers = multipliers
    self._resample_with_conv = resample_with_conv
    self._double_latent_channels = double_latent_channels
    self._use_linear_attention = use_linear_attention
    self._attention_type = attention_type
    self._activation = activation

    self._conv_in = tf.keras.layers.Conv2D(
        channels, kernel_size=3, padding="SAME", name="conv_in")

    num_resolutions = len(multipliers)
    channels_list = [channels * mul for mul in multipliers]

    down = []
    for i in range(num_resolutions):
      for j in range(num_blocks):
        down.append(DownBlock(channels_list[i], attention_resolutions, name=f"down{i}_block{j}"))
      if i < num_resolutions - 1:
        down.append(Downsample(channels_list[i], resample_with_conv=resample_with_conv, name=f"downsample{i}"))

    self._down = down
    self._middle = MiddleBlock(channels_list[-1])

    self._group_norm = tf.keras.layers.GroupNormalization(32, epsilon=GROUP_NORM_EPSILON)
    self._conv_out = tf.keras.layers.Conv2D(2 * latent_channels if double_latent_channels else latent_channels, kernel_size=3, padding="SAME", name="conv_out")


  def call(self, inputs, training=False):

    outputs = self._conv_in(inputs)
    hiddens = [outputs]

    for m in self._down:
      outputs = m(outputs, training=training)
      hiddens.append(outputs)
    outputs = self._middle(outputs)
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
      give_pre_end=False,
      tanh_out=False,
      use_linear_attention=False,
      attention_type="vanilla",
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
    self._give_pre_end = give_pre_end
    self._tanh_out = tanh_out
    self._use_linear_attention = use_linear_attention
    self._attention_type = attention_type
    self._activation = activation

    channels_list = [channels * mul for mul in multipliers]

    if use_linear_attention:
      attention_type = "linear"

    self._conv_in = tf.keras.layers.Conv2D(channels_list[-1], kernel_size=3, padding="SAME", name="conv_in")

    self._middle = MiddleBlock(channels_list[-1])

    up = []
    num_resolutions = len(multipliers)
    for i in reversed(range(num_resolutions)):
      for j in range(num_blocks + 1):
        up.append(UpBlock(channels_list[i], attention_resolutions, name=f"up{i}_block{j}"))
      if i > 0:
        up.append(Upsample(channels_list[i], name=f"upsample{i}"))
    self._up = up

    self._group_norm = tf.keras.layers.GroupNormalization(32, epsilon=GROUP_NORM_EPSILON)
    self._conv_out = tf.keras.layers.Conv2D(out_channels, kernel_size=3, padding="SAME", name="conv_out")


  def call(self, inputs, training=False):
    outputs = self._conv_in(inputs)
    outputs = self._middle(outputs)

    for m in self._up:
      outputs = m(outputs, training=training)

    if self._give_pre_end:
      return outputs

    outputs = self._conv_out(self._activation(self._group_norm(outputs)))
    if self._tanh_out:
      outputs = tf.tanh(outputs)

    return outputs



class AutoencoderKL(tf.keras.layers.Layer):
  def __init__(self, z_channels=4): 
    super(AutoencoderKL, self).__init__()
    self._encoder = Encoder(
      channels=128,
      num_blocks=2,
      latent_channels=z_channels,
      attention_resolutions=(),
      dropout_rate=0.0,
      multipliers=(1, 2, 4, 4),
      resample_with_conv=True,
      double_latent_channels=True,
      use_linear_attention=False,
      attention_type="vanilla",
      activation=tf.nn.swish,
    )
    self._quant_conv = tf.keras.layers.Dense(z_channels * 2)
    self._post_quant_conv = tf.keras.layers.Dense(z_channels)

    self._decoder = Decoder(
      channels=128,
      out_channels=3,
      num_blocks=2,
      multipliers=(1, 2, 4, 4),
      resample_with_conv=True,
      attention_resolutions=(),
      give_pre_end=False,
    )
    

  def call(self, inputs, sample_posterior=True):
    posterior = self.encode(inputs)
    if sample_posterior:
      latents = posterior.sample()
    else:
      latents = posterior.mode()

    outputs = self.decode(latents)
    return outputs, posterior


  def encode(self, inputs):
    outputs = self._encoder(inputs)
    outputs = self._quant_conv(outputs)
    mean, logvar = tf.split(outputs, 2, axis=-1)
    posterior = DiagonalGaussian(mean, logvar)
    return posterior

  def decode(self, inputs):
    outputs = self._post_quant_conv(inputs)
    outputs = self._decoder(outputs, training=False)
    return outputs

  def get_last_layer(self):
    return self._decoder._conv_out.weights[0]


class AutoencoderVQ(tf.keras.layers.Layer):
  def __init__(self, z_channels=4):
    """
      double_z: false
      z_channels: 4
      resolution: 256
      in_channels: 3
      out_ch: 3
      ch: 128
      ch_mult:
      - 1
      - 2
      - 2
      - 4
      num_res_blocks: 2
      attn_resolutions:
      - 32
      dropout: 0.0
    """
    super(AutoencoderVQ, self).__init__()
    self._encoder = Encoder(
      channels=128,
      num_blocks=2,
      latent_channels=z_channels,
      attention_resolutions=(32,),
      dropout_rate=0.0,
      multipliers=(1, 2, 2, 4),
      resample_with_conv=True,
      double_latent_channels=False,
      use_linear_attention=False,
      attention_type="vanilla",
      activation=tf.nn.swish,
    )
    self._quant_conv = tf.keras.layers.Dense(z_channels)
    self._post_quant_conv = tf.keras.layers.Dense(z_channels)
    self._decoder = Decoder(
      channels=128,
      out_channels=3,
      num_blocks=2,
      multipliers=(1, 2, 2, 4),
      resample_with_conv=True,
      attention_resolutions=(32,),
      give_pre_end=False,
    )
    self._quantize = VectorQuantizer(vocab_size=16384, hidden_size=4, beta=0.25)


  def encode(self, x):
    h = self._encoder(x)
    h = self._quant_conv(h)
    quant, emb_loss, info = self._quantize(h)
    return quant, emb_loss, info

  def decode(self, quant):
    quant = self._post_quant_conv(quant)
    dec = self._decoder(quant)
    return dec

  def call(self, inputs, return_pred_indices=False):
    quant, diff, ind = self.encode(inputs)
    dec = self.decode(quant)
    if return_pred_indices:
      return dec, diff, ind
    return dec, diff

  def get_last_layer(self):
    return self._decoder._conv_out.weights[0]




if __name__ == "__main__":
  import numpy as np
  scale_factor = 0.18215
  inputs = np.load("/home/chaoji/work/genmo/diffusion/latent-diffusion/samples_ddim.npy").transpose(0, 2, 3, 1)


  encoder = Encoder(
      channels=128,
      num_blocks=2,
      latent_channels=4,
      attention_resolutions=(),
      dropout_rate=0.0,
      multipliers=(1, 2, 4, 4),
      resample_with_conv=True,
      double_latent_channels=True,
      use_linear_attention=False,
      attention_type="vanilla",
      activation=tf.nn.swish,      
  )

