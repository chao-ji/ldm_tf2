"""UNet backbone for diffusion model."""
import tensorflow as tf

from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    LayerNormalization,
    GroupNormalization,
)

from transformer import Projection


class Downsample(tf.keras.layers.Layer):
  def __init__(self, channels, resample_with_conv=True):
    super(Downsample, self).__init__()
    self._channels = channels
    self._resample_with_conv = resample_with_conv

    if resample_with_conv:
      self._conv = Conv2D(channels, kernel_size=3, strides=2, padding="VALID")

  def call(self, inputs, training=False):
    if self._resample_with_conv:
      outputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
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


class UNet(tf.keras.layers.Layer):
  def __init__(
      self,
      model_channels=320,
      out_channels=4,
      num_blocks=2,
      attention_resolutions=[4, 2, 1],
      dropout_rate=0.1,
      channel_mult=[1, 2, 4, 4],
      num_heads=8,
    ):
    super(UNet, self).__init__()
    self._model_channels = model_channels
    self._out_channels = out_channels
    self._num_blocks = num_blocks
    self._attention_resolutions = attention_resolutions
    self._dropout_rate = dropout_rate
    self._channel_mult = channel_mult
    self._num_heads = num_heads

    self._conv_in = Conv2D(model_channels, 3, strides=1, padding="SAME")
    self._time_dense1 = Dense(model_channels * 4, activation="silu")
    self._time_dense2 = Dense(model_channels * 4)

    self._input_blocks = []
    for i, mult in enumerate(channel_mult):
      for j in range(num_blocks):
        self._input_blocks.append(
          InputBlock(
              channels=model_channels * mult,
              num_heads=num_heads,
              size_per_head=40 * mult,
              hidden_size=1280,
              dropout_rate=dropout_rate,
              use_spatial_transformer=i < len(channel_mult) - 1,
          )
        )
      if i < len(channel_mult) - 1:
        self._input_blocks.append(
            InputBlock(channels=model_channels * mult, use_downsample=True))

    self._middle_block = MiddleBlock(
        channels=model_channels * channel_mult[-1],
        context_channels=1280,
        num_heads=num_heads,
        size_per_head=40*channel_mult[-1],
        dropout_rate=dropout_rate,
    )

    self._output_blocks = []
    for i, mult in list(enumerate(channel_mult))[::-1]:
      for j in range(num_blocks + 1):
        self._output_blocks.append(
          OutputBlock(
              channels=model_channels * mult,
              num_heads=num_heads,
              size_per_head=40 * mult,
              hidden_size=1280,
              dropout_rate=dropout_rate,
              use_spatial_transformer=i < len(channel_mult) - 1,
              use_upsample=i > 0 and j == num_blocks,
          )
        )

    self._groupnorm = GroupNormalization(groups=32, epsilon=1e-5)
    self._conv_out = Conv2D(out_channels, 3, strides=1, padding="SAME")
 
  def call(self, inputs, time, context=None, y=None, training=False):
    """
    Args:
      inputs (Tensor): [batch_size, height, width, channels]
      time (Tensor): [batch_size]
      context (Tensor): [batch_size, seq_len, context_channels]
    """
    outputs = self._conv_in(inputs)
    time_emb = get_time_embedding(time, self._model_channels)
    time_embedding = self._time_dense2(self._time_dense1(time_emb))

    hiddens = [outputs]
    for block in self._input_blocks:
      outputs = block(outputs, time_embedding, context, training)
      hiddens.append(outputs)
    outputs = self._middle_block(outputs, time_embedding, context, training)
    for block in self._output_blocks:
      outputs = tf.concat([outputs, hiddens.pop()], axis=-1)
      outputs = block(outputs, time_embedding, context, training)
    outputs = self._conv_out(tf.nn.silu(self._groupnorm(outputs)))
    return outputs


class InputBlock(tf.keras.layers.Layer):
  def __init__(
      self,
      channels,
      dropout_rate=0.1,
      use_spatial_transformer=False,
      use_downsample=False,
      num_heads=8,
      size_per_head=40,
      hidden_size=512,
    ):
    super(InputBlock, self).__init__()
    self._channels = channels
    self._dropout_rate = dropout_rate
    self._use_spatial_transformer = use_spatial_transformer
    self._use_downsample = use_downsample 
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._hidden_size = hidden_size

    self._residual = ResidualBlock(channels, dropout_rate)
    if use_spatial_transformer:
      self._spatial_transformer = SpatialTransformer(
          num_heads, size_per_head, hidden_size, dropout_rate)
    if use_downsample:
      self._downsample = Downsample(channels)

  def call(self, inputs, time_embedding=None, context=None, training=False):
    if self._use_downsample:
      outputs = self._downsample(inputs)
    else:
      outputs = self._residual(inputs, time_embedding, training=training)
      if self._use_spatial_transformer:
        outputs = self._spatial_transformer(outputs, context, training=training)
    return outputs


class MiddleBlock(tf.keras.layers.Layer):
  def __init__(
      self,
      channels,
      context_channels,
      num_heads,
      size_per_head,
      dropout_rate=0.1,
    ):
    super(MiddleBlock, self).__init__()
    self._channels = channels
    self._context_channels = context_channels
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._dropout_rate = dropout_rate

    self._residual1 = ResidualBlock(channels, dropout_rate)
    self._spatial_transformer = SpatialTransformer(
        num_heads, size_per_head, context_channels, dropout_rate)
    self._residual2 = ResidualBlock(channels, dropout_rate)

  def call(self, inputs, time_embedding, context, training=False):
    outputs = self._residual2(
        self._spatial_transformer(
            self._residual1(inputs, time_embedding, training=training),
            context,
            training=training,
        ),
        time_embedding,
        training=training,
    )
    return outputs


class OutputBlock(tf.keras.layers.Layer):
  def __init__(
      self,
      channels,
      dropout_rate=0.1,
      use_spatial_transformer=False,
      use_upsample=False,
      num_heads=8,
      size_per_head=40,
      hidden_size=512,
    ):
    super(OutputBlock, self).__init__()
    self._channels = channels
    self._dropout_rate = dropout_rate
    self._use_spatial_transformer = use_spatial_transformer
    self._use_upsample = use_upsample
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._hidden_size = hidden_size

    self._residual = ResidualBlock(channels, dropout_rate)
    if self._use_spatial_transformer:
      self._spatial_transformer = SpatialTransformer(
          num_heads, size_per_head, hidden_size, dropout_rate)
    if self._use_upsample:
      self._upsample = Upsample(channels)

  def call(self, inputs, time_embedding, context=None, training=False):
    outputs = self._residual(inputs, time_embedding, training=training)
    if self._use_spatial_transformer:
      outputs = self._spatial_transformer(outputs, context, training=training)
    if self._use_upsample:
      outputs = self._upsample(outputs)
    return outputs

 
class CrossAttention(tf.keras.layers.Layer):
  def __init__(
      self, num_heads=8, size_per_head=64, dropout_rate=0., hidden_size=None):
    super(CrossAttention, self).__init__()

    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._dropout_rate = dropout_rate
    self._hidden_size = (num_heads * size_per_head if hidden_size is None
        else hidden_size)

    self._dense_layer_query = Projection(
        num_heads, size_per_head, num_heads * size_per_head, mode="split")
    self._dense_layer_key = Projection(
        num_heads, size_per_head, hidden_size, mode="split")
    self._dense_layer_value = Projection(
        num_heads, size_per_head, hidden_size, mode="split")
    self._dense_layer_output = Projection(
        num_heads, size_per_head, num_heads * size_per_head, True, mode="merge")
    self._dropout_layer = Dropout(dropout_rate)

  def call(self, query, context=None, attention_mask=None, training=False):
    context = query if context is None else context

    # [batch_size, q_seq_len, num_heads, size_per_head]
    q = self._dense_layer_query(query)

    # [batch_size, c_seq_len, num_heads, size_per_head]
    k = self._dense_layer_key(context)
    v = self._dense_layer_value(context)

    # [batch_size, num_heads, q_seq_len, c_seq_len]
    attention_weights = tf.einsum("NQHS,NCHS->NHQC", q, k)
    attention_weights *= self._size_per_head ** -0.5
    if attention_mask is not None:
      attention_weights += attention_mask * NEG_INF
    attention_weights = tf.nn.softmax(attention_weights, axis=3)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    outputs = tf.einsum("NHQC,NCHS->NQHS", attention_weights, v)

    # [batch_size, q_seq_len, hidden_size]
    outputs = self._dense_layer_output(outputs)
    outputs = self._dropout_layer(outputs, training=training)
    return outputs    


class BasicTransformerBlock(tf.keras.layers.Layer):
  def __init__(
      self, num_heads=8, size_per_head=64, dropout_rate=0.1, hidden_size=512):
    super(BasicTransformerBlock, self).__init__()

    self._att_layer1 = CrossAttention(num_heads, size_per_head, dropout_rate)
    self._att_layer2 = CrossAttention(
        num_heads, size_per_head, dropout_rate, hidden_size)
    self._ffn_layer = FeedForward(num_heads * size_per_head,)
    self._layernorm1 = LayerNormalization(epsilon=1e-05)
    self._layernorm2 = LayerNormalization(epsilon=1e-05)
    self._layernorm3 = LayerNormalization(epsilon=1e-05)

  def call(self, inputs, context=None, training=False):
    outputs = self._att_layer1(
        self._layernorm1(inputs), training=training) + inputs
    outputs = self._att_layer2(
        self._layernorm2(outputs), context, training=training) + outputs
    outputs = self._ffn_layer(self._layernorm3(outputs)) + outputs
    return outputs


class GEGLU(tf.keras.layers.Layer):
  def __init__(self, channels):
    super(GEGLU, self).__init__()
    self._dense_layer = Dense(channels * 2)

  def call(self, inputs):
    outputs, gate = tf.split(self._dense_layer(inputs), 2, axis=-1)
    outputs = outputs * tf.nn.gelu(gate)
    return outputs


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, channels, multiplier=4, dropout_rate=0.):
    super(FeedForward, self).__init__()
    self._geglu_layer = GEGLU(channels * multiplier)
    self._dense_layer = Dense(channels)
    self._dropout_layer = Dropout(dropout_rate)

  def call(self, inputs, training=False):
    outputs = self._dense_layer(
        self._dropout_layer(self._geglu_layer(inputs), training=training))
    return outputs


class SpatialTransformer(tf.keras.layers.Layer):
  def __init__(
      self, num_heads=8, size_per_head=40, hidden_size=512, dropout_rate=0.1):
    super(SpatialTransformer, self).__init__()
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._hidden_size = hidden_size
    self._dropout_rate = dropout_rate

    self._dense1 = Dense(num_heads * size_per_head)
    self._block = BasicTransformerBlock(
        num_heads, size_per_head, dropout_rate, hidden_size,)
    self._dense2 = Dense(num_heads * size_per_head)
    self._groupnorm = GroupNormalization(groups=32, epsilon=1e-6)

  def call(self, inputs, context=None, training=False):
    batch_size, height, width, channels = inputs.shape
    outputs = self._groupnorm(inputs) 
    outputs = self._dense1(outputs)
    outputs = tf.reshape(outputs, [batch_size, height * width, channels])
    outputs = self._block(outputs, context, training=training)
    outputs = tf.reshape(outputs, [batch_size, height, width, channels])
    outputs = self._dense2(outputs)
    outputs += inputs
    return outputs


class ResidualBlock(tf.keras.layers.Layer):
  def __init__(self, channels, dropout_rate):
    super(ResidualBlock, self).__init__()
    self._channels = channels
    self._dropout_rate = dropout_rate

    self._group_norm_1 = GroupNormalization(groups=32, epsilon=1e-5)
    self._conv2d_1 = Conv2D(channels, kernel_size=3, strides=1, padding="SAME")
    self._dense = Dense(channels)
    self._group_norm_2 = GroupNormalization(groups=32, epsilon=1e-5)
    self._conv2d_2 = Conv2D(channels, kernel_size=3, strides=1, padding="SAME")
    self._shortcut = Dense(channels)
    self._dropout = Dropout(dropout_rate)

  def call(self, inputs, time_embedding, training=False):
    outputs = tf.nn.silu(self._group_norm_1(inputs)) 
    outputs = self._conv2d_1(outputs)

    time_embedding = self._dense(tf.nn.silu(time_embedding))
    time_embedding = time_embedding[:, tf.newaxis, tf.newaxis] 
    outputs = outputs + time_embedding

    outputs = tf.nn.silu(self._group_norm_2(outputs))
    outputs = self._dropout(outputs, training=training)
    outputs = self._conv2d_2(outputs)

    if inputs.shape[-1] != outputs.shape[-1]:
      inputs = self._shortcut(inputs)

    outputs = outputs + inputs
    return outputs


def get_time_embedding(time, channels, max_time=10000):
    """
    Create sinusoidal timestep embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    Args:
      time (Tensor):
      channels (Tensor):

    Returns:
      embedding (Tensor): 
    """
    half = channels // 2
    freqs = tf.exp(
        -tf.math.log(tf.cast(max_time, "float32")) *
          tf.range(start=0, limit=half, dtype="float32") / half
    )
    args = tf.cast(time[:, None], "float32") * freqs[None]
    embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
    if channels % 2:
      embedding = tf.concat(
          [embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
    return embedding
