import tensorflow as tf

from transformer import Projection

GROUP_NORM_EPSILON = 1e-6



class Downsample(tf.keras.layers.Layer):
  def __init__(self, channels, resample_with_conv=True, name=None):
    super(Downsample, self).__init__(name=name)
    self._channels = channels
    self._resample_with_conv = resample_with_conv

    if resample_with_conv:
      self._conv = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=2, padding="VALID", name="conv")

  def call(self, inputs, training=False):
    if self._resample_with_conv:
      outputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
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


class UNet(tf.keras.layers.Layer):
  def __init__(
    self,
    image_size=32,
    in_channels=4,
    model_channels=320,
    out_channels=4,
    num_res_blocks=2,
    attention_resolutions=[4, 2, 1],
    dropout_rate=0.1,
    channel_mult=[1, 2, 4, 4],
    conv_resample=True,
    dims=2,
    num_classes=None,
    use_checkpoint=True,
    use_fp16=False,
    num_heads=8,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    resblock_updown=False,
    use_new_attention_order=False,
    use_spatial_transformer=True,
    transformer_depth=1,
    context_dim=1280,
    n_embed=None,
    legacy=False,
  ):
    super(UNet, self).__init__()

    self._channel_mult = channel_mult
    self._attention_resolutions = attention_resolutions
    self._model_channels = model_channels

    self._conv_in = tf.keras.layers.Conv2D(model_channels, kernel_size=3, strides=1, padding="SAME")   
    self._time_dense1 = tf.keras.layers.Dense(model_channels * 4, activation="silu")
    self._time_dense2 = tf.keras.layers.Dense(model_channels * 4)

    self._input_blocks = []
    for i, mult in enumerate(channel_mult):
      for j in range(num_res_blocks):
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
        self._input_blocks.append(InputBlock(channels=model_channels * mult, use_downsample=True))

    self._middle_block = MiddleBlock(
        channels=model_channels * channel_mult[-1],
        context_channels=1280,
        num_heads=num_heads,
        size_per_head=40*channel_mult[-1],
        dropout_rate=dropout_rate,
    )

    self._output_blocks = []
    for i, mult in list(enumerate(channel_mult))[::-1]:
      for j in range(num_res_blocks + 1):
        self._output_blocks.append(
          OutputBlock(
            channels=model_channels * mult,
            num_heads=num_heads,
            size_per_head=40 * mult,
            hidden_size=1280,
            dropout_rate=dropout_rate,
            use_spatial_transformer=i < len(channel_mult) - 1,
            use_upsample=i > 0 and j == num_res_blocks,
          )
        )

    self._groupnorm = tf.keras.layers.GroupNormalization(groups=32, epsilon=1e-5)
    self._conv_out = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1, padding="SAME")
 
  def call(self, inputs, time, context=None, y=None, training=False):
    """
    Args:
      inputs (Tensor): [batch_size, height, width, channels]
      time (Tensor): [batch_size]
      context (Tensor): [batch_size, seq_len, context_channels]
    """
    #print("inputs", inputs.numpy().sum(), inputs.shape, "time", time.numpy().sum(), time.shape, "context", context.numpy().sum(), context.shape)
    outputs = self._conv_in(inputs) 

    time_emb = timestep_embedding(time, self._model_channels)
    time_embedding = self._time_dense2(self._time_dense1(time_emb))

    hiddens = [outputs]

    for block in self._input_blocks:
      outputs = block(outputs, time_embedding=time_embedding, context=context, training=training)
      hiddens.append(outputs)

    outputs = self._middle_block(outputs, time_embedding, context, training=training)

    for block in self._output_blocks:
      outputs = tf.concat([outputs, hiddens.pop()], axis=-1)
      outputs = block(outputs, time_embedding=time_embedding, context=context, training=training)

    #print("h", outputs.numpy().sum(), outputs.shape)
    outputs = self._conv_out(tf.nn.silu(self._groupnorm(outputs)))
    #print("hhh", outputs.numpy().sum(), outputs.shape)
    #input("hhh")
    return outputs


class InputBlock(tf.keras.layers.Layer):
  def __init__(self, channels, dropout_rate=0.1, use_spatial_transformer=False, use_downsample=False, num_heads=8, size_per_head=40, hidden_size=512,):
    super(InputBlock, self).__init__()
    self._channels = channels
    self._dropout_rate = dropout_rate
    self._use_spatial_transformer = use_spatial_transformer
    self._use_downsample = use_downsample 
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._hidden_size = hidden_size

    self._res_block = ResBlock(channels=channels, dropout_rate=dropout_rate)
    if use_spatial_transformer:
      self._spatial_transformer = SpatialTransformer(num_heads=num_heads, size_per_head=size_per_head, hidden_size=hidden_size, dropout_rate=dropout_rate)
    if use_downsample:
      self._downsample = Downsample(channels)

  def call(self, inputs, time_embedding=None, context=None, training=False):
    if self._use_downsample:
      outputs = self._downsample(inputs)
    else:
      outputs = self._res_block(inputs, time_embedding, training=training)
      if self._use_spatial_transformer:
        outputs = self._spatial_transformer(outputs, context, training=training)
    return outputs


class MiddleBlock(tf.keras.layers.Layer):
  def __init__(self, channels, context_channels, num_heads, size_per_head, dropout_rate=0.1):
    super(MiddleBlock, self).__init__()
    self._channels = channels
    self._context_channels = context_channels
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._dropout_rate = dropout_rate

    self._resblock1 = ResBlock(channels, dropout_rate=dropout_rate)
    self._spatial_transformer = SpatialTransformer(num_heads=num_heads, size_per_head=size_per_head, hidden_size=context_channels, dropout_rate=dropout_rate)
    self._resblock2 = ResBlock(channels, dropout_rate=dropout_rate)   

  def call(self, inputs, time_embedding, context, training=False):
    outputs = self._resblock2(
        self._spatial_transformer(
            self._resblock1(inputs, time_embedding, training=training),
            context,
            training=training,
        ),
        time_embedding,
        training=training,
    )
    return outputs


class OutputBlock(tf.keras.layers.Layer):
  def __init__(self, channels, dropout_rate=0.1, use_spatial_transformer=False, use_upsample=False, num_heads=8, size_per_head=40, hidden_size=512):
    super(OutputBlock, self).__init__()
    self._channels = channels
    self._dropout_rate = dropout_rate
    self._use_spatial_transformer = use_spatial_transformer
    self._use_upsample = use_upsample
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._hidden_size = hidden_size

    self._res_block = ResBlock(channels=channels, dropout_rate=dropout_rate)
    if self._use_spatial_transformer:
      self._spatial_transformer = SpatialTransformer(num_heads=num_heads, size_per_head=size_per_head, hidden_size=hidden_size, dropout_rate=dropout_rate)
    if self._use_upsample:
      self._upsample = Upsample(channels)

  def call(self, inputs, time_embedding, context=None, training=False):
    outputs = self._res_block(inputs, time_embedding, training=training)
    if self._use_spatial_transformer:
      outputs = self._spatial_transformer(outputs, context, training=training)
    if self._use_upsample:
      outputs = self._upsample(outputs)
    return outputs

 
class CrossAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads=8, size_per_head=64, dropout_rate=0., hidden_size=None):
    super(CrossAttention, self).__init__()

    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._dropout_rate = dropout_rate   
    self._hidden_size = (num_heads * size_per_head if hidden_size is None
        else hidden_size
    )

    self._dense_layer_query = Projection(
        num_heads, size_per_head, num_heads * size_per_head, mode='split')
    self._dense_layer_key = Projection(
        num_heads, size_per_head, hidden_size, mode='split')
    self._dense_layer_value = Projection(
        num_heads, size_per_head, hidden_size, mode='split')
    self._dense_layer_output = Projection(
        num_heads, size_per_head, num_heads * size_per_head, use_bias=True, mode='merge')
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, query, context=None, attention_mask=None, training=False):

    context = query if context is None else context

    # [batch_size, q_seq_len, num_heads, size_per_head]
    q = self._dense_layer_query(query)

    # [batch_size, c_seq_len, num_heads, size_per_head]
    k = self._dense_layer_key(context)
    v = self._dense_layer_value(context)

    # [batch_size, num_heads, q_seq_len, c_seq_len]
    attention_weights = tf.einsum('NQHS,NCHS->NHQC', q, k)
    attention_weights *= self._size_per_head ** -0.5
    if attention_mask is not None:
      attention_weights += attention_mask * NEG_INF
    attention_weights = tf.nn.softmax(attention_weights, axis=3)
    #attention_weights = self._dropout_layer(
    #    attention_weights, training=training)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    outputs = tf.einsum('NHQC,NCHS->NQHS', attention_weights, v)

    # [batch_size, q_seq_len, hidden_size]
    outputs = self._dense_layer_output(outputs)
    outputs = self._dropout_layer(outputs, training=training)
    return outputs    


class BasicTransformerBlock(tf.keras.layers.Layer):
  def __init__(self, num_heads=8, size_per_head=64, dropout_rate=0.1, hidden_size=512):
    super(BasicTransformerBlock, self).__init__()

    self._attention_layer1 = CrossAttention(num_heads=num_heads, size_per_head=size_per_head, dropout_rate=dropout_rate)

    self._attention_layer2 = CrossAttention(num_heads=num_heads, size_per_head=size_per_head, dropout_rate=dropout_rate, hidden_size=hidden_size) 

    self._ffn_layer = FeedForward(num_heads * size_per_head, )

    self._layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-05)
    self._layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-05)
    self._layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-05)

  def call(self, inputs, context=None, training=False):
    outputs = self._attention_layer1(self._layernorm1(inputs), training=training) + inputs 
    outputs = self._attention_layer2(self._layernorm2(outputs), context, training=training) + outputs
    outputs = self._ffn_layer(self._layernorm3(outputs)) + outputs

    return outputs


class GEGLU(tf.keras.layers.Layer):
  def __init__(self, channels):
    super(GEGLU, self).__init__()
    self._dense_layer = tf.keras.layers.Dense(channels * 2)

  def call(self, inputs):
    outputs, gate = tf.split(self._dense_layer(inputs), 2, axis=-1)
    outputs = outputs * tf.nn.gelu(gate)
    return outputs


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, channels, multiplier=4, dropout_rate=0.):
    super(FeedForward, self).__init__()
    self._geglu_layer = GEGLU(channels * multiplier)
    self._dense_layer = tf.keras.layers.Dense(channels)
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs, training=False):
    outputs = self._dense_layer(self._dropout_layer(self._geglu_layer(inputs), training=training))
    return outputs


class SpatialTransformer(tf.keras.layers.Layer):
  def __init__(self, num_heads=8, size_per_head=40, hidden_size=512, dropout_rate=0.1):
    super(SpatialTransformer, self).__init__()

    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._hidden_size = hidden_size
    self._dropout_rate = dropout_rate

    self._dense1 = tf.keras.layers.Dense(num_heads * size_per_head)

    self._block = BasicTransformerBlock(
        num_heads=num_heads, size_per_head=size_per_head, hidden_size=hidden_size, dropout_rate=dropout_rate
    )

    self._dense2 = tf.keras.layers.Dense(num_heads * size_per_head)
    self._groupnorm = tf.keras.layers.GroupNormalization(groups=32, epsilon=1e-6)

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


class ResBlock(tf.keras.layers.Layer):
  def __init__(self, channels, dropout_rate):
    super(ResBlock, self).__init__()
    self._channels = channels
    self._dropout_rate = dropout_rate

    self._group_norm_1 = tf.keras.layers.GroupNormalization(groups=32, epsilon=1e-5)
    self._conv2d_1 = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding="SAME")
    
    self._dense = tf.keras.layers.Dense(channels)

    self._group_norm_2 = tf.keras.layers.GroupNormalization(groups=32, epsilon=1e-5)
    self._conv2d_2 = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding="SAME")

    self._shortcut = tf.keras.layers.Dense(channels)
    self._dropout = tf.keras.layers.Dropout(dropout_rate)


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


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = tf.exp(
            -tf.math.log(tf.cast(max_period, "float32")) * tf.range(start=0, limit=half, dtype="float32") / half
        )
        args = tf.cast(timesteps[:, None], "float32") * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding



if __name__ == "__main__":
  import numpy as np
  from transformer import TransformerModel

  #"""
  vocab_size = 30522
  transformer = TransformerModel(vocab_size,
               encoder_stack_size=32,
               hidden_size=1280,
               num_heads=8,
               filter_size=1280*4,
               dropout_rate=0.1,)

  token_ids = np.asarray([[  101,  1037,  7865,  6071,  2003,  2652,  2858,  1010,  3514,  2006,
                           10683,   102,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0]])
  ti = np.asarray([[101, 102] + [0] * 75]) 

  token_ids = tf.constant(np.vstack([np.tile(ti, [4, 1]), np.tile(token_ids, [4, 1])]))

  transformer(token_ids, None)

  sd = np.load("/home/chaoji/work/genmo/diffusion/latent-diffusion/sd.npy", allow_pickle=True).item()
  weights = []


  for i in range(32):
    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".1.to_q.weight"]  
    w = w.T.reshape(-1, 8, 64)
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".1.to_k.weight"]
    w = w.T.reshape(-1, 8, 64)
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".1.to_v.weight"]
    w = w.T.reshape(-1, 8, 64)
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".1.to_out.weight"]
    w = w.T.reshape(8, 64, -1)
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".1.to_out.bias"]
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".0.weight"]
    weights.append(w)
    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".0.bias"]
    weights.append(w)
  

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".1.net.0.0.weight"]
    weights.append(w.T)
    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".1.net.0.0.bias"] 
    weights.append(w)
    
    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".1.net.2.weight"]
    weights.append(w.T)
    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".1.net.2.bias"]
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".0.weight"]
    weights.append(w)
    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".0.bias"]
    weights.append(w)
   
  weights.append(sd["cond_stage_model.transformer.norm.weight"])
  weights.append(sd["cond_stage_model.transformer.norm.bias"]) 
  weights.append(sd["cond_stage_model.transformer.token_emb.weight"])
  weights.append(sd["cond_stage_model.transformer.pos_emb.emb.weight"])

  transformer.set_weights(weights)
  print("\n" * 10) 
  context = transformer(token_ids, None)
  #"""












  #'''  
  unet = UNet()
  x = np.load("/home/chaoji/work/genmo/diffusion/latent-diffusion/x.npy").transpose(0, 2, 3, 1)
  x = np.concatenate([x, x], axis=0)
  #t_emb = timestep_embedding(tf.constant([981] * 8), 320)
  t_emb = tf.constant([981] * 8)


  shape_dict = {1: 1, 2: 1, 4: 2, 5: 2, 7: 4, 8: 4}
  shape_dict1 = {3: 4, 4: 4, 5: 4, 6: 2, 7: 2, 8: 2, 9: 1, 10: 1, 11: 1}

  weights = []
  weights.append(sd["model.diffusion_model.input_blocks.0.0.weight"].transpose(2, 3, 1, 0))
  weights.append(sd["model.diffusion_model.input_blocks.0.0.bias"])
  weights.append(sd["model.diffusion_model.time_embed.0.weight"].T)
  weights.append(sd["model.diffusion_model.time_embed.0.bias"])
  weights.append(sd["model.diffusion_model.time_embed.2.weight"].T)
  weights.append(sd["model.diffusion_model.time_embed.2.bias"])

  weights1 = []
  for i in range(1, 12):
    if i in (3, 6, 9):
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.op.weight"].transpose(2, 3, 1, 0))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.op.bias"])
      continue

    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.in_layers.0.weight"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.in_layers.0.bias"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.in_layers.2.weight"].transpose(2, 3, 1, 0)) 
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.in_layers.2.bias"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.emb_layers.1.weight"].T)
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.emb_layers.1.bias"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.out_layers.0.weight"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.out_layers.0.bias"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.out_layers.3.weight"].transpose(2, 3, 1, 0))
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.out_layers.3.bias"])

    if i in (4, 7):
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.skip_connection.weight"].squeeze().T)
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.skip_connection.bias"])

    if i in (1, 2, 4, 5, 7, 8):

      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.proj_in.weight"].squeeze().T)
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.proj_in.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn1.to_q.weight"].T.reshape(320 * shape_dict[i], 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn1.to_k.weight"].T.reshape(320 * shape_dict[i], 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn1.to_v.weight"].T.reshape(320 * shape_dict[i], 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn1.to_out.0.weight"].T.reshape(8, 40 * shape_dict[i], 320 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn1.to_out.0.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn2.to_q.weight"].T.reshape(320 * shape_dict[i], 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn2.to_k.weight"].T.reshape(1280, 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn2.to_v.weight"].T.reshape(1280, 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn2.to_out.0.weight"].T.reshape(8, 40 * shape_dict[i], 320 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn2.to_out.0.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.ff.net.0.proj.weight"].T)
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.ff.net.0.proj.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.ff.net.2.weight"].T)
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.ff.net.2.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm1.weight"]) 
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm1.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm2.weight"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm2.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm3.weight"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm3.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.proj_out.weight"].squeeze().T)
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.proj_out.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.norm.weight"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.norm.bias"])


  weights2 = []
  for i in range(3):
    if i in (0, 2):
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.in_layers.0.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.in_layers.0.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.in_layers.2.weight"].transpose(2, 3, 1, 0))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.in_layers.2.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.emb_layers.1.weight"].T)
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.emb_layers.1.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.out_layers.0.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.out_layers.0.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.out_layers.3.weight"].transpose(2, 3, 1, 0))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.out_layers.3.bias"])

    else:
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.proj_in.weight"].squeeze().T)
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.proj_in.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_q.weight"].T.reshape(320 * 4, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_k.weight"].T.reshape(320 * 4, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_v.weight"].T.reshape(320 * 4, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_out.0.weight"].T.reshape(8, 40 * 4, 320 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_out.0.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_q.weight"].T.reshape(320 * 4, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_k.weight"].T.reshape(1280, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_v.weight"].T.reshape(1280, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_out.0.weight"].T.reshape(8, 40 * 4, 320 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_out.0.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.ff.net.0.proj.weight"].T)
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.ff.net.0.proj.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.ff.net.2.weight"].T)
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.ff.net.2.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm1.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm1.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm2.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm2.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm3.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm3.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.proj_out.weight"].squeeze().T)
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.proj_out.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.norm.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.norm.bias"])


  weights3 = []
  for i in range(12):
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.in_layers.0.weight"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.in_layers.0.bias"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.in_layers.2.weight"].transpose(2, 3, 1, 0))
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.in_layers.2.bias"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.emb_layers.1.weight"].T)
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.emb_layers.1.bias"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.out_layers.0.weight"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.out_layers.0.bias"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.out_layers.3.weight"].transpose(2, 3, 1, 0))
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.out_layers.3.bias"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.skip_connection.weight"].squeeze().T)
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.skip_connection.bias"])

    if i in (3, 4, 5, 6, 7, 8, 9, 10, 11):
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.proj_in.weight"].squeeze().T)
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.proj_in.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn1.to_q.weight"].T.reshape(320 * shape_dict1[i], 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn1.to_k.weight"].T.reshape(320 * shape_dict1[i], 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn1.to_v.weight"].T.reshape(320 * shape_dict1[i], 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn1.to_out.0.weight"].T.reshape(8, 40 * shape_dict1[i], 320 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn1.to_out.0.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn2.to_q.weight"].T.reshape(320 * shape_dict1[i], 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn2.to_k.weight"].T.reshape(1280, 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn2.to_v.weight"].T.reshape(1280, 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn2.to_out.0.weight"].T.reshape(8, 40 * shape_dict1[i], 320 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn2.to_out.0.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.ff.net.0.proj.weight"].T)
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.ff.net.0.proj.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.ff.net.2.weight"].T)
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.ff.net.2.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm1.weight"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm1.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm2.weight"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm2.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm3.weight"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm3.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.proj_out.weight"].squeeze().T)
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.proj_out.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.norm.weight"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.norm.bias"])

    if i in (2, 5, 8):
      if i == 2:
        j = 1
      else:
        j = 2
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.{j}.conv.weight"].transpose(2, 3, 1, 0))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.{j}.conv.bias"])
      
  weights = weights + weights1 + weights2 + weights3
  weights.append(sd[f"model.diffusion_model.out.0.weight"])
  weights.append(sd[f"model.diffusion_model.out.0.bias"])
  weights.append(sd[f"model.diffusion_model.out.2.weight"].transpose(2, 3, 1, 0))
  weights.append(sd[f"model.diffusion_model.out.2.bias"])

  unet(x, t_emb, context)
  unet.set_weights(weights) 
  print("\n" * 5)
  outputs = unet(x, t_emb, context)

  #'''

