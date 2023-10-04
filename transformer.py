import tensorflow as tf


class Projection(tf.keras.layers.Layer):
  def __init__(self,
               num_heads,
               size_per_head,
               hidden_size=None,
               use_bias=False,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               mode="split"):

    super(Projection, self).__init__()
    if mode not in ("split", "merge"):
      raise ValueError('"mode" must be either "split" or "merge".')
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._hidden_size = (num_heads * size_per_head if hidden_size is None
        else hidden_size
    )
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._mode = mode


  def build(self, inputs_shape):
    depth = inputs_shape[-1]
    if depth is None:
      raise ValueError("The depth of inputs must not be None.")

    if self._mode == "merge":
      kernel_shape = self._num_heads, self._size_per_head, self._hidden_size
      if self._use_bias:
        bias_shape = self._hidden_size
    else:
      kernel_shape = self._hidden_size, self._num_heads, self._size_per_head
      if self._use_bias:
        bias_shape = self._size_per_head

    self.add_weight(name="kernel",
                    shape=kernel_shape,
                    initializer=self._kernel_initializer,
                    dtype="float32",
                    trainable=True)
    if self._use_bias:
      self.add_weight(name="bias",
                      shape=bias_shape,
                      initializer=self._bias_initializer,
                      dtype="float32",
                      trainable=True)
    super(Projection, self).build(inputs_shape)

  def call(self, inputs):
    kernel = self.trainable_variables[0]
    if self._mode == "merge":
      outputs = tf.einsum("NTHS,HSD->NTD", inputs, kernel)
    else:
      outputs = tf.einsum("NTD,DHS->NTHS", inputs, kernel)

    if self._use_bias:
      outputs += self.trainable_variables[1]

    return outputs


class Attention(tf.keras.layers.Layer):

  def __init__(self, num_heads, size_per_head, dropout_rate, hidden_size=None):
    super(Attention, self).__init__()
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate
    self._size_per_head = size_per_head
    self._hidden_size = (num_heads * size_per_head if hidden_size is None
        else hidden_size
    )

    self._dense_layer_query = Projection(
        num_heads, self._size_per_head, hidden_size, mode='split')
    self._dense_layer_key = Projection(
        num_heads, self._size_per_head, hidden_size, mode='split')
    self._dense_layer_value = Projection(
        num_heads, self._size_per_head, hidden_size, mode='split')
    self._dense_layer_output = Projection(
        num_heads, self._size_per_head, hidden_size, use_bias=True, mode='merge')
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, query, context, attention_mask=None, training=False):

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
    attention_weights = self._dropout_layer(
        attention_weights, training=training)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    outputs = tf.einsum('NHQC,NCHS->NQHS', attention_weights, v)

    # [batch_size, q_seq_len, hidden_size]
    outputs = self._dense_layer_output(outputs)
    return outputs


class FeedForwardNetwork(tf.keras.layers.Layer):
  def __init__(self,
               hidden_size,
               filter_size,
               dropout_rate,
               filter_activation=tf.nn.relu):

    super(FeedForwardNetwork, self).__init__()
    self._hidden_size = hidden_size
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate
    self._filter_activation = filter_activation

    self._dense_layer_filter = tf.keras.layers.Dense(
        filter_size, use_bias=True, activation=filter_activation)
    self._dense_layer_output = tf.keras.layers.Dense(hidden_size, use_bias=True)
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs, training):
    outputs = self._dense_layer_filter(inputs)
    outputs = self._dropout_layer(outputs, training=training)
    outputs = self._dense_layer_output(outputs)
    return outputs



class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, hidden_size, num_heads, size_per_head, filter_size, dropout_rate):
    super(EncoderLayer, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate

    self._mha = Attention(num_heads, size_per_head, dropout_rate, hidden_size=hidden_size)
    self._layernorm_mha = tf.keras.layers.LayerNormalization(epsilon=1e-05)
    self._dropout_mha = tf.keras.layers.Dropout(dropout_rate)

    self._ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate, filter_activation=tf.nn.gelu)
    self._layernorm_ffn = tf.keras.layers.LayerNormalization(epsilon=1e-05)
    self._dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs, padding_mask, training=False):
    #print("inputs", inputs.numpy().mean())
    query = reference = self._layernorm_mha(inputs)
    #print("ln", query.numpy().mean())
    outputs = self._mha(query, reference, padding_mask, training)
    #print("attn", outputs.numpy().mean())
    ffn_inputs = self._dropout_mha(outputs, training=training) + inputs
    #print("ffn_inputs", ffn_inputs.numpy().mean())

    outputs = self._layernorm_ffn(ffn_inputs)
    #print("outputs", outputs.numpy().mean())
    outputs = self._ffn(outputs, training)
    #print("ffn", outputs.numpy().mean())

    outputs = self._dropout_ffn(outputs, training=training) + ffn_inputs
    return outputs


class Encoder(tf.keras.layers.Layer):
  def __init__(
      self, stack_size, hidden_size, num_heads, size_per_head, filter_size, dropout_rate):

    super(Encoder, self).__init__()
    self._stack_size = stack_size
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate

    self._stack = [EncoderLayer(hidden_size,
                                num_heads,
                                size_per_head,
                                filter_size,
                                dropout_rate) for _ in range(self._stack_size)]
    self._layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-05)

  def call(self, inputs, padding_mask, training):
    #print("a", inputs.numpy().mean())
    for layer in self._stack:
      inputs = layer.call(inputs, padding_mask, training)
    #print("b", inputs.numpy().mean())
    outputs = self._layernorm(inputs)
    #print("c", outputs.numpy().mean())
    return outputs


class TransformerModel(tf.keras.layers.Layer):
  def __init__(self,
               vocab_size,
               encoder_stack_size=6,
               hidden_size=512,
               num_heads=8,
               size_per_head=64,
               max_seq_len=77,
               filter_size=2048,
               dropout_rate=0.1,
    ):

    super(TransformerModel, self).__init__()
    self._vocab_size = vocab_size
    self._encoder_stack_size = encoder_stack_size
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._max_seq_len = max_seq_len
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate

    self._encoder = Encoder(
        encoder_stack_size, hidden_size, num_heads, size_per_head, filter_size, dropout_rate)

    self._embedding_layer = tf.keras.layers.Embedding(vocab_size, hidden_size)
    self._positional_embedding_layer = tf.keras.layers.Embedding(max_seq_len, hidden_size)
    self._logits_layer = tf.keras.layers.Dense(vocab_size)

    self._encoder_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, token_ids, padding_mask=None, training=False):
    encoder_outputs = self._encode(token_ids, padding_mask=None, training=training)
    return encoder_outputs

  def _encode(self, token_ids, padding_mask=None, training=False):
    seq_len = tf.shape(token_ids)[1]

    # [batch_size, seq_len, hidden_size]
    token_embeddings = self._embedding_layer(token_ids)
    #return token_embeddings 

    # [src_seq_len, hidden_size]
    positional_encoding = self._positional_embedding_layer(tf.range(seq_len)[tf.newaxis])
    token_embeddings += positional_encoding

    token_embeddings = self._encoder_dropout_layer(
        token_embeddings, training)

    encoder_outputs = self._encoder(
        token_embeddings, padding_mask, training)
    return encoder_outputs

