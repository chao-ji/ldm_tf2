"""LPIPS (Learned Perceptual Image Patch Similarity) for computing perceptual
 loss.

Ref: https://arxiv.org/abs/1801.03924 
"""
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense 


SHIFT = tf.constant([-.030, -.088, -.188], dtype="float32", shape=[1, 1, 1, 3])
SCALE = tf.constant([.458, .448, .450], dtype="float32", shape=[1, 1, 1, 3])

def _normalize_tensor(x, eps=1e-10):
    norm_factor = tf.sqrt(tf.reduce_sum(x ** 2, axis=-1, keepdims=True))
    return x / (norm_factor + eps)


class VGG16(tf.keras.layers.Layer):
  """Compute feature maps from five different layers of VGG16."""
  def __init__(self):
    super(VGG16, self).__init__()

    self._conv_layers = [
        [Conv2D(64,  3, 1, "VALID") for _ in range(2)],
        [Conv2D(128, 3, 1, "VALID") for _ in range(2)],
        [Conv2D(256, 3, 1, "VALID") for _ in range(3)],
        [Conv2D(512, 3, 1, "VALID") for _ in range(3)],
        [Conv2D(512, 3, 1, "VALID") for _ in range(3)],
    ]
    self._conv_indices = [[0, 1]] * 2 + [[0, 1, 2]] * 3 

  def call(self, inputs): 
    outputs = inputs
    features = []

    for i in range(5):
      if i > 0:
        outputs = tf.nn.max_pool2d(outputs, ksize=2, strides=2, padding="VALID")
      for j in self._conv_indices[i]:
        outputs = tf.pad(outputs, [[0, 0], [1, 1], [1, 1], [0, 0]]) 
        outputs = self._conv_layers[i][j](outputs)
        outputs = tf.nn.relu(outputs)
      features.append(outputs)
    return features 


class LPIPS(tf.keras.layers.Layer):
  def __init__(self):
    # freeze weights in `LPIPS`, set trainable=False
    super(LPIPS, self).__init__(trainable=False)
    self._vgg16 = VGG16()
    self._projs = [Dense(1, use_bias=False) for _ in range(5)]

  def call(self, images1, images2):
    images1 = (images1 - SHIFT) / SCALE
    images2 = (images2 - SHIFT) / SCALE

    images1_features = self._vgg16(images1)
    images2_features = self._vgg16(images2) 

    outputs = []
    for i in range(5):
      diff = tf.math.squared_difference(
          _normalize_tensor(images1_features[i]),
          _normalize_tensor(images2_features[i]),
      )
      diff = tf.reduce_mean(self._projs[i](diff), axis=[1, 2], keepdims=True)
      outputs.append(diff)

    dissimilarity = tf.add_n(outputs)
    return dissimilarity
