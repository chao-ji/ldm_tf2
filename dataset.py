import tensorflow as tf

import glob

flip = True
batch_size = 2

dataset = tf.data.Dataset.from_tensor_slices(glob.glob("/home/chaoji/data/generative/celeba_hq/celeba_hq/train/male/*.jpg"))
dataset = dataset.map(lambda x: tf.io.read_file(x))
dataset = dataset.map(lambda x: tf.io.decode_jpeg(x, channels=3))
dataset = dataset.map(lambda x: tf.image.resize(x, (256, 256)))
if flip:
  dataset = dataset.map(tf.image.random_flip_left_right)

dataset = dataset.map(lambda inputs: tf.cast(inputs, "float32") / 127.5 - 1.)
celebahq_dataset = dataset.batch(batch_size, drop_remainder=True)



