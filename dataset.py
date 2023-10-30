"""Create dataset for training DDPM."""
import os
import tensorflow as tf

import glob

flip = True
batch_size = 4

SHUFFLE_SIZE = 8192

def create_celebeahq256_dataset(dir_paths, image_size, batch_size, epochs=16, flip=False):
  """Create dataset for celeba hq dataset or LSUN dataset."""
  filenames = []
  for dir_path in dir_paths:
    filenames.extend(glob.glob(os.path.join(dir_path, "*.jpg")))

  dataset = tf.data.Dataset.from_tensor_slices(filenames, name="celebahq256").repeat(epochs)
  dataset = dataset.map(lambda x: tf.io.read_file(x))
  dataset = dataset.map(lambda x: tf.io.decode_jpeg(x, channels=3))
  dataset = dataset.map(lambda x: tf.image.resize(x, (image_size, image_size)))

  if flip:
    dataset = dataset.map(tf.image.random_flip_left_right)
  dataset = dataset.map(lambda inputs: tf.cast(inputs, "float32") / 127.5 - 1.)
  dataset = dataset.shuffle(SHUFFLE_SIZE)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  return dataset


def create_cifar10_dataset(batch_size, epochs=16, flip=False):
  """Create CIFAR10 dataset."""
  (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
  dataset = tf.data.Dataset.from_tensor_slices(x_train).repeat(epochs)

  if flip:
    dataset = dataset.map(tf.image.random_flip_left_right)
  dataset = dataset.map(lambda inputs: tf.cast(inputs, "float32") / 127.5 - 1.)
  dataset = dataset.shuffle(x_train.shape[0])
  dataset = dataset.batch(batch_size, drop_remainder=True)

  return dataset
