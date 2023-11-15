"""Create dataset for training DDPM."""
import os
from transformers import BertTokenizerFast

import tensorflow as tf

import json
import glob


BUFFER_SIZE = 1024


def _raw_data_to_example(image_filepath, caption=None):
  with open(image_filepath, "rb") as f:
    image_string = f.read()

  image = tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[image_string]))
  features = {"image": image}
  if caption is not None:
    caption = tf.train.Feature(int64_list=tf.train.Int64List(value=caption))
    features["caption"] = caption
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return example


def convert_images_to_tfrecord(filenames, out_path, num_shards=100):
  """"""
  writers = [tf.io.TFRecordWriter(
      os.path.join(out_path, f"images_{i:02d}-{num_shards:02d}.tfrecord"))
        for i in range(num_shards)]

  shard = 0
  for file_path in filenames:
    example = _raw_data_to_example(file_path)
    writers[shard].write(example.SerializeToString())
    shard = (shard + 1) % num_shards
  for writer in writers:
    writer.close()


def convert_coco_captions_to_tfrecord(
    root_path,
    part,
    ann_filename,
    tokenizer,
    out_path,
    max_length=77,
    num_shards=20,
  ):
  """Convert coco captions dataset to tfrecord format."""
  with open(os.path.join(root_path, "annotations", ann_filename)) as f:
    raw_data = json.load(f)
  image_dict = {image["id"]: image for image in raw_data["images"]}

  examples = [
      (
        os.path.join(root_path, part, image_dict[ann["image_id"]]["file_name"]),
        tokenizer(
          ann["caption"],
          truncation=True,
          max_length=max_length,
          return_length=True,
          return_overflowing_tokens=False,
          padding="max_length",
          return_tensors="pt")["input_ids"][0].numpy().tolist(),
      )
      for ann in raw_data["annotations"]
  ]

  writers = [tf.io.TFRecordWriter(
      os.path.join(out_path, f"coco_caption_{i:02d}-{num_shards:02d}.tfrecord"))
        for i in range(num_shards)]

  shard = 0
  for file_path, caption in examples:
    example = _raw_data_to_example(file_path, caption)
    writers[shard].write(example.SerializeToString())
    shard = (shard + 1) % num_shards
  for writer in writers:
    writer.close()


def create_dataset(
    filenames,
    batch_size=1,
    image_size=256,
    keys=("image", "caption"),
    flip=False,
    max_seq_len=77,
    random_seed=None):
  """Create captions dataset (image, caption) pair from previously created
  tfrecord files.
  """
  dataset = tf.data.Dataset.from_tensor_slices(filenames).shuffle(
      len(filenames), seed=random_seed).repeat()
  dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(
      filename).shuffle(BUFFER_SIZE))

  def parse_fn(serialized_example):
    parse_dict = {"image": tf.io.FixedLenFeature([], "string")}
    if "caption" in keys:
      parse_dict["caption"] = tf.io.VarLenFeature("int64")

    parsed = tf.io.parse_single_example(serialized_example, parse_dict)
    parsed["image"] = tf.io.decode_jpeg(parsed["image"], channels=3)
    if "caption" in keys and "caption" in parsed:
      parsed["caption"] = tf.sparse.to_dense(parsed["caption"])
      parsed["caption"].set_shape([max_seq_len])
    return parsed

  dataset = dataset.map(
    parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def process_image(image):
    """Pad image to square and resize to image_size x image_size."""
    if flip:
      image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, "float32") / 127.5 - 1.
    height, width, _ = tf.unstack(tf.shape(image))
    if height > width:
      pad_size = height - width
      pad_low = pad_size // 2
      pad_high = pad_size - pad_low
      paddings = [[0, 0], [pad_low, pad_high], [0, 0]]
      image = tf.pad(image, paddings)
    elif width > height:
      pad_size = width - height
      pad_low = pad_size // 2
      pad_high = pad_size - pad_low
      paddings = [[pad_low, pad_high], [0, 0], [0, 0]]
      image = tf.pad(image, paddings)
    image = tf.image.resize(image, (image_size, image_size),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image

  def proj_fn(data_dict):
    if "caption" in data_dict:
      return process_image(data_dict["image"]), data_dict["caption"]
    else:
      return process_image(data_dict["image"])

  dataset = dataset.map(proj_fn)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  return dataset
