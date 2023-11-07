import os
import glob

from transformers import BertTokenizerFast
from dataset import convert_coco_captions_to_tfrecord, convert_images_to_tfrecord


if __name__ == "__main__":

  # 1. Convert images dataset to tfrecord files (for training autoencoder)
  root_path = "/path/to/coco_root"
  part = "train2017"
  # `out_path`: path to output directory where tfrecord files will be located
  out_path = "/path/to/tfrecord/images"

  # `filenames`: a list of image file paths
  filenames = glob.glob(os.path.join(root_path, part, "*.jpg"))

  convert_images_to_tfrecord(
      filenames,
      out_path,
      num_shards=100,
  )

  # 2. Convert coco captions dataset to tfrecord files (for training ldm)
  root_path = "/path/to/coco_root"
  part = "val2017"
  ann_filename = "captions_val2017.json"
  tokenizer = BertTokenizerFast.from_pretrained("bert_model/")
  # `out_path`: path to output directory where tfrecord files will be located
  out_path = "/path/to/tfrecord/images_captions"
  max_length = 77
  num_shards = 20

  convert_coco_captions_to_tfrecord(
      root_path,
      part,
      ann_filename,
      tokenizer,
      out_path,
      max_length=max_length,
      num_shards=num_shards,
  )
