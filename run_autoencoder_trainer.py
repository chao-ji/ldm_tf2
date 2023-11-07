"""Train autoencoders."""
import glob
import yaml
import os

import tensorflow as tf
from absl import app
from absl import flags

from autoencoder import AutoencoderKL, AutoencoderVQ 
from discriminator import Discriminator
from dataset import create_dataset
from lpips import LPIPS
from model_runners import AutoencoderTrainerKL, AutoencoderTrainerVQ


flags.DEFINE_string("config_path", None, "Path to yaml config file.")

FLAGS = flags.FLAGS


def main(_):
  with open(FLAGS.config_path) as f:
    config = yaml.safe_load(f)

  # create `lpips` (to compute percetual loss) and load weights from pretrained
  lpips = LPIPS()
  tf.train.Checkpoint(lpips=lpips).restore(config["lpips_ckpt_path"])

  # create `autoencoder`, `discriminator` as submodules of autoencoder trainer
  if config["autoencoder_training"]["autoencoder_type"] == "kl":
    kwargs = config["autoencoder_kl_trainer"]
    autoencoder = AutoencoderKL(**config["autoencoder_kl"])
    discriminator = Discriminator(**config["ae_kl_discriminator"])
    trainer = AutoencoderTrainerKL(autoencoder, lpips, discriminator, **kwargs)
    print("[INFO] training KL-regularized autoencoder...")
  elif config["autoencoder_training"]["autoencoder_type"] == "vq":
    kwargs = config["autoencoder_vq_trainer"]
    autoencoder = AutoencoderVQ(**config["autoencoder_vq"])
    discriminator = Discriminator(**config["ae_vq_discriminator"])
    trainer = AutoencoderTrainerVQ(autoencoder, lpips, discriminator, **kwargs)
    print("[INFO] training VQ-regularized autoencoder...")
  else:
    raise NotImplementedError("invalid autoencoder type.")

  # initialize dataset
  filenames = glob.glob(
      os.path.join(config["autoencoder_training"]["root_path"], "*.tfrecord"))
  kwargs = config["autoencoder_training"]["params"]
  dataset = create_dataset(filenames, **kwargs)

  # create optimizers separately for autoencoder and discriminator
  autoencoder_optimizer = tf.keras.optimizers.Adam(
      **config["autoencoder_optimizer"])
  discriminator_optimizer = tf.keras.optimizers.Adam(
      **config["discriminator_optimizer"])

  # training!
  ckpt = tf.train.Checkpoint(
      autoencoder=autoencoder,
      ae_optimizer=autoencoder_optimizer,
      d_optimizer=discriminator_optimizer)
  ckpt_path = config["autoencoder_training"]["ckpt_path"]
  num_iterations = config["autoencoder_training"]["num_iterations"]
  print(f"[INFO] Start training for {num_iterations} iterations.")
  trainer.train(
      dataset,
      autoencoder_optimizer,
      discriminator_optimizer,
      ckpt,
      ckpt_path,
      num_iterations,
  )

if __name__ == "__main__":
  flags.mark_flag_as_required("config_path")
  app.run(main)
