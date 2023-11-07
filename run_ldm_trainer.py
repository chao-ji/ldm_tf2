import yaml
import os
import glob

import tensorflow as tf
from absl import app
from absl import flags

from dataset import create_dataset
from model_runners import LatentDiffusionModelTrainer
from autoencoder import AutoencoderKL, AutoencoderVQ
from unet import UNet
from transformer import TransformerModel


flags.DEFINE_string("config_path", None, "Path to yaml config file.")

FLAGS = flags.FLAGS


def main(_):
  with open(FLAGS.config_path) as f:
    config = yaml.safe_load(f)

  with tf.device("/cpu:0"):

    kwargs = config["latent_diffusion_optimizer"]
    optimizer = tf.keras.optimizers.AdamW(**kwargs)


    # initialize dataset
    filenames = glob.glob(
        os.path.join(config["ldm_training"]["root_path"], "*.tfrecord"))
    kwargs = config["ldm_training"]["params"]
    dataset = create_dataset(filenames, **kwargs)

    # create unet, transformer, autoencoder
    # load pretrained weights for transformer and autoencoder
    kwargs = config["unet"]
    unet = UNet(**kwargs)

    kwargs = config["cond_stage_model"]
    transformer = TransformerModel(**kwargs)
    tf.train.Checkpoint(transformer=transformer).restore(
        config["pre_ckpt_paths"]["cond_stage_model"]).expect_partial()

    if config["ldm_training"]["autoencoder_type"] == "kl":
      autoencoder = AutoencoderKL(**config["autoencoder_kl"])
    elif config["ldm_training"]["autoencoder_type"] == "vq":
      autoencoder = AutoencoderVQ(**config["autoencoder_vq"])
    else:
      raise NotImplementedError("invalid autoencoder type.")
    tf.train.Checkpoint(autoencoder=autoencoder).restore(
        config["pre_ckpt_paths"]["autoencoder"]).expect_partial()

    kwargs = config["ldm"]
    trainer = LatentDiffusionModelTrainer(
        unet=unet,
        autoencoder=autoencoder,
        cond_stage_model=transformer,
        **kwargs,
    )
    ckpt = tf.train.Checkpoint(model=unet, transformer=transformer, optimizer=optimizer)
    ckpt_path = config["ldm_training"]["ckpt_path"]
   
    trainer.train(
        dataset=dataset,
        optimizer=optimizer,
        ckpt=ckpt,
        ckpt_path=ckpt_path,
        train_cond_model=config["ldm_training"]["train_cond_model"],
        num_iterations=config["ldm_training"]["num_iterations"]
    )

if __name__ == "__main__":
  flags.mark_flag_as_required("config_path")
  app.run(main)
