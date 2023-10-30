import os
import tensorflow as tf

from autoencoder import AutoencoderKL, AutoencoderVQ 
from discriminator import NLayerDiscriminator
from lpips import LPIPS

from dataset import create_celebeahq256_dataset
from model_runners import AutoencoderTrainerKL, AutoencoderTrainerVQ


if __name__ == "__main__":

  learning_rate = 4.5e-6
  beta_1 = 0.5
  beta_2 = 0.9
  epsilon = 1e-8

  autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
  discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)




  #autoencoder = AutoencoderKL(z_channels=4)
  autoencoder = AutoencoderVQ(z_channels=4)

  lpips = LPIPS()
  disc_num_layers = 2 # 3
  discriminator = NLayerDiscriminator(num_layers=disc_num_layers)
  lpips_ckpt = tf.train.Checkpoint(lpips=lpips)
  lpips_ckpt.restore("lpips.ckpt-1")

  #ckpt = tf.train.Checkpoint(autoencoder=autoencoder, lpips=lpips, discriminator=discriminator)
  #ckpt.restore("kl-f8-1")
  #ckpt.restore("vq-f8-1")



  celeba_hq_path = "/home/chaoji/data/generative/celeba_hq/celeba_hq"
  paths = [
      os.path.join(celeba_hq_path, part) for part in [ "train/male", "train/female"]
  ]
  batch_size = 3
  dataset = create_celebeahq256_dataset(paths, 256, batch_size, epochs=128, flip=True)



  trainer = AutoencoderTrainerVQ(autoencoder, lpips, discriminator, global_step_discriminator=25001, discriminator_weight=0.6, codebook_weight=1.0,)
  #trainer = AutoencoderTrainerKL(autoencoder, lpips, discriminator, global_step_discriminator=50001, discriminator_weight=0.5, kl_weight=1e-6)
  grads = trainer.train(dataset, autoencoder_optimizer, discriminator_optimizer)

