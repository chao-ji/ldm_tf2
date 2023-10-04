
import numpy as np
import tensorflow as tf

from discriminator import NLayerDiscriminator

from distribution import DiagonalGaussian

from autoencoder import AutoencoderKL

from loss import LPIPSWithDiscriminator

from dataset import celebahq_dataset



disc_loss = LPIPSWithDiscriminator(
      disc_start=50001,
      kl_weight=0.000001,
      disc_weight=0.5,)

autoencoder = AutoencoderKL()

base_lr = 4.5e-06

ae_adam = tf.keras.optimizers.Adam(base_lr, epsilon=1e-8, beta_1=0.5, beta_2=0.9)
disc_adam = tf.keras.optimizers.Adam(base_lr, epsilon=1e-8, beta_1=0.5, beta_2=0.9)


@tf.function
def func(inputs):
  #with tf.GradientTape() as tape:
  recon, posterior = autoencoder(inputs, sample_posterior=False)

  last_layer = autoencoder.get_last_layer()

  ae_loss = disc_loss(inputs, recon, posterior, optimizer_idx=0, last_layer=last_layer, global_step=ae_adam.iterations)
  d_loss = disc_loss(inputs, recon, posterior, optimizer_idx=1, last_layer=last_layer, global_step=disc_adam.iterations)

  gradients = tf.gradients(ae_loss, autoencoder.trainable_weights)
  ae_adam.apply_gradients(
          zip(gradients, autoencoder.trainable_variables))

  gradients = tf.gradients(d_loss, disc_loss.discriminator.trainable_weights) 
  disc_adam.apply_gradients(
          zip(gradients, disc_loss.discriminator.trainable_weights))
  return ae_loss, d_loss






