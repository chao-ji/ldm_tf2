import os
import numpy as np
import tensorflow as tf

from dataset import create_celebeahq256_dataset
from loss import LPIPSWithDiscriminator, LPIPS
from discriminator import NLayerDiscriminator


def hinge_d_loss(logits_real, logits_fake):
  loss_real = tf.reduce_mean(tf.nn.relu(1. - logits_real))
  loss_fake = tf.reduce_mean(tf.nn.relu(1. + logits_fake))
  d_loss = 0.5 * (loss_real + loss_fake)
  return d_loss


def vanilla_d_loss(logits_real, logits_fake):
  d_loss = 0.5 * (
      tf.reduce_mean(tf.nn.softplus(-logits_real)) +
      tf.reduce_mean(tf.nn.softplus(logits_fake)))
  return d_loss


class _AutoencoderTrainer(object):
  def __init__(
      self,
      autoencoder,
      lpips,
      discriminator,
      global_step_discriminator,
      lpips_weight=1.,
      kl_weight=1.,
      discriminator_weight=1.,
      discriminator_factor=1.,
      discriminator_loss_type="hinge",
    ):
    self._autoencoder = autoencoder
    self._lpips = lpips
    self._discriminator = discriminator
    self._global_step_discriminator = global_step_discriminator

    self._lpips_weight = lpips_weight
    self._kl_weight = kl_weight
    self._discriminator_weight = discriminator_weight
    self._discriminator_factor = discriminator_factor
    self._discriminator_loss_type = discriminator_loss_type
    self._discriminator_loss = hinge_d_loss if discriminator_loss_type == "hinge" else vanilla_d_loss

  def _compute_adaptive_weight(self, nll_loss, generator_loss):
    last_layer = self._autoencoder.get_last_layer()

    nll_grads = tf.gradients(nll_loss, last_layer)[0]
    generator_grads = tf.gradients(generator_loss, last_layer)[0]

    weight = tf.norm(nll_grads) / (tf.norm(generator_grads) + 1e-4)
    weight = tf.clip_by_value(weight, 0.0, 1e4)
    weight = tf.stop_gradient(weight)
    weight = weight * self._discriminator_weight
    return weight

  def _compute_nll_loss(self, inputs, outputs, reduce_loss=False):
    reconstruction_loss = tf.abs(inputs - outputs)
    lpips_loss = self._lpips(inputs, outputs)
    nll_loss = reconstruction_loss + self._lpips_weight * lpips_loss
 
    if reduce_loss:
      nll_loss = tf.reduce_sum(nll_loss) / nll_loss.shape[0]
    else:
      nll_loss = tf.reduce_mean(nll_loss)

    return nll_loss

class AutoencoderTrainerKL(_AutoencoderTrainer):

  def __init__(
      self,
      autoencoder,
      lpips,
      discriminator,
      global_step_discriminator=50001,
      lpips_weight=1.,
      kl_weight=1.,
      discriminator_weight=1.,
      discriminator_factor=1.,
      discriminator_loss_type="hinge",
      
    ):
    super(AutoencoderTrainerKL, self).__init__(
      autoencoder=autoencoder,
      lpips=lpips,
      discriminator=discriminator,
      global_step_discriminator=global_step_discriminator,
      lpips_weight=lpips_weight,
      kl_weight=kl_weight,
      discriminator_weight=discriminator_weight,
      discriminator_factor=discriminator_factor,
      discriminator_loss_type=discriminator_loss_type,
    )

  def train(
      self,
      dataset,
      autoencoder_optimizer,
      discriminator_optimizer,
      #ckpt,
      #ckpt_path,
      #persist_per_iterations,
      log_per_iterations=100,
      logdir="log"
    ):

    train_step_signature = [
        tf.TensorSpec(
            shape=dataset.element_spec.shape,
            dtype=dataset.element_spec.dtype,
        ),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step_autoencoder(inputs):

      with tf.GradientTape() as tape:
        posterior = self._autoencoder.encode(inputs)
        latents = posterior.sample()
        outputs = self._autoencoder.decode(latents)
        nll_loss = self._compute_nll_loss(inputs, outputs, reduce_loss=True)
        kl_loss = posterior.kl()
        kl_loss = tf.reduce_sum(kl_loss) / kl_loss.shape[0]

        autoencoder_loss = nll_loss + self._kl_weight * kl_loss

      autoencoder_grads = tape.gradient(autoencoder_loss, self._autoencoder.trainable_weights)
      autoencoder_optimizer.apply_gradients(zip(autoencoder_grads, self._autoencoder.trainable_weights))
      return autoencoder_loss, autoencoder_optimizer.iterations, autoencoder_optimizer.learning_rate 


    @tf.function(input_signature=train_step_signature)
    def train_step_autoencoder_discriminator(inputs):

      with tf.GradientTape(persistent=True) as tape:
        posterior = self._autoencoder.encode(inputs)
        #latents = posterior.mode() 
        latents = posterior.sample()
        outputs = self._autoencoder.decode(latents)


        nll_loss = self._compute_nll_loss(inputs, outputs, reduce_loss=True)
        kl_loss = posterior.kl()
        kl_loss = tf.reduce_sum(kl_loss) / kl_loss.shape[0]

        autoencoder_loss = nll_loss + self._kl_weight * kl_loss


        logits_fake = self._discriminator(outputs)
        generator_loss = -tf.reduce_mean(logits_fake)

        adaptive_weight = self._compute_adaptive_weight(nll_loss, generator_loss)
        autoencoder_loss += adaptive_weight * self._discriminator_factor * generator_loss

        logits_real = self._discriminator(tf.stop_gradient(inputs))
        logits_fake = self._discriminator(tf.stop_gradient(outputs))

        discriminator_loss = self._discriminator_loss(logits_real, logits_fake)
        discriminator_loss *= self._discriminator_factor

      autoencoder_grads = tape.gradient(autoencoder_loss, self._autoencoder.trainable_weights) 
      autoencoder_optimizer.apply_gradients(zip(autoencoder_grads, self._autoencoder.trainable_weights))
      discriminator_grads = tape.gradient(discriminator_loss, self._discriminator.trainable_weights)
      discriminator_optimizer.apply_gradients(zip(discriminator_grads, self._discriminator.trainable_weights))

      return autoencoder_loss, discriminator_loss, autoencoder_optimizer.iterations, discriminator_optimizer.iterations, autoencoder_optimizer.learning_rate, discriminator_optimizer.learning_rate 


    """
    from PIL import Image
    img0 = np.asarray(Image.open("/home/chaoji/work/genmo/diffusion/latent-diffusion/000016.jpg"))
    img1 = np.asarray(Image.open("/home/chaoji/work/genmo/diffusion/latent-diffusion/000415.jpg"))
    img2 = np.asarray(Image.open("/home/chaoji/work/genmo/diffusion/latent-diffusion/002628.jpg"))

    images = np.stack([img0, img1, img2, ])
    inputs = images.astype("float32") / 127.5 - 1
   
    grads = train_step_autoencoder_discriminator(inputs) 
    return grads 
    """

    for i, inputs in enumerate(dataset):
      if i >= self._global_step_discriminator:
        ae_loss, d_loss, ae_step, d_step, ae_lr, d_lr = train_step_autoencoder_discriminator(inputs)
      else:
        ae_loss, ae_step, ae_lr = train_step_autoencoder(inputs)
      if i % 100 == 0:
        print(i, ae_loss.numpy())

class AutoencoderTrainerVQ(_AutoencoderTrainer):
  def __init__(
      self,
      autoencoder,
      lpips,
      discriminator,
      global_step_discriminator=1,
      codebook_weight=1.,
      lpips_weight=1.,
      kl_weight=1.,
      discriminator_weight=1.,
      discriminator_factor=1.,
      discriminator_loss_type="hinge",

    ):
    super(AutoencoderTrainerVQ, self).__init__(
      autoencoder=autoencoder,
      lpips=lpips,
      discriminator=discriminator,
      global_step_discriminator=global_step_discriminator,
      lpips_weight=lpips_weight,
      kl_weight=kl_weight,
      discriminator_weight=discriminator_weight,
      discriminator_factor=discriminator_factor,
      discriminator_loss_type=discriminator_loss_type,
    )
    self._codebook_weight = codebook_weight

  def train(
      self,
      dataset,
      autoencoder_optimizer,
      discriminator_optimizer,
      #ckpt,
      #ckpt_path,
      #persist_per_iterations,
      log_per_iterations=100,
      logdir="log"
    ):

    train_step_signature = [
        tf.TensorSpec(
            shape=dataset.element_spec.shape,
            dtype=dataset.element_spec.dtype,
        ),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step_autoencoder(inputs):

      with tf.GradientTape() as tape:
        latents, codebook_loss, _ = self._autoencoder.encode(inputs)
        outputs = self._autoencoder.decode(latents)
        nll_loss = self._compute_nll_loss(inputs, outputs, reduce_loss=True)

        autoencoder_loss = nll_loss + codebook_loss * self._codebook_weight

      autoencoder_grads = tape.gradient(autoencoder_loss, self._autoencoder.trainable_weights)
      autoencoder_optimizer.apply_gradients(zip(autoencoder_grads, self._autoencoder.trainable_weights))
      return autoencoder_loss, autoencoder_optimizer.iterations, autoencoder_optimizer.learning_rate


    @tf.function(input_signature=train_step_signature)
    def train_step_autoencoder_discriminator(inputs):

      with tf.GradientTape(persistent=True) as tape:
        latents, codebook_loss, _ = self._autoencoder.encode(inputs)
        outputs = self._autoencoder.decode(latents)
        nll_loss = self._compute_nll_loss(inputs, outputs, reduce_loss=True)
        autoencoder_loss = nll_loss + codebook_loss * self._codebook_weight


        logits_fake = self._discriminator(outputs)
        generator_loss = -tf.reduce_mean(logits_fake)
        adaptive_weight = self._compute_adaptive_weight(nll_loss, generator_loss)

        autoencoder_loss += adaptive_weight * self._discriminator_factor * generator_loss

        logits_real = self._discriminator(tf.stop_gradient(inputs))
        logits_fake = self._discriminator(tf.stop_gradient(outputs))

        discriminator_loss = self._discriminator_loss(logits_real, logits_fake)
        discriminator_loss *= self._discriminator_factor

      autoencoder_grads = tape.gradient(autoencoder_loss, self._autoencoder.weights)
      autoencoder_optimizer.apply_gradients(zip(autoencoder_grads, self._autoencoder.trainable_weights))
      discriminator_grads = tape.gradient(discriminator_loss, self._discriminator.trainable_weights)
      discriminator_optimizer.apply_gradients(zip(discriminator_grads, self._discriminator.trainable_weights))

      return autoencoder_loss, discriminator_loss, autoencoder_optimizer.iterations, discriminator_optimizer.iterations, autoencoder_optimizer.learning_rate, discriminator_optimizer.learning_rate

    """
    from PIL import Image
    img0 = np.asarray(Image.open("/home/chaoji/work/genmo/diffusion/latent-diffusion/000016.jpg"))
    img1 = np.asarray(Image.open("/home/chaoji/work/genmo/diffusion/latent-diffusion/000415.jpg"))
    img2 = np.asarray(Image.open("/home/chaoji/work/genmo/diffusion/latent-diffusion/002628.jpg"))

    images = np.stack([img0, img1, img2, ])
    inputs = images.astype("float32") / 127.5 - 1

    grads = train_step_autoencoder_discriminator(inputs)
    return grads
    """

    for i, inputs in enumerate(dataset):
      if i >= self._global_step_discriminator:
        ae_loss, d_loss, ae_step, d_step, ae_lr, d_lr = train_step_autoencoder_discriminator(inputs)
        #print(i, ae_loss.numpy(), d_loss.numpy(), ae_lr, d_lr, ae_step, d_step)
      else:
        ae_loss, ae_step, ae_lr = train_step_autoencoder(inputs)
        #print(i, ae_loss.numpy(), ae_lr, ae_step)

      if i % 100 == 0:
        if i >= self._global_step_discriminator:
          print(i, ae_loss.numpy(), d_loss.numpy())
        else:
          print(i, ae_loss.numpy())


class LDMTrainer(object):
  pass


