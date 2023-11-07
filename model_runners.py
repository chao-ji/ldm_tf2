"""Defines classes for training autoencoders, latent diffusion models, and
samplers using pretrained latent diffusion models.
"""
import os
import sys
import numpy as np
import tensorflow as tf

from autoencoder import AutoencoderKL, AutoencoderVQ


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

D_LOSS_MAP = {"hinge": hinge_d_loss, "vanilla": vanilla_d_loss}


def _extract(data, t):
  """Extract some coefficients at specified time steps, then reshape to
  [batch_size, 1, 1, 1] for broadcasting purpose.

  Args:
    data (Tensor): tensor of shape [num_steps], coefficients for a beta
      schedule.
    t (Tensor): int tensor of shape [batch_size], sampled time steps in a batch.

  Returns:
    outputs (Tensor): tensor of shape [batch_size, 1, 1, 1], the extracted
      coefficients.
  """
  outputs = tf.reshape(
      tf.gather(tf.cast(data, dtype="float32"), t, axis=0),
      [-1, 1, 1, 1]
  )
  return outputs


class _AutoencoderTrainer(object):
  """Base class for autoencoder trainers."""
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
    self._discriminator_loss = D_LOSS_MAP[discriminator_loss_type]

  def _compute_adaptive_weight(self, nll_loss, g_loss):
    last_layer = self._autoencoder.get_last_layer()

    nll_grads = tf.gradients(nll_loss, last_layer)[0]
    g_grads = tf.gradients(g_loss, last_layer)[0]

    #nll_grads2 = tf.constant(np.load("../latent-diffusion/nll_grads.npy").transpose(2, 3, 1, 0))
    #g_grads2 = tf.constant(np.load("../latent-diffusion/g_grads.npy").transpose(2, 3, 1, 0))

    #tf.print("nll_grads", tf.reduce_sum(tf.reshape(nll_grads, [-1]) * tf.reshape(nll_grads2, [-1])) /(tf.norm(nll_grads) * tf.norm(nll_grads2)))
    #tf.print("g_grads", tf.reduce_sum(tf.reshape(g_grads, [-1]) * tf.reshape(g_grads2, [-1])) / (tf.norm(g_grads) * tf.norm(g_grads2))) 

    weight = tf.norm(nll_grads) / (tf.norm(g_grads) + 1e-4)
    weight = tf.clip_by_value(weight, 0.0, 1e4)
    weight = tf.stop_gradient(weight)
    weight = weight * self._discriminator_weight
    return weight

  def _compute_nll_loss(self, inputs, outputs, reduce_loss=False):
    reconstruction_loss = tf.abs(inputs - outputs)
    lpips_loss = self._lpips(inputs, outputs)
    #tf.print("lpips_loss", tf.reduce_mean(lpips_loss))
    nll_loss = reconstruction_loss + self._lpips_weight * lpips_loss
    #tf.print("recon_loss", tf.reduce_mean(nll_loss))

    if reduce_loss:
      nll_loss = tf.reduce_sum(nll_loss) / nll_loss.shape[0]
    else:
      nll_loss = tf.reduce_mean(nll_loss)
    return nll_loss


class AutoencoderTrainerKL(_AutoencoderTrainer):
  """KL Divergence regularized autoencoder."""
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
      ae_optimizer,
      d_optimizer,
      ckpt,
      ckpt_path,
      num_iterations,
      persist_per_iterations=5000,
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
        posterior = self._autoencoder.encode(inputs, training=True)
        latents = posterior.sample()
        outputs = self._autoencoder.decode(latents, training=True)
        nll_loss = self._compute_nll_loss(inputs, outputs, reduce_loss=True)
        kl_loss = posterior.kl()
        kl_loss = tf.reduce_sum(kl_loss) / kl_loss.shape[0]
        ae_loss = nll_loss + self._kl_weight * kl_loss

      ae_grads = tape.gradient(ae_loss, self._autoencoder.trainable_weights)
      ae_optimizer.apply_gradients(
          zip(ae_grads, self._autoencoder.trainable_weights))
      return ae_loss, ae_optimizer.iterations - 1, ae_optimizer.learning_rate

    @tf.function(input_signature=train_step_signature)
    def train_step_autoencoder_discriminator(inputs):
      with tf.GradientTape(persistent=True) as tape:
        posterior = self._autoencoder.encode(inputs, training=True)
        latents = posterior.sample()
        outputs = self._autoencoder.decode(latents, training=True)
        nll_loss = self._compute_nll_loss(inputs, outputs, reduce_loss=True)
        kl_loss = posterior.kl()
        kl_loss = tf.reduce_sum(kl_loss) / kl_loss.shape[0]
        ae_loss = nll_loss + self._kl_weight * kl_loss

        logits_fake = self._discriminator(outputs)
        g_loss = -tf.reduce_mean(logits_fake)
        adaptive_weight = self._compute_adaptive_weight(nll_loss, g_loss)
        ae_loss += adaptive_weight * self._discriminator_factor * g_loss

        logits_real = self._discriminator(tf.stop_gradient(inputs))
        logits_fake = self._discriminator(tf.stop_gradient(outputs))
        d_loss = self._discriminator_loss(logits_real, logits_fake)
        d_loss *= self._discriminator_factor

      ae_grads = tape.gradient(ae_loss, self._autoencoder.trainable_weights)
      ae_optimizer.apply_gradients(
          zip(ae_grads, self._autoencoder.trainable_weights))
      d_grads = tape.gradient(d_loss, self._discriminator.trainable_weights)
      d_optimizer.apply_gradients(
          zip(d_grads, self._discriminator.trainable_weights))

      return (ae_loss,
              d_loss,
              ae_optimizer.iterations - 1,
              d_optimizer.iterations - 1,
              ae_optimizer.learning_rate,
              d_optimizer.learning_rate) 

    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
      print("[INFO] Restoring from checkpoint: %s ..." % latest_ckpt)
      ckpt.restore(latest_ckpt)
    else:
      print("[INFO] Training from scratch...")

    for i, inputs in enumerate(dataset):
      if i >= self._global_step_discriminator:
        ae_loss, d_loss = train_step_autoencoder_discriminator(inputs)[:2]
        ae_loss, d_loss = ae_loss.numpy(), d_loss.numpy()
      else:
        ae_loss = train_step_autoencoder(inputs)[0]
        ae_loss = ae_loss.numpy()

      if i % log_per_iterations == 0:
        if i >= self._global_step_discriminator:
          print(f"global step: {i}, ae_loss: {ae_loss}, d_loss: {d_loss}")
        else:
          print(f"global step: {i}, ae_loss: {ae_loss}")

      if i % persist_per_iterations == 0:
        ckpt.save(os.path.join(ckpt_path, "aekl"))

      if i == num_iterations:
        break


class AutoencoderTrainerVQ(_AutoencoderTrainer):
  """Vector quantization regularized autoencoder."""
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
      ae_optimizer,
      d_optimizer,
      ckpt,
      ckpt_path,
      num_iterations,
      persist_per_iterations=5000,
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
        latents, quant_loss, _ = self._autoencoder.encode(inputs, training=True)
        outputs = self._autoencoder.decode(latents, training=True)
        nll_loss = self._compute_nll_loss(inputs, outputs, reduce_loss=True)
        ae_loss = nll_loss + quant_loss * self._codebook_weight

      ae_grads = tape.gradient(ae_loss, self._autoencoder.trainable_weights)
      ae_optimizer.apply_gradients(
          zip(ae_grads, self._autoencoder.trainable_weights))
      return ae_loss, ae_optimizer.iterations - 1, ae_optimizer.learning_rate

    @tf.function(input_signature=train_step_signature)
    def train_step_autoencoder_discriminator(inputs):
      with tf.GradientTape(persistent=True) as tape:
        latents, quant_loss, _ = self._autoencoder.encode(inputs, training=True)
        outputs = self._autoencoder.decode(latents, training=True)
        nll_loss = self._compute_nll_loss(inputs, outputs, reduce_loss=True)
        ae_loss = nll_loss + quant_loss * self._codebook_weight

        logits_fake = self._discriminator(outputs)
        g_loss = -tf.reduce_mean(logits_fake)
        adaptive_weight = self._compute_adaptive_weight(nll_loss, g_loss)
        ae_loss += adaptive_weight * self._discriminator_factor * g_loss

        logits_real = self._discriminator(tf.stop_gradient(inputs))
        logits_fake = self._discriminator(tf.stop_gradient(outputs))
        d_loss = self._discriminator_loss(logits_real, logits_fake)
        d_loss *= self._discriminator_factor

      ae_grads = tape.gradient(ae_loss, self._autoencoder.weights)
      ae_optimizer.apply_gradients(
          zip(ae_grads, self._autoencoder.trainable_weights))
      d_grads = tape.gradient(d_loss, self._discriminator.trainable_weights)
      d_optimizer.apply_gradients(
          zip(d_grads, self._discriminator.trainable_weights))

      return (ae_loss,
              d_loss,
              ae_optimizer.iterations - 1,
              d_optimizer.iterations - 1,
              ae_optimizer.learning_rate,
              d_optimizer.learning_rate)

    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
      print("[INFO] Restoring from checkpoint: %s ..." % latest_ckpt)
      ckpt.restore(latest_ckpt)
    else:
      print("[INFO] Training from scratch...")

    for i, inputs in enumerate(dataset):
      if i >= self._global_step_discriminator:
        ae_loss, d_loss = train_step_autoencoder_discriminator(inputs)[:2]
        ae_loss, d_loss = ae_loss.numpy(), d_loss.numpy()
      else:
        ae_loss = train_step_autoencoder(inputs)[0]
        ae_loss = ae_loss.numpy()

      if i % log_per_iterations == 0:
        if i >= self._global_step_discriminator:
          print(f"global step: {i}, ae_loss: {ae_loss}, d_loss: {d_loss}")
        else:
          print(f"global step: {i}, ae_loss: {ae_loss}")

      if i % persist_per_iterations == 0:
        ckpt.save(os.path.join(ckpt_path, "aevq"))

      if i == num_iterations:
        break


class LatentDiffusionModel(object):

  def __init__(
      self,
      unet,
      autoencoder,
      cond_stage_model,
      num_steps=1000,
      beta_start=1e-4,
      beta_end=2e-2,
      v_posterior=0.,
      scale_factor=0.18215,
      eta=0.,
      num_ddim_steps=50,
    ):
    self._unet = unet
    self._autoencoder = autoencoder
    self._cond_stage_model = cond_stage_model

    self._num_steps = num_steps
    self._beta_start = beta_start
    self._beta_end = beta_end
    self._v_posterior = v_posterior
    self._scale_factor = scale_factor
    self._eta = eta
    self._num_ddim_steps = num_ddim_steps

    self._betas = tf.cast(
        tf.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2,
        dtype="float64",
    )
    self._alphas = 1. - self._betas
    self._alphas_cumprod = tf.math.cumprod(self._alphas, axis=0)
    self._alphas_cumprod_prev = tf.concat(
        [[1.], self._alphas_cumprod[:-1]], axis=0)
    self._sqrt_alphas_cumprod = tf.sqrt(self._alphas_cumprod)
    self._sqrt_one_minus_alphas_cumprod = tf.sqrt(1. - self._alphas_cumprod)
    self._log_one_minus_alphas_cumprod = tf.math.log(1. - self._alphas_cumprod)
    self._sqrt_recip_alphas_cumprod = tf.sqrt(1. / self._alphas_cumprod)
    self._sqrt_recipm1_alphas_cumprod = tf.sqrt(1. / self._alphas_cumprod - 1)
    self._posterior_variance = (
        (1 - v_posterior) * self._betas *
          (1. - self._alphas_cumprod_prev) /
          (1. - self._alphas_cumprod)
        + v_posterior * self._betas
    )

    self._posterior_log_variance_clipped = tf.math.log(
        tf.maximum(self._posterior_variance, 1e-20))
    self._posterior_mean_coef1 = self._betas * tf.sqrt(
        self._alphas_cumprod_prev) / (1. - self._alphas_cumprod)
    self._posterior_mean_coef2 = (1. - self._alphas_cumprod_prev) * tf.sqrt(
        self._alphas) / (1. - self._alphas_cumprod)

    self._ddim_steps = tf.range(
        0, num_steps, num_steps // num_ddim_steps, dtype="int32")
    if self._num_ddim_steps < self._num_steps:
      self._ddim_steps += 1

    alphas_cumprod = tf.gather(self._alphas_cumprod, self._ddim_steps)
    self._ddim_alphas_cumprod_prev = tf.concat([
        [self._alphas_cumprod[0]],
        tf.gather(self._alphas_cumprod, self._ddim_steps[:-1])], axis=0
    )
    self._ddim_sigmas = eta * tf.sqrt(
        (1 - self._ddim_alphas_cumprod_prev) / (1 - alphas_cumprod) *
        (1 - alphas_cumprod / self._ddim_alphas_cumprod_prev)
    )
    self._ddim_sqrt_recip_alphas_cumprod = tf.gather(
        self._sqrt_recip_alphas_cumprod, self._ddim_steps)
    self._ddim_sqrt_recipm1_alphas_cumprod = tf.gather(
        self._sqrt_recipm1_alphas_cumprod, self._ddim_steps)

  def decode_first_stage(self, latents):
    latents = latents / self._scale_factor

    if isinstance(self._autoencoder, AutoencoderKL):
      outputs = self._autoencoder.decode(latents, training=False)
    elif isinstance(self._autoencoder, AutoencoderVQ):
      outputs = self._autoencoder.decode(latents, force_quantize=True, training=False)
    else:
      raise NotImplementedError("autoencoder not implemented")
    return outputs


class LatentDiffusionModelSampler(LatentDiffusionModel):
  def ddim_sample(
      self,
      xt,
      cond,
      index,
      guidance_scale=1.,
      clip_denoised=True,
      return_pred_x0=False
    ):

    _maybe_clip = lambda x: (tf.clip_by_value(x, -1, 1) if clip_denoised else x)
    t = tf.fill([xt.shape[0] * 2], self._ddim_steps[index])

    eps_uncond, eps = tf.split(
        self._unet(tf.concat([xt, xt], axis=0), t, cond), 2, axis=0)
    eps = eps_uncond + guidance_scale * (eps - eps_uncond)

    pred_x0 = (
        _extract(self._ddim_sqrt_recip_alphas_cumprod, index) * xt -
        _extract(self._ddim_sqrt_recipm1_alphas_cumprod, index) * eps
    )
    pred_x0 = _maybe_clip(pred_x0)

    alphas_cumprod_prev = _extract(self._ddim_alphas_cumprod_prev, index)
    model_std = _extract(self._ddim_sigmas, index)
    model_mean = tf.sqrt(alphas_cumprod_prev) * pred_x0 + tf.sqrt(
        1 - alphas_cumprod_prev - model_std**2) * eps

    noise = tf.random.normal(xt.shape)
    #noise = tf.constant(np.load(f"../latent-diffusion/noise{index}.npy").transpose(0, 2, 3, 1))
    sample = model_mean + noise * model_std
    if return_pred_x0:
      return sample, pred_x0
    else:
      return sample

  def ddim_p_sample_loop(self, cond_model_inputs, shape, guidance_scale=5.):
    context = self._cond_stage_model(cond_model_inputs)
    index = tf.size(self._ddim_steps) - 1
    #xt = tf.constant(np.load("/home/chaoji/work/genmo/diffusion/latent-diffusion/x.npy").transpose(0, 2, 3, 1))
    xt = tf.random.normal(shape)

    uncond=context[:4]
    cond=context[4:]
    cond_combined = tf.concat([uncond, cond], axis=0)

    x_final = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
          cond=lambda index, _: tf.greater_equal(index, 0),
          body=lambda index, xt: [
            index - 1,
            self.ddim_sample(
              xt=xt,
              cond=cond_combined,
              index=index,
              guidance_scale=guidance_scale,
              clip_denoised=False,
              return_pred_x0=False,
            ),
          ],
          loop_vars=[index, xt],
          shape_invariants=[index.shape, xt.shape],
        ),
    )[1]
    print(f"[INFO] Done running denoising for {self._num_ddim_steps} steps with"
      f" eta {self._eta}")
    #print(x_final.shape, x_final.numpy().sum())
    images = self.decode_first_stage(x_final)
    print("[INFO] Done decoding images from the final latent variable.")
    #print(images.shape, images.numpy().sum())
    return images 

  @tf.function
  def ddim_p_sample_loop_progressive(
      self,
      cond_model_inputs,
      shape,
      guidance_scale=5.,
      record_freq=5,
    ):
    context = self._cond_stage_model(cond_model_inputs)

    index = tf.size(self._ddim_steps) - 1
    xt = tf.random.normal(shape)

    uncond=context[:4]
    cond=context[4:]
    cond_combined = tf.concat([uncond, cond], axis=0)

    num_records = tf.size(self._ddim_steps) // record_freq
    sample_progress = tf.zeros(
        [shape[0], num_records, *shape[1:]], dtype="float32")
    pred_x0_progress = tf.zeros(
        [shape[0], num_records, *shape[1:]], dtype="float32")

    def _loop_body(index, xt, sample_progress, pred_x0_progress):
      sample, pred_x0 = self.ddim_p_sample(
          xt=xt,
          cond=cond_combined,
          index=index,
          guidance_scale=guidance_scale,
          clip_denoised=False,
          return_pred_x0=True,
      )
      insert_mask = tf.equal(tf.math.floordiv(index, record_freq),
                             tf.range(num_records, dtype="int32"))
      insert_mask = tf.reshape(
          tf.cast(insert_mask, dtype="float32"),
          [1, num_records, *([1] * len(shape[1:]))])
      new_sample_progress = insert_mask * sample[:, tf.newaxis] + (
          1. - insert_mask) * sample_progress
      new_pred_x0_progress = insert_mask * pred_x0[:, tf.newaxis] + (
          1. - insert_mask) * pred_x0_progress

      return [index - 1, sample, new_sample_progress, new_pred_x0_progress]

    _, x_final, sample_prog_final, pred_x0_prog_final = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
          cond=lambda index, _1, _2, _3: tf.greater_equal(index, 0),
          body=_loop_body,
          loop_vars=[index, xt, sample_progress, pred_x0_progress],
          shape_invariants=[index.shape, xt.shape] + [sample_progress.shape] * 2
        )
    )

    x_final = self.decode_first_stage(x_final)
    flat_shape = tf.concat([[shape[0] * num_records], shape[1:]], axis=0)
    sample_prog_final = self.decode_first_stage(
        tf.reshape(sample_prog_final, flat_shape))
    pred_x0_prog_final = self.decode_first_stage(
        tf.reshape(pred_x0_prog_final, flat_shape))

    out_shape = tf.concat([[shape[0], num_records], x_final.shape[1:]], axis=0)
    sample_prog_final = tf.reshape(sample_prog_final, out_shape)
    pred_x0_prog_final = tf.reshape(pred_x0_prog_final, out_shape)
    return x_final, sample_prog_final, pred_x0_prog_final


class LatentDiffusionModelTrainer(LatentDiffusionModel):

  def q_sample(self, x0, t, eps):
    """Sample a noised version of `x0` (original image) according to `t` (
    sampled time steps), i.e. `q(x_t | x_0)`. `t` contains integers sampled from
    0, 1, ..., num_steps - 1, so 0 means 1 diffusion step, and so on.

    Args:
      x0 (Tensor): tensor of shape [batch_size, height, width, 3], the original
        image.
      t (Tensor): int tensor of shape [batch_size], time steps in a batch.
      eps (Tensor): tensor of shape [batch_size, height, width, 3], noise from
        prior distribution.

    Returns:
      xt (Tensor): tensor of shape [batch_size, height, width, 3], noised
        version of `x0`.
    """
    xt = (
        _extract(self._sqrt_alphas_cumprod, t) * x0 +
        _extract(self._sqrt_one_minus_alphas_cumprod, t) * eps
    )
    return xt

  def get_latents(self, inputs):
    """Compute latent variables using a pre-trained autoencoder (KL- or
    VQ-regularized), and scale it by `scale_factor` (1 / {componentwise std of
    latents}), and make the latents non-differentiable (i.e. stop gradient).

    Args:
      inputs (Tensor): tensor shape [batch_size, image_height, image_width, 3],
        original images in pixel space.

    Returns:
      latents (Tensor): tensor of shape [batch_size, height, width,
        latent_channels], images encoded as latent variables.
    """
    if isinstance(self._autoencoder, AutoencoderKL):
      posterior = self._autoencoder.encode(inputs)
      latents = posterior.sample()
    elif isinstance(self._autoencoder, AutoencoderVQ):
      latents = self._autoencoder.encode(inputs, only_encode=True)
    else:
      raise NotImplementedError("Invalid autoencoder")

    latents = self._scale_factor * latents
    latents = tf.stop_gradient(latents)
    return latents

  def train(
      self,
      dataset,
      optimizer,
      ckpt,
      ckpt_path,
      num_iterations,
      train_cond_model=False,
      persist_per_iterations=1000,
      log_per_iterations=100,
      logdir="log",
    ):
    """"""
    train_step_signature = [
        tf.TensorSpec(
            shape=dataset.element_spec[0].shape,
            dtype=dataset.element_spec[0].dtype,
        ),
        tf.TensorSpec(
            shape=dataset.element_spec[1].shape,
            dtype=dataset.element_spec[1].dtype,
        ),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(images, cond_model_inputs):
      loss = self.compute_loss(images, cond_model_inputs)

      variables = self._unet.trainable_variables
      if train_cond_model:
        variables += self._cond_stage_model.trainable_variables

      gradients = tf.gradients(loss, variables)
      optimizer.apply_gradients(zip(gradients, variables))

      step = optimizer.iterations
      lr = optimizer.learning_rate

      return loss, step - 1, lr

    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
      print("[INFO] Restoring from checkpoint: %s ..." % latest_ckpt)
      ckpt.restore(latest_ckpt)
    else:
      print("[INFO] Training from scratch...")

    for images, cond_model_inputs in dataset:
      loss, step, lr = train_step(images, cond_model_inputs)
      print(loss.numpy(), step.numpy())
      #if step.numpy() % log_per_iterations == 0:
      #  print("global step: %d, loss: %f, learning rate:" %
      #      (step.numpy(), loss.numpy()), lr.numpy())
      #  sys.stdout.flush()

      #if step.numpy() % persist_per_iterations == 0:
      #  print("Saving checkpoint at global step %d ..." % step.numpy())
      #  ckpt.save(os.path.join(ckpt_path, "ddpm"))

      if step == num_iterations:
        break


  def compute_loss(self, inputs, cond_model_inputs):
    batch_size = tf.shape(inputs)[0]
    t = tf.random.uniform((batch_size,), 0, self._num_steps, dtype="int32")
    latents = self.get_latents(inputs)

    context = self._cond_stage_model(cond_model_inputs)
    noise = tf.random.normal(tf.shape(latents))
    xt = self.q_sample(latents, t, noise)
    eps = self._unet(xt, tf.cast(t, "float32"), context, training=True)
    loss = tf.reduce_mean(tf.math.squared_difference(noise, eps), [1, 2, 3])
    loss = tf.reduce_mean(loss)
    return loss
