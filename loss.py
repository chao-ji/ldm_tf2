import numpy as np
import tensorflow as tf

from discriminator import NLayerDiscriminator
from distribution import DiagonalGaussian
from autoencoder import AutoencoderKL, VQModel


def hinge_d_loss(logits_real, logits_fake):
  loss_real = tf.reduce_mean(tf.nn.relu(1. - logits_real))
  loss_fake = tf.reduce_mean(tf.nn.relu(1. + logits_fake))
  d_loss = 0.5 * (loss_real + loss_fake)
  return d_loss


class NetLinLayer(tf.keras.layers.Layer):
  """ A single linear layer which does a 1x1 conv """
  def __init__(self, chn_out=1, use_dropout=False):
    super(NetLinLayer, self).__init__()

    self._use_dropout = use_dropout
    self._dropout_layer = tf.keras.layers.Dropout(rate=0.5)
    self._conv2d = tf.keras.layers.Conv2D(chn_out, kernel_size=1, strides=1, padding="VALID", use_bias=False) 

  def call(self, inputs):
    if self._use_dropout:
      outputs = self._dropout_layer(inputs)
    else:
      outputs = inputs
    outputs = self._conv2d(outputs)
    return outputs


class ScalingLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(ScalingLayer, self).__init__()
    self._shift = tf.constant([-.030, -.088, -.188])[None, None, None, :]
    self._scale = tf.constant([.458, .448, .450])[None, None, None, :]

  def call(self, inputs):
    outputs = (inputs - self._shift) / self._scale
    return outputs



def normalize_tensor(x, eps=1e-10):
    norm_factor = tf.sqrt(tf.reduce_sum(x ** 2, axis=-1, keepdims=True))
    return x / (norm_factor + eps)

class VGG16(tf.keras.layers.Layer):
  def __init__(self):
    super(VGG16, self).__init__()

    self._conv_layers = [
      [
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="VALID",)
        for _ in range(2)
      ], [
        tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="VALID",)
        for _ in range(2)
      ], [
        tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="VALID",)
        for _ in range(3)
      ], [
        tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding="VALID",)
        for _ in range(3)
      ], [
        tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding="VALID",)
        for _ in range(3)
      ]
    ]
    self._conv_indices = [[0, 1]] * 2 + [[0, 1, 2]] * 3 

  def call(self, inputs): 
    outputs = inputs
    features = []

    for i in range(5):
      if i > 0:
        outputs = tf.nn.max_pool2d(outputs, ksize=2, strides=2, padding="VALID")
      for j in self._conv_indices[i]:
        outputs = tf.pad(outputs, [[0, 0], [1, 1], [1, 1], [0, 0]]) 
        outputs = self._conv_layers[i][j](outputs)
        outputs = tf.nn.relu(outputs)
      features.append(outputs)

    return features 


class LPIPS(tf.keras.layers.Layer):
  # Learned perceptual metric
  def __init__(self, use_dropout=True):
    super(LPIPS, self).__init__()
    self.scaling_layer = ScalingLayer()
    self.chns = [64, 128, 256, 512, 512]  # vg16 features
    self.net = VGG16()
    self.lin0 = NetLinLayer(use_dropout=use_dropout)
    self.lin1 = NetLinLayer(use_dropout=use_dropout)
    self.lin2 = NetLinLayer(use_dropout=use_dropout)
    self.lin3 = NetLinLayer(use_dropout=use_dropout)
    self.lin4 = NetLinLayer(use_dropout=use_dropout)

  def call(self, inputs, targets):
    inputs = self.scaling_layer(inputs)
    targets = self.scaling_layer(targets)

    inputs_features = self.net(inputs)
    targets_features = self.net(targets) 

    lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

    feats0, feats1, diffs = {}, {}, {}
    for kk in range(5):
      feats0[kk] = normalize_tensor(inputs_features[kk])
      feats1[kk] = normalize_tensor(targets_features[kk])
      diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

    res = [tf.reduce_mean(lins[kk](diffs[kk]), axis=[1, 2], keepdims=True) for kk in range(5)]
    val = tf.add_n(res)
    return val


class LPIPSWithDiscriminator(tf.keras.layers.Layer):
  def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):
    super(LPIPSWithDiscriminator, self).__init__()
    assert disc_loss in ["hinge", "vanilla"]
    self.kl_weight = kl_weight
    self.pixel_weight = pixelloss_weight
    self.perceptual_loss = LPIPS()
    self.perceptual_weight = perceptual_weight
    
    self.discriminator = NLayerDiscriminator(num_layers=disc_num_layers)
    self.discriminator_iter_start = disc_start
    self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
    self.disc_factor = disc_factor
    self.discriminator_weight = disc_weight
    self.disc_conditional = disc_conditional
    self.logvar = 0.0

  def call(self, inputs, targets, posteriors=None, optimizer_idx=None,
                global_step=None, last_layer=None, cond=None, split="train",
                weights=None):
    rec_loss = tf.abs(inputs - targets)
    if self.perceptual_weight > 0:
      p_loss = self.perceptual_loss(inputs, targets)
      rec_loss = rec_loss + self.perceptual_weight * p_loss 

    nll_loss = rec_loss / tf.exp(self.logvar) + self.logvar
    weighted_nll_loss = nll_loss

    #return weighted_nll_loss
    weighted_nll_loss = tf.reduce_sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
    nll_loss = tf.reduce_sum(nll_loss) / nll_loss.shape[0]
    kl_loss = posteriors.kl()
    kl_loss = tf.reduce_sum(kl_loss) / kl_loss.shape[0]    

    if optimizer_idx == 0:
      # autoencoder loss
      logits_fake = self.discriminator(targets)
      g_loss = -tf.reduce_mean(logits_fake)
      d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)

      #return g_loss 
      disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
      loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

      log = {"total_loss": loss, "kl_loss": kl_loss, "rec_loss": rec_loss, "nll_loss": nll_loss, "d_weight": d_weight, "disc_factor": disc_factor, "g_loss": g_loss}

      return loss, log
    elif optimizer_idx == 1:
      # discriminator loss
      logits_real = self.discriminator(tf.stop_gradient(inputs))
      logits_fake = self.discriminator(tf.stop_gradient(targets))
      #print("logits_real", logits_real.numpy().sum(), logits_real.shape)
      #print("logits_fake", logits_fake.numpy().sum(), logits_fake.shape)
      d_loss = self.disc_loss(logits_real, logits_fake)

      disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
      d_loss = d_loss * disc_factor

      log = {"disc_loss": d_loss, "logits_real": logits_real, "logits_fake": logits_fake}
      return d_loss, log
    else:
      raise ValueError("adfadsf")

  def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
    nll_grads = tf.gradients(nll_loss, last_layer)
    #print("nll_grads", type(nll_grads), len(nll_grads))
    nll_grads = nll_grads[0]
    #return nll_grads
    g_grads = tf.gradients(g_loss, last_layer)
    #print("g_grads", type(g_grads), len(g_grads))
    g_grads = g_grads[0]
    #return tf.norm(nll_grads), tf.norm(g_grads)
    d_weight = tf.norm(nll_grads) / (tf.norm(g_grads) + 1e-4)
    d_weight = tf.clip_by_value(d_weight, 0.0, 1e4)
    d_weight = tf.stop_gradient(d_weight)
    d_weight = d_weight * self.discriminator_weight
    return d_weight


class VQLPIPSWithDiscriminator(tf.keras.layers.Layer):
  def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
    super(VQLPIPSWithDiscriminator, self).__init__()

    assert disc_loss in ["hinge", "vanilla"]
    self.codebook_weight = codebook_weight
    self.pixel_weight = pixelloss_weight
    self.perceptual_loss = LPIPS()
    self.perceptual_weight = perceptual_weight 

    self.discriminator = NLayerDiscriminator(num_layers=disc_num_layers)
    self.discriminator_iter_start = disc_start

    if disc_loss == "hinge":
      self.disc_loss = hinge_d_loss
    elif disc_loss == "vanilla":
      self.disc_loss = vanilla_d_loss
    else:
      raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
    print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
    self.disc_factor = disc_factor
    self.discriminator_weight = disc_weight
    self.disc_conditional = disc_conditional    

  def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
    nll_grads = tf.gradients(nll_loss, last_layer)
    #print("nll_grads", type(nll_grads), len(nll_grads))
    nll_grads = nll_grads[0]
    #return nll_grads
    g_grads = tf.gradients(g_loss, last_layer)
    #print("g_grads", type(g_grads), len(g_grads))
    g_grads = g_grads[0]
    #return tf.norm(nll_grads), tf.norm(g_grads)
    d_weight = tf.norm(nll_grads) / (tf.norm(g_grads) + 1e-4)
    d_weight = tf.clip_by_value(d_weight, 0.0, 1e4)
    d_weight = tf.stop_gradient(d_weight)
    d_weight = d_weight * self.discriminator_weight
    return d_weight

  def call(self, codebook_loss, inputs, reconstructions, optimizer_idx,
              global_step, last_layer=None, cond=None, split="train"):
    rec_loss = tf.abs(inputs - reconstructions)
    if self.perceptual_weight > 0:
      p_loss = self.perceptual_loss(inputs, reconstructions)
      rec_loss = rec_loss + self.perceptual_weight * p_loss
    else:
      p_loss = 0.0

    nll_loss = rec_loss
    nll_loss = tf.reduce_mean(nll_loss)
   
    if optimizer_idx == 0:
      # generator update
      assert not self.disc_conditional
      logits_fake = self.discriminator(reconstructions)
      g_loss = -tf.reduce_mean(logits_fake)

      d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)

      disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
      loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * tf.reduce_mean(codebook_loss)

      ae_log = {"total_loss": loss, "quant_loss": tf.reduce_mean(codebook_loss), "nll_loss": nll_loss, "rec_loss": rec_loss, "p_loss": p_loss, "d_weight": d_weight, "disc_factor": disc_factor, "g_loss": g_loss}

      return loss, ae_log
    elif optimizer_idx == 1:
      logits_real = self.discriminator(tf.stop_gradient(inputs))
      logits_fake = self.discriminator(tf.stop_gradient(reconstructions)) 

      disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
      d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

      d_log = {"d_loss": d_loss, "logits_real": logits_real, "logits_fake": logits_fake}

      return d_loss, d_log
    else:
      raise ValueError("adfadsf")


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


if __name__ == "__main__":
  from PIL import Image

  img0 = np.asarray(Image.open("/home/chaoji/work/genmo/diffusion/latent-diffusion/000016.jpg"))
  img1 = np.asarray(Image.open("/home/chaoji/work/genmo/diffusion/latent-diffusion/000415.jpg"))
  img2 = np.asarray(Image.open("/home/chaoji/work/genmo/diffusion/latent-diffusion/002628.jpg"))
  images = np.stack([img0, img1, img2, ])
  inputs = images.astype("float32") / 127.5 - 1


  vq = False

  if vq:
    autoencoder = VQModel(z_channels=4)
    disc_loss = VQLPIPSWithDiscriminator(
        disc_conditional=False,
        disc_in_channels=3,
        disc_num_layers=2,
        disc_start=1,
        disc_weight=0.6,
        codebook_weight=1.0,
    )
    ckpt = tf.train.Checkpoint(autoencoder=autoencoder, disc_loss=disc_loss)
    ckpt.restore("../latent-diffusion/models/first_stage_models/vq-f8/vq-f8-1").expect_partial()

    @tf.function
    def func(inputs):
      quant, diff, (_,_,ind) = autoencoder.encode(inputs)
      dec = autoencoder.decode(quant)
      reconstructions = dec
      qloss = diff 

      # for debugging:
      #tf.print("quant", tf.reduce_sum(quant), quant.shape)
      #tf.print("qloss", tf.reduce_sum(qloss), qloss.shape)
      #tf.print("ind", tf.reduce_sum(ind), ind.shape)

      last_layer = autoencoder.get_last_layer()
      aeloss, ae_log = disc_loss(qloss, inputs, reconstructions, optimizer_idx=0, global_step=50001,
                                              last_layer=last_layer, split="train")
      discloss, d_log = disc_loss(qloss, inputs, reconstructions, optimizer_idx=1, global_step=50001,
                                                last_layer=last_layer, split="train")
      return reconstructions, aeloss, discloss, ae_log, d_log

    recon, ae_loss, d_loss, ae_log, d_log = func(inputs)

  else:
    autoencoder = AutoencoderKL(z_channels=4)
    disc_loss = LPIPSWithDiscriminator(disc_start=50001,
        kl_weight=0.000001,
        disc_weight=0.5,)

    ckpt = tf.train.Checkpoint(autoencoder=autoencoder, disc_loss=disc_loss)
    ckpt.restore("../latent-diffusion/models/first_stage_models/kl-f8/kl-f8-1").expect_partial()

    @tf.function
    def func(inputs):
      posterior = autoencoder.encode(inputs)
      latents = posterior.mode()
      recon = autoencoder.decode(latents)

      # for debugging:
      #tf.print("mean", tf.reduce_sum(latents), latents.shape)

      last_layer = autoencoder.get_last_layer()
      ae_loss, ae_log = disc_loss(inputs, recon, posterior, optimizer_idx=0, last_layer=last_layer, global_step=50001)
      d_loss, d_log = disc_loss(inputs, recon, posterior, optimizer_idx=1, last_layer=last_layer, global_step=50001)

      grads = None #tf.gradients(ae_loss, autoencoder._decoder.weights, 1e-7)
      return recon, ae_loss, d_loss, ae_log, d_log, latents

    recon, ae_loss, d_loss, ae_log, d_log, latents = func(inputs)

