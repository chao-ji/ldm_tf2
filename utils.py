import tensorflow as tf


def compute_nll_loss(inputs, outputs, lpips_model, lpips_weight):
  recon_loss = tf.abs(inputs - outputs)
  lpips_loss = lpips_model(inputs, outputs)
  nll_loss = recon_loss + lpips_weight * lpips_loss
  return nll_loss

def compute_kl_loss(posterior):
  kl_loss = posterior.kl()
  kl_loss = tf.reduce_sum(kl_loss) / kl_loss.shape[0]
  return kl_loss

def compute_generator_loss(outputs, ):
  pass
