import tensorflow as tf
import numpy as np


class DDPM(object):
  def __init__(
      self,
      model=None,
      num_steps=1000,
      beta_schedule="linear",
      linear_start=1e-4,
      linear_end=2e-2,
      cosine_s=8e-3,
      v_posterior=0.
    ):
    self._model = model
    self._num_steps = num_steps
    self._beta_schedule = beta_schedule
    self._linear_start = linear_start
    self._linear_end = linear_end
    self._cosine_s = cosine_s
    self._v_posterior = v_posterior

    self.create_schedule()
   
  def create_schedule(self):

    if self._beta_schedule == "linear":
      betas = np.linspace(self._linear_start ** 0.5, self._linear_end ** 0.5, self._num_steps, dtype="float64") ** 2

    elif schedule == "cosine":
      pass

    elif schedule == "sqrt_linear":
      pass

    elif schedule == "sqrt":
      pass


    #print(self._linear_start, self._linear_end)
    #print("betas")
    #print(betas)
    alphas = 1. - betas


    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
    log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)
    sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

    posterior_variance = (1 - self._v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self._v_posterior * betas
 

    self._betas = betas
    self._alphas_cumprod = alphas_cumprod
    self._alphas_cumprod_prev = alphas_cumprod_prev
    self._sqrt_alphas_cumprod = sqrt_alphas_cumprod 
    self._sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
    self._log_one_minus_alphas_cumprod = log_one_minus_alphas_cumprod
    self._sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod
    self._sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod

    self._posterior_variance = posterior_variance
    self._posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
    self._posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
    self._posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)


class LatentDiffusion(DDPM):
  def __init__(self, cond_stage_model, diffusion_wrapper, first_stage_model=None, scale_factor=0.18215, *args, **kwargs):
    super(LatentDiffusion, self).__init__(*args, **kwargs)

    self.cond_stage_model = cond_stage_model
    self.diffusion_wrapper = diffusion_wrapper
    self.first_stage_model = first_stage_model
    self.scale_factor = scale_factor

  def decode_first_stage(self, z):
    z = 1. / self.scale_factor * z

    self.first_stage_model.decode(z)

  def get_learned_conditioning(self, c):
    return self.cond_stage_model(c)

  def apply_model(self, x_noisy, t, cond, return_ids=False):
    cond = [cond]
    key = 'c_concat' if self.diffusion_wrapper._conditioning_key == 'concat' else 'c_crossattn'
    cond = {key: cond}    
    x_recon = self.diffusion_wrapper(x_noisy, t, **cond)
    return x_recon


class DiffusionWrapper(object):
  def __init__(self, model, conditioning_key):
    self._diffusion_model = model
    self._conditioning_key = conditioning_key

  def __call__(self, x, t, c_concat=None, c_crossattn=None):
    if self._conditioning_key is None:
      out = self.diffusion_model(x, t)
    elif self._conditioning_key == 'concat':
      xc = torch.cat([x] + c_concat, dim=1)
      out = self.diffusion_model(xc, t)
    elif self._conditioning_key == 'crossattn':
      cc = tf.concat(c_crossattn, axis=1)
      out = self._diffusion_model(x, t, context=cc)
    elif self._conditioning_key == 'hybrid':
      xc = torch.cat([x] + c_concat, dim=1)
      cc = torch.cat(c_crossattn, 1)
      out = self.diffusion_model(xc, t, context=cc)
    elif self._conditioning_key == 'adm':
      cc = c_crossattn[0]
      out = self.diffusion_model(x, t, y=cc)
    else:
      raise NotImplementedError()

    return out
      
      
if __name__ == "__main__":
  ddpm = DDPM()


