import numpy as np
from transformers import BertTokenizerFast 
import tensorflow as tf
from ddpm import DDPM


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



def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    return sigmas, alphas, alphas_prev














class DDIMSampler2(object):
  def __init__(
      self,  
      model,
      num_ddim_steps=50,
      eta=0.,
    ):
    self._model = model
    
    self._num_steps = self._model._num_steps
    self._num_ddim_steps = num_ddim_steps

    self._betas = self._model._betas
    self._alphas_cumprod = self._model._alphas_cumprod
    self._sqrt_recip_alphas_cumprod = self._model._sqrt_recip_alphas_cumprod
    self._sqrt_recipm1_alphas_cumprod = self._model._sqrt_recipm1_alphas_cumprod
 

    self._ddim_steps = tf.range(0, self._num_steps, self._num_steps // self._num_ddim_steps, dtype="int32")
    if num_ddim_steps < self._num_steps:
      self._ddim_steps += 1

    alphas_cumprod = tf.gather(self._alphas_cumprod, self._ddim_steps)
    self._ddim_alphas_cumprod_prev = tf.concat([[self._alphas_cumprod[0]], tf.gather(self._alphas_cumprod, self._ddim_steps[:-1])], axis=0)

    self._ddim_sigmas = eta * tf.sqrt(
        (1 - self._ddim_alphas_cumprod_prev) / (1 - alphas_cumprod) *
        (1 - alphas_cumprod / self._ddim_alphas_cumprod_prev)
    )


    self._ddim_sqrt_recip_alphas_cumprod = tf.gather(self._sqrt_recip_alphas_cumprod, self._ddim_steps)
    self._ddim_sqrt_recipm1_alphas_cumprod = tf.gather(self._sqrt_recipm1_alphas_cumprod, self._ddim_steps)


  #def ddim_p_sample(self, xt, uncond, cond, index, unconditional_guidance_scale=1., clip_denoised=True, return_pred_x0=False):
  def ddim_p_sample(self, xt, cond, index, unconditional_guidance_scale=1., clip_denoised=True, return_pred_x0=False):
    #tf.print(index)

    _maybe_clip = lambda x: (tf.clip_by_value(x, -1, 1) if clip_denoised else x)

    #t = tf.fill([xt.shape[0]], self._ddim_steps[index])
    t = tf.fill([xt.shape[0] * 2], self._ddim_steps[index])

    e_t_uncond, e_t = tf.split(self._model.apply_model(
        tf.concat([xt, xt], axis=0),
        #tf.cast(tf.concat([t, t], axis=0), "float32"),
        t,
        cond, #tf.concat([uncond, cond], axis=0),
    ), 2, axis=0)
    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
    eps = e_t #self._model(xt, tf.cast(t, "float32"))







    pred_x0 = (
        _extract(self._ddim_sqrt_recip_alphas_cumprod, index) * xt -
        _extract(self._ddim_sqrt_recipm1_alphas_cumprod, index) * eps
    )
    pred_x0 = _maybe_clip(pred_x0)


    alphas_cumprod_prev = _extract(self._ddim_alphas_cumprod_prev, index)

    model_std = _extract(self._ddim_sigmas, index)

    model_mean = tf.sqrt(alphas_cumprod_prev) * pred_x0 + tf.sqrt(1 - alphas_cumprod_prev - model_std**2) * eps

    #noise = tf.random.normal(xt.shape)
    noise = np.load(f"../latent-diffusion/noise{index}.npy").transpose(0, 2, 3, 1)
    noise = tf.constant(noise)
    sample = model_mean + noise * model_std #tf.sqrt(model_variance)

    return sample

  def ddim_p_sample_loop(self, shape, uncond, cond, unconditional_guidance_scale=1.):
    index = tf.size(self._ddim_steps) - 1
    xt = tf.constant(np.load("/home/chaoji/work/genmo/diffusion/latent-diffusion/x.npy").transpose(0, 2, 3, 1)) #tf.random.normal(shape)
    cond_combined = tf.concat([uncond, cond], axis=0)


    x_final = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
          cond=lambda index, _: tf.greater_equal(index, 0),
          body=lambda index, xt: [
              index - 1,
              self.ddim_p_sample(
                  xt=xt, cond=cond_combined, index=index, unconditional_guidance_scale=unconditional_guidance_scale, clip_denoised=False, return_pred_x0=False,
              )
          ],
          loop_vars=[index, xt],
          shape_invariants=[index.shape, xt.shape],
        ),
    )[1]
    return x_final










class DDIMSampler(object):
  def __init__(self, model=None, ddpm_num_timesteps=1000):
    self._model = model
    self._ddpm_num_timesteps = ddpm_num_timesteps

  def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
    self._ddim_steps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps, num_ddpm_timesteps=self._ddpm_num_timesteps) 

    self._betas = self._model._betas
    self._alphas_cumprod = self._model._alphas_cumprod
    self._alphas_cumprod_prev = self._model._alphas_cumprod_prev
    self._sqrt_alphas_cumprod = self._model._sqrt_alphas_cumprod
    self._sqrt_one_minus_alphas_cumprod = self._model._sqrt_one_minus_alphas_cumprod
    self._log_one_minus_alphas_cumprod = self._model._log_one_minus_alphas_cumprod
    self._sqrt_recip_alphas_cumprod = self._model._sqrt_recip_alphas_cumprod
    self._sqrt_recipm1_alphas_cumprod = self._model._sqrt_recipm1_alphas_cumprod

    ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(self._alphas_cumprod, self._ddim_steps, eta=ddim_eta)
    
    self._ddim_sigmas = ddim_sigmas
    self._ddim_alphas = ddim_alphas
    self._ddim_alphas_prev = ddim_alphas_prev
    self._ddim_sqrt_one_minus_alphas = np.sqrt(1. - ddim_alphas)
    
    self._ddim_sigmas_for_original_num_steps = ddim_eta * np.sqrt(
        (1 - self._alphas_cumprod_prev) / (1 - self._alphas_cumprod) * (
            1 - self._alphas_cumprod / self._alphas_cumprod_prev)) 

  def sample(
      self,
      S=50,
      batch_size=4,
      shape=[32, 32, 4],
      conditioning=None,
      callback=None,
      normals_sequence=None,
      img_callback=None,
      quantize_x0=False,
      eta=0.,
      mask=None,
      x0=None,
      temperature=1.,
      noise_dropout=0.,
      score_corrector=None,
      corrector_kwargs=None,
      x_T=None,
      log_every_t=100,
      unconditional_guidance_scale=5.,
      unconditional_conditioning=None,
    ):
    self.make_schedule(ddim_num_steps=S, ddim_eta=eta)
    
    size = batch_size, *shape

    samples = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )

    #print("samples", samples.numpy().sum(), samples.shape)
    #input("samples")
    return samples

  def ddim_sampling(
      self,
      cond,
      shape,
      x_T=None,
      ddim_use_original_steps=False,
      callback=None,
      timesteps=None,
      quantize_denoised=False,
      mask=None,
      x0=None,
      img_callback=None,
      log_every_t=100,
      temperature=1.,
      noise_dropout=0.,
      score_corrector=None,
      corrector_kwargs=None,
      unconditional_guidance_scale=1.,
      unconditional_conditioning=None,
    ):

    b = shape[0]

    #img = tf.random.normal(shape)
    img = np.load("/home/chaoji/work/genmo/diffusion/latent-diffusion/x.npy").transpose(0, 2, 3, 1)
    img = tf.constant(img)

    timesteps = self._ddim_steps
   
    intermediates = {"x_inter": [img], "pred_x0": [img]}

    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]
    #print(time_range) 
    for i, step in enumerate(time_range.tolist()):
      #print("i", i, "step", step)
      index = total_steps - i - 1
      ts = tf.fill((b,), step)

      outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
      
      img, pred_x0 = outs 
    return img

  def p_sample_ddim(
      self,
      x,
      c,
      t,
      index,
      repeat_noise=False,
      use_original_steps=False,
      quantize_denoised=False,
      temperature=1.,
      noise_dropout=0.,
      score_corrector=None,
      corrector_kwargs=None,
      unconditional_guidance_scale=1.,
      unconditional_conditioning=None
    ):
    #print("index", index, x.numpy().sum())
    b = x.shape[0]
   
    x_in = tf.concat([x, x], axis=0)
    t_in = tf.concat([t, t], axis=0) 
    c_in = tf.concat([unconditional_conditioning, c], axis=0)
    #print("in", x_in.shape, t_in.shape, c_in.shape)

    e_t_uncond, e_t = tf.split(self._model.apply_model(x_in, t_in, c_in), 2, axis=0)
    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

    alphas = self._model._alphas_cumprod if use_original_steps else self._ddim_alphas
    alphas_prev = self._model._alphas_cumprod_prev if use_original_steps else self._ddim_alphas_prev
    sqrt_one_minus_alphas = self._model._sqrt_one_minus_alphas_cumprod if use_original_steps else self._ddim_sqrt_one_minus_alphas
    sigmas = self._model._ddim_sigmas_for_original_num_steps if use_original_steps else self._ddim_sigmas
    #print("alphas", alphas, alphas.shape)
    #print("alphas_prev", alphas_prev, alphas_prev.shape)
    #print("sqrt_one_minus_alphas", sqrt_one_minus_alphas, sqrt_one_minus_alphas.shape)
    #print("sigmas", sigmas, sigmas.shape)


    a_t = tf.fill((b, 1, 1, 1), alphas[index])
    a_prev = tf.fill((b, 1, 1, 1), alphas_prev[index])
    sigma_t = tf.fill((b, 1, 1, 1), sigmas[index])
    sqrt_one_minus_at = tf.fill((b, 1, 1, 1), sqrt_one_minus_alphas[index])

    pred_x0 = (x - tf.cast(sqrt_one_minus_at, e_t.dtype) * e_t) / tf.sqrt(tf.cast(a_t, e_t.dtype))

    dir_xt = tf.sqrt(1. - tf.cast(a_prev, e_t.dtype) - tf.cast(sigma_t, e_t.dtype)**2) * e_t
    #noise = tf.cast(sigma_t, x.dtype) * tf.random.normal(x.shape,) * temperature
    eee = np.load(f"../latent-diffusion/noise{index}.npy").transpose(0, 2, 3, 1) #reshape(x.shape)
    #print("index", index, eee.shape, eee.sum())
    noise = tf.cast(sigma_t, x.dtype) * tf.constant(eee) * temperature

    x_prev = tf.sqrt(tf.cast(a_prev, pred_x0.dtype)) * pred_x0 + dir_xt + noise
    np.save(f"xx{index}", x_prev.numpy())
    #input("asdf")
    return x_prev, pred_x0


if __name__ == "__main__":

  from unet import UNet, timestep_embedding
  from transformer import TransformerModel
  from ddpm import DiffusionWrapper
  from ddpm import LatentDiffusion
  from autoencoder import Decoder, Encoder, AutoencoderKL

  transformer = TransformerModel(vocab_size=30522,
               encoder_stack_size=32,
               hidden_size=1280,
               num_heads=8,
               filter_size=1280*4,
               dropout_rate=0.1,)
  unet = UNet()
  autoencoder = AutoencoderKL()
  

  ckpt = tf.train.Checkpoint(transformer=transformer, unet=unet, autoencoder=autoencoder)
  ckpt.restore("txt2image-1").expect_partial()

 
  #token_ids = np.asarray([[  101,  1037,  7865,  6071,  2003,  2652,  2858,  1010,  3514,  2006,
  #                         10683,   102,     0,     0,     0,     0,     0,     0,     0,     0,
  #                             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
  #                             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
  #                             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
  #                             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
  #                             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
  #                             0,     0,     0,     0,     0,     0,     0]])
  #ti = np.asarray([[101, 102] + [0] * 75])


  max_length = 77
  tokenizer = BertTokenizerFast.from_pretrained("bert_model/")
  text = "a virus monster is playing guitar, oil on canvas" 
  token_ids = tokenizer(text, truncation=True, max_length=max_length, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"].numpy()
  ti = tokenizer("", truncation=True, max_length=max_length, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"].numpy()
  

  token_ids = tf.constant(np.vstack([np.tile(ti, [4, 1]), np.tile(token_ids, [4, 1])]))
  context = transformer(token_ids)


  diffusion_wrapper = DiffusionWrapper(model=unet, conditioning_key="crossattn")
  latent_diffusion = LatentDiffusion(linear_start=0.00085, linear_end=0.012, cond_stage_model=transformer, diffusion_wrapper=diffusion_wrapper)

  #sampler = DDIMSampler(model=latent_diffusion)
  #z = sampler.sample(conditioning=context[4:], unconditional_conditioning=context[:4], eta=1.0, S=50)
  sampler = DDIMSampler2(model=latent_diffusion, num_ddim_steps=50, eta=1.)
  z = sampler.ddim_p_sample_loop(shape=[4, 32, 32, 4], uncond=context[:4], cond=context[4:], unconditional_guidance_scale=5.)
  print("z", z.numpy().sum(), z.shape)

  scale_factor=0.18215
  inputs = 1 / scale_factor * z

  outputs = autoencoder.decode(inputs)


  print("outputs", outputs.numpy().sum(), outputs.shape)
  outputs = outputs.numpy()

  images = np.clip((outputs+1.0)/2.0, a_min=0.0, a_max=1.0)

