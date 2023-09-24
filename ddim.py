import numpy as np
import tensorflow as tf
from ddpm import DDPM


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
    b = x.shape[0]
   
    x_in = tf.concat([x, x], axis=0)
    t_in = tf.concat([t, t], axis=0) 
    c_in = tf.concat([unconditional_conditioning, c], axis=0)
    #print("in", x_in.shape, t_in.shape, c_in.shape)

    e_t_uncond, e_t = tf.split(self._model.apply_model(x_in, t_in, c_in), 2, axis=0)
    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
    #print("e_t", e_t.numpy().sum(), e_t.shape)

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
    #print("pred_x0", pred_x0.numpy().sum(), pred_x0.shape)

    dir_xt = tf.sqrt(1. - tf.cast(a_prev, e_t.dtype) - tf.cast(sigma_t, e_t.dtype)**2) * e_t
    noise = tf.cast(sigma_t, x.dtype) * tf.random.normal(x.shape,) * temperature

    x_prev = tf.sqrt(tf.cast(a_prev, pred_x0.dtype)) * pred_x0 + dir_xt + noise
    #print("x_prev", x_prev.numpy().sum(), x_prev.shape)
    #input("asdf")
    return x_prev, pred_x0


if __name__ == "__main__":

  from unet import UNet, timestep_embedding
  from transformer import TransformerModel
  

  vocab_size = 30522
  transformer = TransformerModel(vocab_size,
               encoder_stack_size=32,
               hidden_size=1280,
               num_heads=8,
               filter_size=1280*4,
               dropout_rate=0.1,)
  #t_ckpt = tf.train.Checkpoint(transformer=transformer)
  #t_ckpt.restore("transformer-1")

  token_ids = np.asarray([[  101,  1037,  7865,  6071,  2003,  2652,  2858,  1010,  3514,  2006,
                           10683,   102,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                               0,     0,     0,     0,     0,     0,     0]])
  ti = np.asarray([[101, 102] + [0] * 75])
  token_ids = tf.constant(np.vstack([np.tile(ti, [4, 1]), np.tile(token_ids, [4, 1])]))


  print("\n" * 10)






  







  unet = UNet()
  x = np.load("/home/chaoji/work/genmo/diffusion/latent-diffusion/x.npy").transpose(0, 2, 3, 1)
  x = np.concatenate([x, x], axis=0)
  t_emb = tf.constant([981] * 8)


  #u_ckpt = tf.train.Checkpoint(unet=unet)
  #u_ckpt.restore("unet-1")

  print("\n" * 5)








  from autoencoder import Decoder
  post_quant_conv = tf.keras.layers.Dense(4)
  decoder = Decoder(
      channels=128,
      out_channels=3,
      num_blocks=2,
      multipliers=(1, 2, 4, 4),
      resample_with_conv=True,
      attention_resolutions=(),
      give_pre_end=False,
  ) 


  ckpt = tf.train.Checkpoint(transformer=transformer, unet=unet, post_quant_conv=post_quant_conv, decoder=decoder)
  ckpt.restore("txt2image-1")

  #p_ckpt = tf.train.Checkpoint(post_quant_conv=post_quant_conv)
  #d_ckpt = tf.train.Checkpoint(decoder=decoder)

  #p_ckpt.restore("post_quant_conv-1")
  #d_ckpt.restore("decoder-1")

  #post_quant_conv(z)
  #decoder(z)






















  context = transformer(token_ids, None)

  outputs = unet(x, t_emb, context)



  from ddpm import DiffusionWrapper
  from ddpm import LatentDiffusion


  diffusion_wrapper = DiffusionWrapper(model=unet, conditioning_key="crossattn")

  latent_diffusion = LatentDiffusion(linear_start=0.00085, linear_end=0.012, cond_stage_model=transformer, diffusion_wrapper=diffusion_wrapper)

  sampler = DDIMSampler(model=latent_diffusion)
  #print(context[:4].numpy().sum(), context[4:].numpy().sum())
  z = sampler.sample(conditioning=context[4:], unconditional_conditioning=context[:4])
  print("z", z.numpy().sum(), z.shape)


  scale_factor=0.18215
  inputs = 1 / scale_factor * z


  inputs = post_quant_conv(inputs)

  outputs = decoder(inputs, training=False)

  print("outputs", outputs.numpy().sum(), outputs.shape)
  outputs = outputs.numpy()

  images = np.clip((outputs+1.0)/2.0, a_min=0.0, a_max=1.0)




