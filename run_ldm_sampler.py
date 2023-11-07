import yaml
import numpy as np
from transformers import BertTokenizerFast
import tensorflow as tf
from absl import app
from absl import flags

from unet import UNet
from transformer import TransformerModel
from model_runners import LatentDiffusionModelSampler
from autoencoder import AutoencoderKL, AutoencoderVQ


flags.DEFINE_string("config_path", None, "Path to yaml config file.")

FLAGS = flags.FLAGS

def tensor_to_image(inputs):
  inputs = inputs.numpy()
  for i in range(inputs.shape[0]):
    inputs[i] = (inputs[i] - inputs[i].min()) / (
        inputs[i].max() - inputs[i].min())
  inputs *= 255
  outputs = inputs.astype("uint8")
  return outputs


def get_token_ids(config):
  vocab_dir = config["ldm_sampling"]["vocab_dir"]
  prompt = config["ldm_sampling"]["text_prompt"]
  max_length = config["cond_stage_model"]["max_seq_len"]
  batch_size = config["ldm_sampling"]["latent_shape"][0]
  tokenizer = BertTokenizerFast.from_pretrained(vocab_dir) 

  cond_ids = tokenizer(prompt, truncation=True, max_length=max_length,
      return_length=True, return_overflowing_tokens=False, padding="max_length",
    return_tensors="pt")["input_ids"].numpy()
  uncond_ids = tokenizer("", truncation=True, max_length=max_length,
      return_length=True, return_overflowing_tokens=False, padding="max_length",
    return_tensors="pt")["input_ids"].numpy()

  token_ids = tf.constant(tf.concat(
      [tf.tile(uncond_ids, [batch_size, 1]),
       tf.tile(cond_ids, [batch_size, 1])], axis=0
  ))
  return token_ids


def main(_):
  with open(FLAGS.config_path) as f:
    config = yaml.safe_load(f)

  vocab_dir = config["ldm_sampling"]["vocab_dir"]
  tokenizer = BertTokenizerFast.from_pretrained(vocab_dir)

  kwargs = config["cond_stage_model"]
  transformer = TransformerModel(**kwargs) 
  kwargs = config["unet"]
  unet = UNet(**kwargs)

  if config["ldm_sampling"]["autoencoder_type"] == "kl":
    kwargs = config["autoencoder_kl"]
    autoencoder = AutoencoderKL(**kwargs)
  elif config["ldm_sampling"]["autoencoder_type"] == "vq": 
    kwargs = config["autoencoder_vq"]
    autoencoder = AutoencoderVQ(**kwargs)
  else:
    raise NotImplementedError("invalid autoencoder type.") 

  tf.train.Checkpoint(transformer=transformer).restore(
      config["pre_ckpt_paths"]["cond_stage_model"]).expect_partial()
  tf.train.Checkpoint(unet=unet).restore(
      config["pre_ckpt_paths"]["unet"]).expect_partial()
  tf.train.Checkpoint(autoencoder=autoencoder).restore(
      config["pre_ckpt_paths"]["autoencoder"]).expect_partial()

  kwargs = config["ldm"]
  sampler = LatentDiffusionModelSampler(
      unet=unet,
      autoencoder=autoencoder,
      cond_stage_model=transformer,
      **kwargs, 
  )

  token_ids = get_token_ids(config)
  shape = config["ldm_sampling"]["latent_shape"]
  guidance_scale = config["ldm_sampling"]["guidance_scale"]

  if config["ldm_sampling"]["sample_save_progress"]:
    sample_prog, pred_x0_prog = sampler.ddim_p_sample_loop_progressive(
        token_ids, shape, guidance_scale)
    print("[INFO] Save progressive sample images to 'sample_prog.npy'...")
    np.save("sample_prog.npy", tensor_to_image(sample_prog))
    print("[INFO] Save progressive estimated `x0` to 'pred_x0_prog.npy'...")
    np.save("pred_x0_prog.npy", tensor_to_image(pred_x0_prog))
  else:
    images = sampler.ddim_p_sample_loop(token_ids, shape, guidance_scale)
    print("[INFO] Save generated images to 'images.npy'...")
    np.save("images.npy", tensor_to_image(images))

if __name__ == "__main__":
  flags.mark_flag_as_required("config_path")
  app.run(main) 
