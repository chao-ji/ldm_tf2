import torch
import tensorflow as tf
import numpy as np

from unet import UNet, timestep_embedding
from transformer import TransformerModel
from autoencoder import Decoder, Encoder, AutoencoderKL


def get_state_dict(filename):
  sd = torch.load(filename)["state_dict"]
  for k in sd.keys():
    sd[k] = sd[k].numpy()
  return sd


def get_transformer_weights(sd):
  weights = []

  for i in range(32):
    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".1.to_q.weight"]
    w = w.T.reshape(-1, 8, 64)
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".1.to_k.weight"]
    w = w.T.reshape(-1, 8, 64)
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".1.to_v.weight"]
    w = w.T.reshape(-1, 8, 64)
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".1.to_out.weight"]
    w = w.T.reshape(8, 64, -1)
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".1.to_out.bias"]
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".0.weight"]
    weights.append(w)
    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2) + ".0.bias"]
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".1.net.0.0.weight"]
    weights.append(w.T)
    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".1.net.0.0.bias"]
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".1.net.2.weight"]
    weights.append(w.T)
    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".1.net.2.bias"]
    weights.append(w)

    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".0.weight"]
    weights.append(w)
    w = sd["cond_stage_model.transformer.attn_layers.layers." + str(i*2+1) + ".0.bias"]
    weights.append(w)

  weights.append(sd["cond_stage_model.transformer.norm.weight"])
  weights.append(sd["cond_stage_model.transformer.norm.bias"])
  weights.append(sd["cond_stage_model.transformer.token_emb.weight"])
  weights.append(sd["cond_stage_model.transformer.pos_emb.emb.weight"]) 
  return weights


def get_unet_weights(sd):
  shape_dict = {1: 1, 2: 1, 4: 2, 5: 2, 7: 4, 8: 4}
  shape_dict1 = {3: 4, 4: 4, 5: 4, 6: 2, 7: 2, 8: 2, 9: 1, 10: 1, 11: 1}

  weights = []
  weights.append(sd["model.diffusion_model.input_blocks.0.0.weight"].transpose(2, 3, 1, 0))
  weights.append(sd["model.diffusion_model.input_blocks.0.0.bias"])
  weights.append(sd["model.diffusion_model.time_embed.0.weight"].T)
  weights.append(sd["model.diffusion_model.time_embed.0.bias"])
  weights.append(sd["model.diffusion_model.time_embed.2.weight"].T)
  weights.append(sd["model.diffusion_model.time_embed.2.bias"])

  weights1 = []
  for i in range(1, 12):
    if i in (3, 6, 9):
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.op.weight"].transpose(2, 3, 1, 0))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.op.bias"])
      continue

    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.in_layers.0.weight"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.in_layers.0.bias"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.in_layers.2.weight"].transpose(2, 3, 1, 0))
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.in_layers.2.bias"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.emb_layers.1.weight"].T)
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.emb_layers.1.bias"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.out_layers.0.weight"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.out_layers.0.bias"])
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.out_layers.3.weight"].transpose(2, 3, 1, 0))
    weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.out_layers.3.bias"])

    if i in (4, 7):
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.skip_connection.weight"].squeeze().T)
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.0.skip_connection.bias"])

    if i in (1, 2, 4, 5, 7, 8):

      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.proj_in.weight"].squeeze().T)
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.proj_in.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn1.to_q.weight"].T.reshape(320 * shape_dict[i], 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn1.to_k.weight"].T.reshape(320 * shape_dict[i], 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn1.to_v.weight"].T.reshape(320 * shape_dict[i], 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn1.to_out.0.weight"].T.reshape(8, 40 * shape_dict[i], 320 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn1.to_out.0.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn2.to_q.weight"].T.reshape(320 * shape_dict[i], 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn2.to_k.weight"].T.reshape(1280, 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn2.to_v.weight"].T.reshape(1280, 8, 40 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn2.to_out.0.weight"].T.reshape(8, 40 * shape_dict[i], 320 * shape_dict[i]))
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn2.to_out.0.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.ff.net.0.proj.weight"].T)
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.ff.net.0.proj.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.ff.net.2.weight"].T)
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.ff.net.2.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm1.weight"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm1.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm2.weight"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm2.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm3.weight"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.transformer_blocks.0.norm3.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.proj_out.weight"].squeeze().T)
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.proj_out.bias"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.norm.weight"])
      weights1.append(sd[f"model.diffusion_model.input_blocks.{i}.1.norm.bias"])

  weights2 = []
  for i in range(3):
    if i in (0, 2):
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.in_layers.0.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.in_layers.0.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.in_layers.2.weight"].transpose(2, 3, 1, 0))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.in_layers.2.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.emb_layers.1.weight"].T)
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.emb_layers.1.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.out_layers.0.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.out_layers.0.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.out_layers.3.weight"].transpose(2, 3, 1, 0))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.out_layers.3.bias"])
    else:
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.proj_in.weight"].squeeze().T)
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.proj_in.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_q.weight"].T.reshape(320 * 4, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_k.weight"].T.reshape(320 * 4, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_v.weight"].T.reshape(320 * 4, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_out.0.weight"].T.reshape(8, 40 * 4, 320 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_out.0.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_q.weight"].T.reshape(320 * 4, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_k.weight"].T.reshape(1280, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_v.weight"].T.reshape(1280, 8, 40 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_out.0.weight"].T.reshape(8, 40 * 4, 320 * 4))
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_out.0.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.ff.net.0.proj.weight"].T)
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.ff.net.0.proj.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.ff.net.2.weight"].T)
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.ff.net.2.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm1.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm1.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm2.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm2.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm3.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm3.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.proj_out.weight"].squeeze().T)
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.proj_out.bias"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.norm.weight"])
      weights2.append(sd[f"model.diffusion_model.middle_block.{i}.norm.bias"])

  weights3 = []
  for i in range(12):
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.in_layers.0.weight"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.in_layers.0.bias"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.in_layers.2.weight"].transpose(2, 3, 1, 0))
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.in_layers.2.bias"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.emb_layers.1.weight"].T)
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.emb_layers.1.bias"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.out_layers.0.weight"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.out_layers.0.bias"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.out_layers.3.weight"].transpose(2, 3, 1, 0))
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.out_layers.3.bias"])
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.skip_connection.weight"].squeeze().T)
    weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.0.skip_connection.bias"])
    if i in (3, 4, 5, 6, 7, 8, 9, 10, 11):
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.proj_in.weight"].squeeze().T)
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.proj_in.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn1.to_q.weight"].T.reshape(320 * shape_dict1[i], 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn1.to_k.weight"].T.reshape(320 * shape_dict1[i], 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn1.to_v.weight"].T.reshape(320 * shape_dict1[i], 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn1.to_out.0.weight"].T.reshape(8, 40 * shape_dict1[i], 320 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn1.to_out.0.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn2.to_q.weight"].T.reshape(320 * shape_dict1[i], 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn2.to_k.weight"].T.reshape(1280, 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn2.to_v.weight"].T.reshape(1280, 8, 40 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn2.to_out.0.weight"].T.reshape(8, 40 * shape_dict1[i], 320 * shape_dict1[i]))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn2.to_out.0.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.ff.net.0.proj.weight"].T)
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.ff.net.0.proj.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.ff.net.2.weight"].T)
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.ff.net.2.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm1.weight"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm1.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm2.weight"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm2.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm3.weight"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.transformer_blocks.0.norm3.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.proj_out.weight"].squeeze().T)
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.proj_out.bias"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.norm.weight"])
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.1.norm.bias"])

    if i in (2, 5, 8):
      if i == 2:
        j = 1
      else:
        j = 2
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.{j}.conv.weight"].transpose(2, 3, 1, 0))
      weights3.append(sd[f"model.diffusion_model.output_blocks.{i}.{j}.conv.bias"])

  weights = weights + weights1 + weights2 + weights3
  weights.append(sd[f"model.diffusion_model.out.0.weight"])
  weights.append(sd[f"model.diffusion_model.out.0.bias"])
  weights.append(sd[f"model.diffusion_model.out.2.weight"].transpose(2, 3, 1, 0))
  weights.append(sd[f"model.diffusion_model.out.2.bias"])
  return weights


def get_decoder_weights(sd):
  weights = []

  def get_block(weights, which):
    weights.append(sd[f"first_stage_model.decoder.{which}.norm1.weight"])
    weights.append(sd[f"first_stage_model.decoder.{which}.norm1.bias"])
    weights.append(sd[f"first_stage_model.decoder.{which}.conv1.weight"].transpose(2, 3, 1, 0))
    weights.append(sd[f"first_stage_model.decoder.{which}.conv1.bias"])
    weights.append(sd[f"first_stage_model.decoder.{which}.norm2.weight"])
    weights.append(sd[f"first_stage_model.decoder.{which}.norm2.bias"])
    weights.append(sd[f"first_stage_model.decoder.{which}.conv2.weight"].transpose(2, 3, 1, 0))
    weights.append(sd[f"first_stage_model.decoder.{which}.conv2.bias"])
    if which in ("up.0.block.0", "up.1.block.0"):
      weights.append(sd[f"first_stage_model.decoder.{which}.nin_shortcut.weight"].squeeze().T)
      weights.append(sd[f"first_stage_model.decoder.{which}.nin_shortcut.bias"])
    return weights

  def get_attn(weights, which):
    weights.append(sd[f"first_stage_model.decoder.{which}.norm.weight"])
    weights.append(sd[f"first_stage_model.decoder.{which}.norm.bias"])
    weights.append(sd[f"first_stage_model.decoder.{which}.q.weight"].squeeze().T)
    weights.append(sd[f"first_stage_model.decoder.{which}.q.bias"])
    weights.append(sd[f"first_stage_model.decoder.{which}.k.weight"].squeeze().T)
    weights.append(sd[f"first_stage_model.decoder.{which}.k.bias"])
    weights.append(sd[f"first_stage_model.decoder.{which}.v.weight"].squeeze().T)
    weights.append(sd[f"first_stage_model.decoder.{which}.v.bias"])
    weights.append(sd[f"first_stage_model.decoder.{which}.proj_out.weight"].squeeze().T)
    weights.append(sd[f"first_stage_model.decoder.{which}.proj_out.bias"])
    return weights

  def get_upsample(weights, i):
    weights.append(sd[f"first_stage_model.decoder.up.{i}.upsample.conv.weight"].transpose(2, 3, 1, 0))
    weights.append(sd[f"first_stage_model.decoder.up.{i}.upsample.conv.bias"])
    return weights

  weights.append(sd[f"first_stage_model.decoder.conv_in.weight"].transpose(2, 3, 1, 0))
  weights.append(sd[f"first_stage_model.decoder.conv_in.bias"])
 
  weights = get_block(weights, "mid.block_1")
  weights = get_attn(weights, "mid.attn_1")
  weights = get_block(weights, "mid.block_2")

  weights = get_block(weights, "up.3.block.0")
  weights = get_block(weights, "up.3.block.1")
  weights = get_block(weights, "up.3.block.2")

  weights = get_upsample(weights, 3)

  weights = get_block(weights, "up.2.block.0")
  weights = get_block(weights, "up.2.block.1")
  weights = get_block(weights, "up.2.block.2")

  weights = get_upsample(weights, 2)

  weights = get_block(weights, "up.1.block.0")
  weights = get_block(weights, "up.1.block.1")
  weights = get_block(weights, "up.1.block.2")

  weights = get_upsample(weights, 1)

  weights = get_block(weights, "up.0.block.0")
  weights = get_block(weights, "up.0.block.1")
  weights = get_block(weights, "up.0.block.2")

  weights.append(sd[f"first_stage_model.decoder.norm_out.weight"])
  weights.append(sd[f"first_stage_model.decoder.norm_out.bias"])
  weights.append(sd[f"first_stage_model.decoder.conv_out.weight"].transpose(2, 3, 1, 0))
  weights.append(sd[f"first_stage_model.decoder.conv_out.bias"])

  return weights


def get_encoder_weights(sd):
  weights = [] 

  weights.append(sd[f"first_stage_model.encoder.conv_in.weight"].transpose(2, 3, 1, 0))
  weights.append(sd[f"first_stage_model.encoder.conv_in.bias"])

  def get_block(weights, which):
    weights.append(sd[f"first_stage_model.encoder.{which}.norm1.weight"])
    weights.append(sd[f"first_stage_model.encoder.{which}.norm1.bias"])
    weights.append(sd[f"first_stage_model.encoder.{which}.conv1.weight"].transpose(2, 3, 1, 0))
    weights.append(sd[f"first_stage_model.encoder.{which}.conv1.bias"])
    weights.append(sd[f"first_stage_model.encoder.{which}.norm2.weight"])
    weights.append(sd[f"first_stage_model.encoder.{which}.norm2.bias"])
    weights.append(sd[f"first_stage_model.encoder.{which}.conv2.weight"].transpose(2, 3, 1, 0))
    weights.append(sd[f"first_stage_model.encoder.{which}.conv2.bias"])
    if which in ("down.1.block.0", "down.2.block.0"):
      weights.append(sd[f"first_stage_model.encoder.{which}.nin_shortcut.weight"].squeeze().T)
      weights.append(sd[f"first_stage_model.encoder.{which}.nin_shortcut.bias"])
    return weights

  def get_attn(weights, which):
    weights.append(sd[f"first_stage_model.encoder.{which}.norm.weight"])
    weights.append(sd[f"first_stage_model.encoder.{which}.norm.bias"])
    weights.append(sd[f"first_stage_model.encoder.{which}.q.weight"].squeeze().T)
    weights.append(sd[f"first_stage_model.encoder.{which}.q.bias"])
    weights.append(sd[f"first_stage_model.encoder.{which}.k.weight"].squeeze().T)
    weights.append(sd[f"first_stage_model.encoder.{which}.k.bias"])
    weights.append(sd[f"first_stage_model.encoder.{which}.v.weight"].squeeze().T)
    weights.append(sd[f"first_stage_model.encoder.{which}.v.bias"])
    weights.append(sd[f"first_stage_model.encoder.{which}.proj_out.weight"].squeeze().T)
    weights.append(sd[f"first_stage_model.encoder.{which}.proj_out.bias"])
    return weights

  def get_downsample(weights, i):
    weights.append(sd[f"first_stage_model.encoder.down.{i}.downsample.conv.weight"].transpose(2, 3, 1, 0))
    weights.append(sd[f"first_stage_model.encoder.down.{i}.downsample.conv.bias"])
    return weights

  weights = get_block(weights, "down.0.block.0")
  weights = get_block(weights, "down.0.block.1")
  
  weights = get_downsample(weights, 0)

  weights = get_block(weights, "down.1.block.0")
  weights = get_block(weights, "down.1.block.1")

  weights = get_downsample(weights, 1)

  weights = get_block(weights, "down.2.block.0")
  weights = get_block(weights, "down.2.block.1")

  weights = get_downsample(weights, 2)
 
  weights = get_block(weights, "down.3.block.0")
  weights = get_block(weights, "down.3.block.1")

  weights = get_block(weights, "mid.block_1")
  weights = get_attn(weights, "mid.attn_1")
  weights = get_block(weights, "mid.block_2")

  weights.append(sd[f"first_stage_model.encoder.norm_out.weight"])
  weights.append(sd[f"first_stage_model.encoder.norm_out.bias"])
  weights.append(sd[f"first_stage_model.encoder.conv_out.weight"].transpose(2, 3, 1, 0))
  weights.append(sd[f"first_stage_model.encoder.conv_out.bias"])

  return weights


def save_checkpoint(sd):
  vocab_size = 30522
  transformer = TransformerModel(vocab_size,
               encoder_stack_size=32,
               hidden_size=1280,
               num_heads=8,
               filter_size=1280*4,
               dropout_rate=0.1,)

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
  transformer(token_ids)
  weights = get_transformer_weights(sd)
  transformer.set_weights(weights)

  batch_size = 8
  unet = UNet()
  x = np.random.uniform(-1, 1, [batch_size // 2, 32, 32, 4]).astype("float32")
  x = np.concatenate([x, x], axis=0)
  t_emb = tf.constant([981] * batch_size)
  context = tf.constant(np.random.uniform(-1, 1, (batch_size, 77, 1280)).astype("float32"))
  unet(x, t_emb, context)
  weights = get_unet_weights(sd)
  unet.set_weights(weights)


  autoencoder = AutoencoderKL(z_channels=4)
  images = tf.constant(np.random.uniform(-1, 1, (4, 256, 256, 3)).astype("float32"))
  recon, _ = autoencoder(images)

  autoencoder._encoder.set_weights(get_encoder_weights(sd))
  autoencoder._quant_conv.set_weights([
      sd["first_stage_model.quant_conv.weight"].squeeze().T,
      sd["first_stage_model.quant_conv.bias"],]
  )
  autoencoder._post_quant_conv.set_weights([
      sd["first_stage_model.post_quant_conv.weight"].squeeze().T,
      sd["first_stage_model.post_quant_conv.bias"],]
  )
  autoencoder._decoder.set_weights(
      get_decoder_weights(sd)
  )

  ckpt = tf.train.Checkpoint(transformer=transformer, unet=unet, autoencoder=autoencoder)
  ckpt.save("txt2image")


if __name__ == "__main__":
  sd = get_state_dict("/home/chaoji/work/genmo/diffusion/latent-diffusion/models/ldm/text2img-large/model.ckpt")
  save_checkpoint(sd)

