# Tensorflow2 Implementation of Latent Diffusion Model (LDM)

This is a TensorFlow2 implementation (tested on version 2.13.0) of **Latent Diffusion Model** ([paper](https://arxiv.org/abs/2112.10752) and [official PyTorch implementation](https://github.com/CompVis/latent-diffusion))


## Quick start

Clone this repo
```bash
git clone git@github.com:chao-ji/ldm_tf2.git
```

And install the required python libraries.

```bash
pip install -r requirements.txt
```
Note that the only thing needed (other than tensorflow) is the library [transformers](https://pypi.org/project/transformers/) by [HuggingFace](https://huggingface.co/), which is used to tokenize text prompts.


### Convert pretrained model to TF2 format

The [official PyTorch implementation](https://github.com/CompVis/latent-diffusion) comes with [pretrained txt2img model](https://ommer-lab.com/files/latent-diffusion/text2img.zip). 

First you need to download and unzip it, and convert it into TF2 format:

```batch
python convert_ckpt_pytorch_to_tf2.py --pytorch_ckpt_path model.ckpt
```

You will get three TF2 checkpoints `transformer-1.data-00000-of-00001`, `transformer-1.index`, `unet-1.data-00000-of-00001`, `unet-1.index`, `autoencoder-1.data-00000-of-00001`, `autoencoder-1.index`.


### Sampling
Now is the time to generate images with text prompt!

Note that all the configurations can be set in [all_in_one_config.yaml](all_in_one_config.yaml)

First make sure that all the checkpoint files for `transformer-1`, `unet-1`, and `autoencoder-1` are correctly set under `pre_ckpt_paths`.

The specific parameters for controling the sampling process can be found under `ldm_sampling`:

```
ldm_sampling:
  # scale for classifier free guidance, larger values indicate less diversity but more fidelity
  guidance_scale: 5.

  # shape of the latent variable in [batch_size, height, width, latent_channels]
  # the shape of the final generated image is `height * 8` by `width * 8`  
  # `batch_size` in `latent_shape` is just the number of images to be generated in one batch
  latent_shape: [4, 32, 32, 4]

  # whetehr to save intermediate results for estimated `x_{t-1}` and `x_{0}`
  sample_save_progress: false

  text_prompt: "a virus monster is playing guitar, oil on canvas"

  vocab_dir: bert_model # directory where the `vocab.txt` file is located

  autoencoder_type: "kl" # ["kl", "vq"]
```

To sample, just run
```python
python run_ldm_sampler.py --config_path all_in_one_config.yaml
```

The generated images are in the form of numpy arrays (with shape `[N,H,W,C]`) and are saved to `.npy` files.

## Training
Training of latent diffusion model is divided into two stages:
* Train an autoencoder, which has an **encoder** that encodes image in pixel space to latent space, and a **decoder** that decodes latent variable back to image space.
* Train diffusion model (conditioned on text) on the encoder of a pretrained autoencoder.

### Convert raw data to TFRecord files

[run_tfrecord_converters.py](run_tfrecord_converters.py) is an example that shows how to generate tfrecord files from raw data. Functions `convert_images_to_tfrecord` and `convert_coco_captions_to_tfrecord` in [dataset.py](dataset.py) handles the conversion of "image only" dataset and "image-caption pair" dataset, which are used to train autoencoders and latent diffusion models, respectively. Modify the parameters in [run_tfrecord_converters.py](run_tfrecord_converters.py) for your own case.

During training, the serialized images (and captions) in TFRecord files will be loaed and deserialized, and then padded and resized into tensors of shape `[batch_size, image_height, image_width, 3]` (and `[batch_size, max_seq_length]` for captions)
 
### Autoencoders
Set the parameters in [all_in_one_config.yaml](all_in_one_config.yaml) controling autoencoder training :

```yaml
autoencoder_training:
  # directory where pre-converted "*.tfrecord" files are located
  root_path: /path/to/tfrecord/images 
  params:
    batch_size: 4
    image_size: 256
    keys: ["image"]   # ["image"] for training autoencoders, and ["image", "caption"] for txt2img latent diffusion model
  autoencoder_type: "vq" # ["kl", "vq"]
  ckpt_path: "aevq" # ["aekl", "aevq"], path to the ckpt in which a trained model will be saved
  num_iterations: 500000  # num of training iterations
```

Then run
```
python run_autoencoder_trainer.py --config_path all_in_one_config.yaml
```

### Latent Diffusion Model

Parameters in [all_in_one_config.yaml](all_in_one_config.yaml) controling ldm training:

```yaml
ldm_training:
  root_path: /path/to/tfrecord/images_captions
  params:
    batch_size: 1
    image_size: 256
    flip: false
    keys: ["image", "caption"]  # ["image"] for training autoencoders, and ["image", "caption"] for txt2img latent diffusion model
  autoencoder_type: "kl" # ["kl", "vq"]
  ckpt_path: "ldm"
  num_iterations: 500000
  train_cond_model: false
```


## Sample images


