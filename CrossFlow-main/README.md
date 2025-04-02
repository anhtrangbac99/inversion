# CrossFlow
This is a PyTorch-based reimplementation of CrossFlow, as proposed in 

>  Flowing from Words to Pixels: A Framework for Cross-Modality Evolution
>
>  CVPR 2025
>
>  [Qihao Liu](https://qihao067.github.io/) | [Xi Yin](https://xiyinmsu.github.io/) | [Alan Yuille](https://cogsci.jhu.edu/directory/alan-yuille/) | [Andrew Brown](https://www.robots.ox.ac.uk/~abrown/) | [Mannat Singh](https://ai.meta.com/people/1287460658859448/mannat-singh/)
>
>  [[project page](https://cross-flow.github.io/)] | [[huggingface demo](https://huggingface.co/spaces/QHL067/CrossFlow)] | [[paper](https://arxiv.org/pdf/2412.15213)] | [[arxiv](https://arxiv.org/abs/2412.15213)]

![teaser](https://github.com/qihao067/CrossFlow/blob/main/imgs/teaser.jpg)

This repository provides a **PyTorch-based reimplementation** of CrossFlow for the text-to-image generation task, with the following differences compared to the original paper:

- **Model Architecture**: The original paper utilizes DiMR as the model architecture. In contrast, this codebase supports training and inference with both [DiT](https://github.com/facebookresearch/DiT) (ICCV 2023, a widely adopted architecture) and [DiMR](https://github.com/qihao067/DiMR) (NeurIPS 2024, a state-of-the-art architecture).
- **Dataset**: The original model was trained on proprietary 350M dataset. In this implementation, the models are trained on open-source datasets, including [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) and [JourneyDB](https://journeydb.github.io/) (4M).
- **LLMs**: The original 1B model only supports CLIP as the language model, whereas this implementation includes 1B models with CLIP and T5-XXL.

______

## TODO

- [x] ~~Release inference code and 512px CLIP DiMR-based model.~~
- [x] ~~Release training code and a detailed training tutorial (ETA: Dec 20).~~
- [x] ~~Release inference code for linear interpolation and arithmetic.~~
- [x] ~~Release all pretrained checkpoints, including:   (ETA: Dec 23)~~
- [x] ~~Update pretrained checkpoints  (ETA: Dec 28)~~
- [x] ~~Provide a demo via Hugging Face Space and Colab.~~

______

## Setup

- ### Environment

  The code has been tested with PyTorch 2.1.2 and Cuda 12.1.

  An example of installation commands is provided as follows:

  ```
  git clone git@github.com:qihao067/CrossFlow.git
  cd CrossFlow
  
  pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
  pip3 install -U --pre triton
  pip3 install -r requirements.txt
  ```

- ### Model Preparation

  To train or test the model, you will also need to download the VAE model from [Stable Diffusion](https://github.com/CompVis/stable-diffusion), and the reference statistics for zero-shot FID on the [MSCOCO](https://cocodataset.org/#home) validation set. For your convenience, you can directly download all the models from [here](https://huggingface.co/QHL067/CrossFlow/blob/main/assets.tar).

------

## Pretrained Models

| Architecture | Resolution | LM     | Download                                                     | Details                                                      |
| :----------- | ---------- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DiMR         | 256x256    | CLIP   | [[t2i_256px_clip_dimr.pth](https://huggingface.co/QHL067/CrossFlow/blob/main/pretrained_models/t2i_256px_clip_dimr.pth)] | Train from scratch on LIAON-400M for 1 epoch,  then fine-tune on JourneyDB for 10 epochs. |
| DiMR         | 256x256    | T5-XXL | [[t2i_256px_t5_dimr.pth](https://huggingface.co/QHL067/CrossFlow/blob/main/pretrained_models/t2i_256px_t5_dimr.pth)] | Initialize with [t2i_256px_clip_dimr.pth] and fine-tune on JourneyDB for 10 epochs. |
| DiMR         | 512x512    | CLIP   | [[t2i_512px_clip_dimr.pth](https://huggingface.co/QHL067/CrossFlow/blob/main/pretrained_models/t2i_512px_clip_dimr.pth)] | Initialize with [t2i_256px_clip_dimr.pth] and fine-tune on JourneyDB for 10 epochs. (Model with the best T-I alignment*) |
| DiMR         | 512x512    | T5-XXL | [[t2i_512px_t5_dimr.pth](https://huggingface.co/QHL067/CrossFlow/blob/main/pretrained_models/t2i_512px_t5_dimr.pth)] | Initialize with [t2i_512px_clip_dimr.pth] and fine-tune on JourneyDB for 10 epochs. |
| DiT          | 512x512    | T5-XXL | [[t2i_512px_t5_dit.pth](https://huggingface.co/QHL067/CrossFlow/blob/main/pretrained_models/t2i_512px_t5_dit.pth)] | Initialize with [t2i_512px_clip_dimr.pth] and fine-tune on JourneyDB for 10 epochs. |

*To save training time, all T5-XXL-based models are initialized with a CLIP-based model and fine-tuned on JourneyDB (4M) for ten epochs. As a result, these models may occasionally exhibit very minor text-image misalignments, which are not observed in the original paper's T5 models since they are trained from scratch.

------

## Sampling

- ### T2I generation

  You can sample from the pre-trained CrossFLow model with the [`demo_t2i.py`](https://github.com/qihao067/CrossFlow/blob/main/demo_t2i.py). Before running the script, download the appropriate checkpoint and configure hyperparameters such as the classifier-free guidance scale, random seed, and mini-batch size in the corresponding configuration files.

  To accelerate the sampling process, the script supports multi-GPU sampling. For example, to sample from the 512px CLIP DiMR-based CrossFlow model with `N` GPUs, you can use the following command. It generates `N x mini-batch size` images each time:

  ```
  # if only sample with one GPU:
  # accelerate launch --num_processes 1 --mixed_precision bf16 demo_t2i.py \
  
  accelerate launch --multi_gpu --num_processes N --mixed_precision bf16 demo_t2i.py \
            --config=configs/t2i_512px_clip_dimr.py \
            --nnet_path=path/to/t2i_512px_clip_dimr.pth \
            --img_save_path=temp_saved_images \
            --prompt='your prompt' \
  ```

- ### Linear Interpolation in Latent Space

  Our model provides visually smooth interpolations in the latent space. By using the [`demo_t2i_arith.py`](https://github.com/qihao067/CrossFlow/blob/main/demo_t2i_arith.py) script, images can be generated through linear interpolation between two input prompts using the following command:

  ```
  accelerate launch --num_processes 1 --mixed_precision bf16 demo_t2i_arith.py \
            --config=configs/t2i_512px_clip_dimr.py \
            --nnet_path=path/to/t2i_512px_clip_dimr.pth \
            --img_save_path=temp_saved_images \
            --test_type=interpolation \
            --prompt_1='A dog cooking dinner in the kitchen' \
            --prompt_2='An orange cat wearing sunglasses on a ship' \
            --num_of_interpolation=40 \
            --save_gpu_memory \
  
  ```

  This script supports sampling on a single GPU only. For linear interpolation, you need to adjust the `num_of_interpolation` parameter, which controls the number of interpolated images generated. The script requires a minimum of `5` images, but we recommend setting it to `40` for smoother interpolations. Additionally, you can enable the `save_gpu_memory` option to optimize GPU VRAM usage, though this will require extra time.

  Finally, the command will generate `num_of_interpolation` images in the specified `img_save_path`. Using the provided random seed (`1234`), the resulting images will appear as follows

  ![teaser](https://github.com/qihao067/CrossFlow/blob/main/imgs/linear_output.gif)

- ### Arithmetic Operations in Latent Space

  Our model supports arithmetic operations in the text latent space. Using the Text Variational Encoder, we first encode the input text into the latent space. Arithmetic operations are then applied within this latent space, and the resulting latent representation is utilized to generate the corresponding image. An example can be demonstrated using the following command:

  ```
  accelerate launch --num_processes 1 --mixed_precision bf16 demo_t2i_arith.py \
            --config=configs/t2i_512px_clip_dimr.py \
            --nnet_path=path/to/t2i_512px_clip_dimr.pth \
            --img_save_path=temp_saved_images \
            --test_type=arithmetic \
            --prompt_ori='A corgi wearing a red hat in the park' \
            --prompt_a='book' \
            --prompt_s='hat' \
  ```

  The images generated in the folder `img_save_path` include images of the the input prompts, followed by the resulting image after the arithmetic operations (`prompt_ori + prompt_a - prompt_s` ) .

  <p align="center" style="display: flex; align-items: center;">
    <img src="https://github.com/qihao067/CrossFlow/blob/main/imgs/0.png" alt="Figure 1" width="200"/>
    <img src="https://github.com/qihao067/CrossFlow/blob/main/imgs/1.png" alt="Figure 2" width="200"/>
    <img src="https://github.com/qihao067/CrossFlow/blob/main/imgs/2.png" alt="Figure 3" width="200"/>
    <img src="https://github.com/qihao067/CrossFlow/blob/main/imgs/3.png" alt="Figure 4" width="200"/>
  </p>
  
  

We also support single arithmetic operation. You can perform addition by providing only `prompt_ori` and `prompt_a`, or subtraction by providing only `prompt_ori` and `prompt_s`.

------

## Training CrossFlow for T2I

- ### Prepare training data

  To train the CrossFlow model, you need a dataset consisting of image-text pairs. We provide a demo dataset ([download here](https://huggingface.co/QHL067/CrossFlow/blob/main/JourneyDB_demo.tar)) containing 100 images sourced from [JourneyDB](https://journeydb.github.io/). The dataset includes an image folder and a `.jsonl` file that specifies the image paths and their corresponding captions.

  To accelerate the training process, you can cache the image latents (from a VAE) and text embeddings (from a language model such as CLIP or T5-XXL) beforehand. We offer preprocessing scripts to simplify this step. 

  Specifically, you can use the [`scripts/extract_train_feature.py`](https://github.com/qihao067/CrossFlow/blob/main/scripts/extract_train_feature.py) script to extract and save these features. Before running the script, ensure that you update the dataset paths (`json_path` and `root_path`) and set an appropriate batch size (`bz`) . Once the features are generated, move them to the dataset directory. The training dataset should then have the following structure. Additionally, remember to update the training dataset path in the [configuration file](https://github.com/qihao067/CrossFlow/blob/main/configs/t2i_training_demo.py#L118).

  ```
  training_dataset   
      ├── img_text_pair.jsonl
      ├── imgs
          ├── 00a44b26-9bb4-415a-980e-a879afcb7e18.jpg
          └── ...
      └── features    # feature generated by `extract_train_feature.py`
          ├── 00a44b26-9bb4-415a-980e-a879afcb7e18.npy
          └── ...
  ```

  Similarly, we cache the latents and embeddings for the test set (e.g., MSCOCO) and the prompts used for visualization during the validation step of the training process. This can be achieved by running the following scripts [`scripts/extract_mscoco_feature.py`](https://github.com/qihao067/CrossFlow/blob/main/scripts/extract_mscoco_feature.py), [`scripts/extract_empty_feature.py`](https://github.com/qihao067/CrossFlow/blob/main/scripts/extract_empty_feature.py), and [`scripts/extract_test_prompt_feature.py`](https://github.com/qihao067/CrossFlow/blob/main/scripts/extract_test_prompt_feature.py). Before running these scripts, ensure you have downloaded the MSCOCO validation set. Then, update the dataset paths and specify the language model in each script. Once the features are generated, organize them into the following file structure in the validation dataset directory. Additionally, make sure to update the validation path in the [configuration file](https://github.com/qihao067/CrossFlow/blob/main/configs/t2i_training_demo.py#L119).

  ```
  val_dataset   
      ├── empty_context.npy
      ├── run_vis
          ├── 0.npy
          └── ...
      └── val
          ├── 0.npy
          └── ...
  ```

- ### Training

  We provide a training script for text-to-image (T2I) generation in [`train_t2i.py`](https://github.com/qihao067/CrossFlow/blob/main/train_t2i.py). Additionally, a demo configuration file is available at [`t2i_training_demo.py`](https://github.com/qihao067/CrossFlow/blob/main/configs/t2i_training_demo.py). Before starting the training, adjust the settings in the configuration file as indicated by the comments. Once configured, you can launch the training process using `N` GPUs on a single node:

  ```
  accelerate launch --multi_gpu --num_processes N --num_machines 1 --mixed_precision bf16 train_t2i_discrete.py \
              --config=configs/t2i_training_demo.py
  ```

  

------

## Terms of use

The project is created for research purposes.

______

## Acknowledgements

This codebase is built upon the following repository:

- [[U-ViT](https://github.com/baofff/U-ViT)]
- [[DiT](https://github.com/facebookresearch/DiT)]
- [[DiMR](https://github.com/qihao067/DiMR)]
- [[DeepFloyd](https://github.com/deep-floyd/IF)]

Much appreciation for their outstanding efforts.

______

## BibTeX

If you use our work in your research, please use the following BibTeX entry.

```
@article{liu2024flowing,
  title={Flowing from Words to Pixels: A Framework for Cross-Modality Evolution},
  author={Liu, Qihao and Yin, Xi and Yuille, Alan and Brown, Andrew and Singh, Mannat},
  journal={arXiv preprint arXiv:2412.15213},
  year={2024}
}
```

