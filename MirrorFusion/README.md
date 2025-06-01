# MirrorFusion

This folder contains the modifications for `MirrorFusion` built on top of the paper "BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion"


**📖 Table of Contents**


- [MirrorFusion](#mirrorfusion)
  - [🚀 Getting Started](#-getting-started)
    - [Environment Requirement 🌍](#environment-requirement-)
    - [Data Download ⬇️](#data-download-️)
  - [🏃🏼 Running Scripts](#-running-scripts)
    - [Training 🤯](#training-)
    - [Inference 📜](#inference-)
    - [Evaluation 📏](#evaluation-)
    - [Visualisation 📏](#visualisation-)
  - [💾 Checkpoint Details](#-checkpoint-details)


## 🚀 Getting Started

### Environment Requirement 🌍

MirrorFusion has been implemented and tested on Pytorch 2.3 on CUDA 12.1 with python 3.11.

We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/). For example:


We use pytorch 2.3 with CUDA 12.1 compute platform.
```
conda create -n mirror python=3.11 -y
conda activate mirror
conda install pytorch==2.3.0 torchvision==0.18.0
```

Then, you can install diffusers (implemented in this repo) with:

```
pip install -e .
```

After that, you can install required packages thourgh:

```
cd examples/brushnet/
pip install -r requirements.txt
```

[Install cuDF](https://docs.rapids.ai/install)

```
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    'cudf-cu12==24.4.*'
```

### Data Download ⬇️


**Dataset**

You can download the [SynMirror](https://huggingface.co/datasets/cs-mshah/SynMirror) dataset which is used for training and testing MirrorFusion. By downloading the data, you are agreeing to the terms and conditions of the license. The data structure should be like:

```
MirrorFusion/
    data/
        blenderproc/
            hf-objaverse-v3/
                000-142/
                    45ee52b34d314255a87af6f4d0cf7b27/
                        0.hdf5
                        1.hdf5
                        2.hdf5
            ...
            abo_v3/
                X/
                    B07B4DBBVX/
                        0.hdf5
                        1.hdf5
                        2.hdf5
            ...
            train.csv
            test.csv
```


## 🏃🏼 Running Scripts

Sample scripts for a `SLURM` setup can be found under the `slurm` folder.

### Training 🤯

Create a `.env` file under this directory with:
```
WANDB_API_KEY=<your wandb api key>
WANDB_ENTITY=<username or entity>
WANDB_PROJECT=brushnet
```

We recommend training the base unet since this gives much better results.

```
# Train using frozen base unet
accelerate launch --num_processes=8 examples/brushnet/train_brushnet_mirror.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --output_dir runs/logs/sd15_depth_20percent_drop_constant_auto_caption \
    --train_data_dir data/blenderproc \
    --resolution 512 \
    --seed 42 \
    --learning_rate 1e-5 \
    --train_batch_size 2 \
    --max_train_steps 20000 \
    --tracker_project_name brushnet \
    --report_to wandb \
    --resume_from_checkpoint latest \
    --checkpointing_steps 30000 \
    --validation_steps 1000 \
    --custom_checkpoints 15000 17000 18000 20000 \
    --num_validation_images 10 \
    --caption_column auto_caption \
    --depth_conditioning_mode concat \
    --proportion_empty_prompts 0.2

# Train with trainable base unet for improved results
accelerate launch --num_processes=4 examples/brushnet/train_brushnet_mirror.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --output_dir runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet \
    --train_data_dir data/blenderproc \
    --resolution 512 \
    --seed 42 \
    --learning_rate 1e-5 \
    --train_batch_size 4 \
    --max_train_steps 20000 \
    --tracker_project_name brushnet \
    --report_to wandb \
    --resume_from_checkpoint latest \
    --checkpointing_steps 30000 \
    --validation_steps 1000 \
    --custom_checkpoints 15000 17000 18000 20000 \
    --num_validation_images 10 \
    --caption_column auto_caption \
    --depth_conditioning_mode concat \
    --proportion_empty_prompts 0.2 \
    --train_base_unet
```


### Inference 📜

You can inference using:

```
# using defualt ground truth depth maps.
# To use monocular depth maps, set --geometric_input_data_dir and --depth_source
accelerate launch examples/brushnet/test_brushnet.py \
        --brushnet_path runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet/checkpoint-17000 \
        --output_dir inference \
        --caption_column auto_caption \
        --depth_conditioning_mode concat
```


### Evaluation 📏

The requirements are already in `examples/brushnet/requirements.txt`.

Requirements used for computing metrics:

```shell
pip install segment-anything==1.0 torchmetrics==1.4.0.post0
pip install image-reward==1.5 open-clip-torch==2.26.1

# for HPSv2:
pip install git+https://github.com/tgxs002/HPSv2.git
wget -P /path/to/envs/brushnet/lib/python3.11/site-packages/hpsv2/src/open_clip/ https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz
```

For inference images obtained, use the `evaluate_metrics.py` script under the `metrics` folder to generete csv files under the inference directories with computed metrics.

To evaluate metrics:

```bash
# first compute the selection metric for all 4 seed validation images
accelerate launch \
    metrics/evaluate_metrics.py \
    --infer_dir runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet/checkpoint-17000/inference \
    --mode calc \
    --metrics mask_SSIM

# compute required metrics based on best selected image. You can add IoU and obj also.
accelerate launch \
    metrics/evaluate_metrics.py \
    --infer_dir runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet/checkpoint-17000/inference \
    --mode best \
    --metrics mask full mirror text_align

# get the metrics df for the best selected image based on select_metric and the avg metrics output for a ckpt
python metrics/evaluate_metrics.py \
    --infer_dir runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet/checkpoint-17000/inference \
    --mode avg
```

This will create `[eval_0.csv, eval_1.csv, ..., eval_best.csv]` files inside the `--infer_dir`.

### Visualisation

To visualise the inference images run:  

Note: This requires `fiftyone`. `pip install fiftyone`.  

```
python examples/brushnet/visualise.py --infer_dir \
    runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet/checkpoint-17000/inference \
    --port 5151 \
    --csv test.csv
```

On the desktop app run the following command and select the dataset in the dropdown to view the inference images:

```
fiftyone app connect --destination <user>@<ip> --port 5151 --local-port 5151
```


## 💾 Checkpoint Details

The following table summarizes the key checkpoints mentioned in the project, along with their links and descriptions.

| Checkpoint Name                                  | Link                                                                  | Description                                                                                                                                                                                                                                                                                                                         |
| :----------------------------------------------- | :----------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MirrorFusion-v1** | [Google Drive](https://drive.google.com/drive/folders/186XN1LgklCJCC6q8-odEkQg1jllE_h2k?usp=drive_link) | This checkpoint is trained on SynMirrorV1.                                                                                                          |                                 |