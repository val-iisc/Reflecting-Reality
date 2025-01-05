#!/usr/bin/env python
# coding=utf-8
import autoroot
import autorootcwd
import argparse
import contextlib
import gc
import logging
import math
import os
import h5py
import shutil
from pathlib import Path
import cudf.pandas
cudf.pandas.install()
import accelerate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, pipeline

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.training_utils import compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from examples.brushnet.dataset.dataset import HDF5Dataset, MSDDataset
from metrics.metrics import compute_metrics


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__)


def summerize_caption(caption, summarizer):
    return summarizer(caption, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]

def log_validation(
    test_df,
    vae,
    text_encoder,
    tokenizer,
    summarizer,
    unet,
    args,
    accelerator,
    weight_dtype,
    step,
    is_final_validation=False,
):

    def get_hdf5_data(index):
        row = test_df.iloc[index]
        caption = str(row[args.caption_column])
        hdf5_path = os.path.join(args.train_data_dir, str(row["path"]))
        hdf5_data = h5py.File(hdf5_path, "r")
        data = HDF5Dataset.extract_data_from_hdf5(hdf5_data)

        validation_full_image = Image.fromarray(data["image"], mode="RGB")
        if args.hint_map_dir is not None:
            validation_image = Image.open(
                os.path.join(args.train_data_dir, args.hint_map_dir, str(row["path"]).replace("hdf5", "png"))
            )
        else:
            validation_image = Image.fromarray(data["masked_image"], mode="RGB")
        validation_mask = Image.fromarray(data["mask"]).convert("RGB")
        depth_image = None
        normal_image = None

        if args.depth_conditioning_mode is not None:
            depth_image = HDF5Dataset.apply_transforms_depth(data["depth"], data["mask"])

        if args.normals_conditioning_mode in ["concat", "latents"]:
            normal_image = Image.fromarray(data["normals"], mode="RGB")

        return caption, validation_full_image, validation_image, validation_mask, depth_image, normal_image

    def get_MSD_data(index):
        row = test_df.iloc[index]
        caption = str(row[args.caption_column])
        image_path = str(row["path"])
        image = np.array(Image.open(Path(args.train_data_dir) / "images" / image_path))
        mask = np.array(Image.open(Path(args.train_data_dir) / "masks" / image_path))
        masked_image = MSDDataset.get_masked_image(image, mask)

        validation_full_image = Image.fromarray(image, mode="RGB")
        validation_image = Image.fromarray(masked_image, mode="RGB")
        validation_mask = Image.fromarray(mask).convert("RGB")
        depth_image = None
        normal_image = None

        return caption, validation_full_image, validation_image, validation_mask, depth_image, normal_image

    logger.info("Running validation... ")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        depth_conditioning_mode=args.depth_conditioning_mode,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    assert test_df.shape[0] == args.num_validation_images, "Number of validation images should be equal to the number of images in the test dataset"

    prompts = []
    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
    all_metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": []
    }
    for i in range(args.num_validation_images):

        if args.dataset_type == "HDF5":
            validation_prompt, validation_full_image, validation_image, validation_mask, depth_image, normal_image = (
                get_hdf5_data(i)
            )
        elif args.dataset_type == "MSD":
            validation_prompt, validation_full_image, validation_image, validation_mask, depth_image, normal_image = (
                get_MSD_data(i)
            )

        validation_prompt = args.mirror_prompt + validation_prompt

        if summarizer is not None:
            validation_prompt = summerize_caption(validation_prompt, summarizer)

        seed_images = [validation_image]
        prompts.append(validation_prompt)
        metrics = {
            "psnr": [],
            "ssim": [],
            "lpips": []
        }

        for _ in range(args.num_images_per_validation):
            with inference_ctx:
                image = pipe(
                    validation_prompt,
                    validation_image,
                    validation_mask,
                    depth=depth_image,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                ).images[0]

            metric = compute_metrics(np.array(image), np.array(validation_full_image))
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            text = f"PSNR: {metric['psnr']:.2f}\nSSIM: {metric['ssim']:.2f}\nLPIPS: {metric['lpips']:.2f}"
            draw.text((10, 10), text, font=font, fill=(0, 255, 0))  # Using green color to highlight the text
            seed_images.append(image)
            metrics["psnr"].append(metric["psnr"])
            metrics["ssim"].append(metric["ssim"])
            metrics["lpips"].append(metric["lpips"])

        all_metrics["psnr"].append(max(metrics["psnr"]))
        all_metrics["ssim"].append(max(metrics["ssim"]))
        all_metrics["lpips"].append(min(metrics["lpips"]))

        image_logs.append(make_image_grid(seed_images, 1, len(seed_images)))

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            # NOTE: not updated. Use wandb
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    tracker_key: [
                        wandb.Image(image, caption=f"{i}: {prompts[i]}")
                        for i, image in enumerate(image_logs)
                    ],
                    "psnr": np.mean(all_metrics["psnr"]),
                    "ssim": np.mean(all_metrics["ssim"]),
                    "lpips": np.mean(all_metrics["lpips"]),
                },
                step=step,
            )

        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# brushnet-{repo_id}

These are brushnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "brushnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a SD inpainting training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--summarizer",
        type=str,
        default=None,
        help="Pretrained summarizer name or path. Use `sshleifer/distilbart-cnn-6-6` when training on MSD long captions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="run/logs/sd_inpainting",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    # custom_checkpoints argument as a list to checkpoint at those particular steps
    parser.add_argument(
        "--custom_checkpoints",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Save a checkpoint of the training state at the specified updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--snr_gamma", type=float, default=None, help="Gamma value for SNR computation. Use: [1, 5]")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="HDF5",
        choices=["HDF5", "MSD"],
        help=(
            "The type of the dataset. Choose between ['HDF5', 'MSD']."
        ),
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="train.csv",
        help="The name of the CSV file containing the training data.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="test.csv",
        help="The name of the CSV file containing the test data.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="auto_caption",
        help="The column of the csv containing the captions.",
        choices=["caption", "auto_caption"],
    )
    parser.add_argument(
        "--mirror_prompt",
        type=str,
        default="A perfect plane mirror reflection of ",
        help="The prompt used to describe the mirror reflection.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of validation images to be generated from train_data_dir/test.csv",
    )
    parser.add_argument(
        "--validation_csv_indices",
        type=int,
        nargs="+",
        default=[],
        help="index of rows from test csv to log in validation logger.",
    )
    parser.add_argument(
        "--num_images_per_validation",
        type=int,
        default=4,
        help="Number of images to be generated per validation image",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of scheduler steps during inference. Default 20.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_inpainting",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--random_mask",
        action="store_true",
        help=(
            "Training BrushNet with random mask"
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--hint_map_dir",
        type=str,
        default=None,
        help=(
            "If using, set to the directory relative to train_data_dir containing the hint maps. \
            Folder contents must follow the structure described in csv['path']. Use: `pcd_map`."
        ),
    )
    parser.add_argument(
        "--depth_conditioning_mode",
        type=str,
        default=None,
        choices=[None, "concat", "latents"],
        help=(
            "Depth conditioning mode. Choose between [None, 'concat', 'latents']."
        ),
    )
    parser.add_argument(
        "--normals_conditioning_mode",
        type=str,
        default=None,
        choices=[None, "concat", "latents"],
        help=("Normals conditioning mode. Choose between [None, 'concat', 'latents']."),
    )
    parser.add_argument(
        "--cam_states",
        action="store_true",
        help=("Return cam2world and cam_K matrix from the dataset."),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    assert args.num_validation_images > 0, "Number of validation images should be greater than 0"
    assert args.num_images_per_validation > 0, "Number of images per validation image should be greater than 0"

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the unet encoder."
        )

    return args


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    masks = torch.stack([example["masks"] for example in examples])
    masks = masks.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    batch = {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "masks": masks,
        "input_ids": input_ids,
    }

    if "depths" in examples[0]:
        depths = torch.stack([example["depths"] for example in examples])
        depths = depths.to(memory_format=torch.contiguous_format).float()
        batch["depths"] = depths

    if "normals" in examples[0]:
        normals = torch.stack([example["normals"] for example in examples])
        normals = normals.to(memory_format=torch.contiguous_format).float()
        batch["normals"] = normals

    if "cam2world" in examples[0]:
        cam2world = torch.stack([example["cam2world"] for example in examples])
        cam2world = cam2world.to(memory_format=torch.contiguous_format).float()
        cam_K = torch.stack([example["cam_K"] for example in examples])
        cam_K = cam_K.to(memory_format=torch.contiguous_format).float()
        batch["cam2world"] = cam2world
        batch["cam_K"] = cam_K

    return batch


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # summarizer for long captions
    summarizer = None
    if args.summarizer:
        summarizer = pipeline("summarization", model=args.summarizer)

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    conditioning_channels = 5 # 4 masked_image, 1 mask
    # depth conditioning
    if args.depth_conditioning_mode == "concat": # concat like mask
        conditioning_channels += 1 # depth is single channel
    elif args.depth_conditioning_mode == "latents": # pass through VAE and concat (like masked_image)
        conditioning_channels += 4

    # normals conditioning
    if args.normals_conditioning_mode == "concat":
        conditioning_channels += 3 # normals is 3 channel
    elif args.normals_conditioning_mode == "latents":
        conditioning_channels += 4

    if unet.conv_in.in_channels == 4:
        logger.info(f"Initializing the Inpainting UNet from the pretrained 4 channel UNet. In channels: {unet.conv_in.in_channels}, conditioning channels: {conditioning_channels}")
        in_channels = unet.conv_in.in_channels + conditioning_channels
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            # preserve the weights of initial 4 channels, rest are initialized to 0
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in

    elif unet.conv_in.in_channels == 9:
        logger.info(f"Initializing the Inpainting UNet from the pretrained 9 channel Inpainting UNet. In channels: {unet.conv_in.in_channels}, conditioning channels: {conditioning_channels - 4}")
        in_channels = unet.conv_in.in_channels + (conditioning_channels - 5) # remove 4 channels for masked latents and 1 channel for the mask
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            # preserve the weights of initial 9 channels, rest are initialized to 0
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :9, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(f"unet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    df = pd.read_csv(os.path.join(args.train_data_dir, args.train_csv))
    if args.max_train_samples is not None:
        df = df.head(args.max_train_samples)
    test_df = pd.read_csv(os.path.join(args.train_data_dir, args.test_csv))
    if len(args.validation_csv_indices) > 0:
        test_df = test_df.iloc[args.validation_csv_indices]
    else:
        test_df = test_df.sample(args.num_validation_images, random_state=args.seed, ignore_index=True)

    if args.dataset_type == "HDF5":
        train_dataset = HDF5Dataset(
            args.train_data_dir,
            df,
            tokenizer,
            args.resolution,
            args.random_mask,
            args.proportion_empty_prompts,
            mirror_prompt=args.mirror_prompt,
            caption_column=args.caption_column,
            random_flip=args.random_flip,
            depth=args.depth_conditioning_mode,  # whether to return depth images
            normals_conditioning_mode=args.normals_conditioning_mode,  # whether to return normal maps/vector
            cam_states=args.cam_states,  # whether to return camera states
            hint_map_dir=args.hint_map_dir,  # hint map directory if using hint map like point cloud reprojection
        )
    elif args.dataset_type == "MSD":
        # override args for MSD training
        args.mirror_prompt = ""
        train_dataset = MSDDataset(
            args.train_data_dir,
            df,
            tokenizer,
            resolution=args.resolution,
            proportion_empty_prompts=args.proportion_empty_prompts,
            mirror_prompt=args.mirror_prompt,
            caption_column=args.caption_column,
            random_flip=args.random_flip,
        )

    train_dataset_len = len(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, lr_scheduler, unet = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler, unet
    )

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {train_dataset_len}")
    logger.info(f"  Num update steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path),map_location="cpu")
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                conditioning_latents=vae.encode(batch["conditioning_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                conditioning_latents = conditioning_latents * vae.config.scaling_factor

                masks = torch.nn.functional.interpolate(
                    batch["masks"],
                    size=(
                        latents.shape[-2],
                        latents.shape[-1]
                    )
                )

                conditioning_latents = torch.concat([masks, conditioning_latents], 1) # note the concat order as per the pipeline

                # condition using depth
                if args.depth_conditioning_mode == "concat":
                    depths = batch["depths"].to(dtype=weight_dtype)
                    depths = torch.nn.functional.interpolate(
                        depths,
                        size=(
                            latents.shape[-2],
                            latents.shape[-1]
                        )
                    )
                    conditioning_latents=torch.concat([conditioning_latents, depths], 1)

                elif args.depth_conditioning_mode == "latents":
                    # Repeat the depths along the channel dimension to get 3 channel depth
                    depths = batch["depths"].repeat(1, 3, 1, 1)
                    depth_latents = vae.encode(depths.to(dtype=weight_dtype)).latent_dist.sample()
                    depth_latents = depth_latents * vae.config.scaling_factor
                    conditioning_latents = torch.concat([conditioning_latents, depth_latents], 1)

                # condition using normals
                # TODO: Add support for normals conditioning in the pipeline
                if args.normals_conditioning_mode == "concat":
                    normals = batch["normals"].to(dtype=weight_dtype)
                    normals = torch.nn.functional.interpolate(
                        normals,
                        size=(
                            latents.shape[-2],
                            latents.shape[-1]
                        )
                    )
                    conditioning_latents=torch.concat([conditioning_latents, normals], 1)

                elif args.normals_conditioning_mode == "latents":
                    normals_latents = vae.encode(batch["normals"].to(dtype=weight_dtype)).latent_dist.sample()
                    normals_latents = normals_latents * vae.config.scaling_factor
                    conditioning_latents = torch.concat([conditioning_latents, normals_latents], 1)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Predict the noise residual and compute loss
                combined_latents = torch.cat([noisy_latents, conditioning_latents], dim=1)
                model_pred = unet(combined_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = (
                    accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                    / args.gradient_accumulation_steps
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    custom_checkpoints = [] if args.custom_checkpoints is None else args.custom_checkpoints
                    if global_step % args.checkpointing_steps == 0 or global_step in custom_checkpoints:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.num_validation_images > 0 and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            test_df,
                            vae,
                            text_encoder,
                            tokenizer,
                            summarizer,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                logger.info(f"gloabl_step: {global_step} >= args.max_train_steps: {args.max_train_steps}")
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # unet = unwrap_model(unet)
        # unet.save_pretrained(os.path.join(args.output_dir, "unet"))

        # Run a final round of validation.
        image_logs = None
        if args.num_validation_images > 0:
            image_logs = log_validation(
                test_df,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                summarizer=summarizer,
                unet=unet,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
