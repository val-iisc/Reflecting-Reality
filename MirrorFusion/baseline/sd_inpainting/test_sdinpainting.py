import autoroot
import autorootcwd
import os
from glob import glob
import numpy as np
import logging
from tqdm.auto import tqdm
import h5py
from examples.brushnet.dataset.dataset import HDF5Dataset
import pandas as pd
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import pipeline
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler, UNet2DConditionModel
import torch
import argparse
from PIL import Image


logger = get_logger(__name__)


def read_from_marigold(geometric_data_path, uid, f_name):
    marigold_path = os.path.join(geometric_data_path, 'marigold')
    #Find the uid in this path
    file_name = os.path.join(marigold_path, 'depth_npy', f"{uid}_{f_name}_pred.npy")
    if not os.path.exists(file_name):
        return None
    else:
        return np.load(file_name)


def read_from_geowizard(geometric_data_path, uid, f_name, mode):
    geowizard_path = os.path.join(geometric_data_path, 'geowizard')
    if mode == 'depth':
        file_name = os.path.join(geowizard_path, 'depth_npy', f"{uid}_{f_name}_pred.npy")
        if not os.path.exists(file_name):
            return None
        else:
            return np.load(file_name)
    elif mode == 'normal':
        file_name = os.path.join(geowizard_path, 'normal_npy', f"{uid}_{f_name}_pred.npy")
        if not os.path.exists(file_name):
            return None
        else:
            return np.load(file_name)
    else:
        logger.error(f"Wrong mode for reading from geowizard: {mode}")
        return None


def image_grid(imgs, args):
    """Create a grid of images from a list of images. Note that the number of images must be equal to rows * cols."""
    rows = 2
    assert len(imgs) == args.num_images_per_validation, f"Number of images must be equal to {args.num_images_per_validation}"
    cols = args.num_images_per_validation // rows
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def summerize_caption(caption, summarizer):
    return summarizer(caption, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]


def get_blended_image(gt_image: Image.Image, gen_image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    create final output with mask region from gen image and rest from gt image
    """
    gt_image = gt_image.convert("RGBA")
    gen_image = gen_image.convert("RGBA")
    mask = mask.convert("RGBA")
    blended_image = Image.blend(gt_image, gen_image, alpha=0.5)
    blended_image.paste(gen_image, (0, 0), mask)
    return blended_image


def main(args):
    # choose the base model here
    # base_model_path = "krnl/realisticVisionV60B1_v51VAE"
    # base_model_path = "runwayml/stable-diffusion-v1-5"

    # input brushnet ckpt path
    # brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt"
    test_df = pd.read_csv(os.path.join(args.train_data_dir, args.csv))
    if not args.input_dir and args.num_samples:
        test_df = test_df.sample(args.num_samples, random_state=args.seed)

    if args.input_dir:
        # all hdf5 files recursively
        input_dir = os.path.join(args.train_data_dir, args.input_dir)
        hdf5_paths = glob(os.path.join(input_dir, "**/*.hdf5"), recursive=True)
        if args.num_samples:
            hdf5_paths = hdf5_paths[:args.num_samples]
        print(f"Found {len(hdf5_paths)} hdf5 files in {args.input_dir}, sample: {hdf5_paths[0]}")

    def get_data(index):
        if not args.input_dir:
            # test on all paths of args.csv
            row = test_df.iloc[index]
            caption = str(row[args.caption_column])
            hdf5_path = os.path.join(args.train_data_dir, str(row["path"]))
            hdf5_data = h5py.File(hdf5_path, "r")
            uid = row["uid"]
            f_name = os.path.split(hdf5_path)[1].split('.')[0] # camera index. eg: 0.hdf5
            return hdf5_data, caption, uid, f_name, str(row["path"])
        else:
            # use a specific folder containing hdf5 files for testing
            hdf5_path = hdf5_paths[index]
            hdf5_data = h5py.File(hdf5_path, "r")
            uid = os.path.basename(os.path.dirname(hdf5_path))  # Ex. B07QDNRBSX_B07HSDW7J6
            f_name = os.path.split(hdf5_path)[1].split('.')[0]  # camera index. eg: 0.hdf5
            if '_' in uid:
                uid1, uid2 = uid.split("_")
                caption1 = test_df[test_df['uid'] == uid1][args.caption_column].iloc[0]
                caption2 = test_df[test_df['uid'] == uid2][args.caption_column].iloc[0]
                caption = f"({caption1}) and ({caption2})"
            else:
                caption = test_df[test_df['uid'] == uid][args.caption_column].iloc[0]
            return hdf5_data, caption, uid, f_name, str(row["path"])

    def get_img_data(index):
        # This code is written for MSD dataset. Assumes images and masks are in .png format.
        row = test_df.iloc[index]
        caption = str(row[args.caption_column])
        # img_path = os.path.join(args.train_data_dir, "masked_images", str(row["path"]))
        img_path = os.path.join(args.train_data_dir, "images", str(row["path"]))
        mask_path = os.path.join(args.train_data_dir, "masks", str(row["path"]))
        uid = row["uid"]
        return (img_path, mask_path), caption, uid, 0 # Returns 0 as default camera id

    weight_dtype = torch.float32
    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif args.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16

    # summarizer for long captions
    summarizer = None
    if args.summarizer:
        summarizer = pipeline("summarization", model=args.summarizer)

    def run_inference(unet_path: str, output_dir: str):
        print(f"Running inference on {unet_path}")
        os.makedirs(output_dir, exist_ok=True)
        subfolder = "unet"
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=weight_dtype, subfolder=subfolder)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            args.base_model_path,
            unet=unet,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=False,
            safety_checker=None,
            depth_conditioning_mode=args.depth_conditioning_mode,
        )
        pipe.set_progress_bar_config(disable=True)
        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # memory optimization.
        # pipe.enable_model_cpu_offload()

        # for distributed inference
        distributed_state = PartialState()
        pipe.to(distributed_state.device)

        generator = torch.Generator("cuda").manual_seed(args.seed)
        iterate_over = range(test_df.shape[0]) if not args.input_dir else range(len(hdf5_paths))
        progress_bar = tqdm(iterate_over, disable=not distributed_state.is_local_main_process)
        with distributed_state.split_between_processes(list(iterate_over)) as idxs:
            for i in idxs:
                if args.image_mode:
                    # Reading from images (MSD dataset)
                    # override mirror prompt
                    args.mirror_prompt = ""
                    img_data, caption, uid, f_name = get_img_data(i)
                else:
                    #HDF5 mode
                    hdf5_data, caption, uid, f_name, rel_path = get_data(i)

                save_name = f"{uid}.png" if uid == f_name else f"{uid}_{f_name}.png"
                save_path = os.path.join(output_dir, save_name)
                # TODO: handle this when multiple objects are present. Shift to a caption function.
                if os.path.exists(save_path):
                    logger.warning(f"Skipping {save_path} as it already exists.")
                    progress_bar.update(distributed_state.num_processes)
                    continue

                depth_image = None
                normal_image = None

                validation_prompt = args.mirror_prompt + caption

                if summarizer:
                    validation_prompt = summerize_caption(validation_prompt, summarizer)

                if args.image_mode:
                    gt_image = Image.open(img_data[0])
                    validation_mask = Image.open(img_data[1]).convert("L")  # Convert mask to grayscale

                    # Create a black image of the same size as the validation image
                    black_image = Image.new("RGB", gt_image.size, "black")

                    # Apply the mask to the black image
                    validation_image = Image.composite(black_image, gt_image, validation_mask)
                else:
                    data = HDF5Dataset.extract_data_from_hdf5(hdf5_data)
                    gt_image = Image.fromarray(data["image"], mode="RGB")
                    if args.hint_map_dir is not None:
                        validation_image = Image.open(
                            os.path.join(args.train_data_dir, args.hint_map_dir, rel_path.replace(".hdf5", ".png"))
                        )
                    else:
                        validation_image = Image.fromarray(data["masked_image"], mode="RGB")
                    validation_mask = Image.fromarray(data["mask"]).convert("RGB")
                    if args.depth_conditioning_mode is not None:
                        #Cyclometric complexity is increaseing.
                        if args.depth_source == 'gt':
                            depth_image = HDF5Dataset.apply_transforms_depth(data["depth"], data["mask"])
                        elif args.depth_source == 'marigold':
                            depth_data = read_from_marigold(args.geometric_input_data_dir, uid, f_name)
                            if depth_data is None:
                                logger.error(f"Marigold depth doesnot exist for {uid}_{f_name}")
                                progress_bar.update(distributed_state.num_processes)
                                continue
                            depth_image = HDF5Dataset.apply_transforms_depth(depth_data, data["mask"])

                    if args.normals_conditioning_mode is not None:
                        # TODO: add support for normals in StableDiffusionInpaintPipeline
                        if args.normal_source == 'gt':
                            normal_image = Image.fromarray(data["normals"], mode="RGB")
                        elif args.normal_source == 'geowizard':
                            normal_data = read_from_geowizard(args.geometric_input_data_dir, uid, f_name, mode='normal')
                            if normal_data is None:
                                logger.error(f"Marigold depth doesnot exist for {uid}_{f_name}")
                                progress_bar.update(distributed_state.num_processes)
                                continue
                            normal_image = Image.fromarray(normal_data, mode="RGB")

                images = []
                for _ in range(args.num_images_per_validation):
                    image = pipe(
                        validation_prompt,
                        validation_image,
                        validation_mask,
                        depth=depth_image,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.CFG,
                        generator=generator,
                    ).images[0]
                    # choose whether to use blended operations
                    if args.blended:
                        image = get_blended_image(gt_image, image, validation_mask)

                    images.append(image)
                grid = image_grid(images, args)
                grid.save(save_path)
                progress_bar.update(distributed_state.num_processes)  # somehow every process isn't updating if kept 1

    if args.all_ckpt:
        ckpts = os.listdir(args.unet_path)
        ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("checkpoint")]
        ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
        for ckpt in ckpts:
            if args.ckpt_modulo is not None:
                if int(ckpt.split("-")[1]) % args.ckpt_modulo != 0:
                    continue
            unet_path = os.path.join(args.unet_path, ckpt)
            output_dir = os.path.join(unet_path, args.output_dir)
            run_inference(unet_path, output_dir)

    else:
        output_dir = os.path.join(args.unet_path, args.output_dir)
        run_inference(args.unet_path, output_dir)

if __name__ == "__main__":
    # accelerate launch --debug --num_processes=8 examples/brushnet/test_brushnet.py --brushnet_path runs/logs/sd15_full/ --num_samples 50 --all_ckpt --ckpt_modulo 10000
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_path", type=str, default="runs/logs/sd15_full", help="Path to the checkpoints folder containing unet subfolder. Example: `runs/logs/sd15_full/checkpoint-10000`")
    parser.add_argument("--weight_dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to the base model. can be krnl/realisticVisionV60B1_v51VAE",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--CFG", type=int, default=7.5)
    parser.add_argument(
        "--mirror_prompt", type=str, default="A perfect plane mirror reflection of ", help="Mirror Prompt"
    )
    parser.add_argument(
        "--summarizer",
        type=str,
        default=None,
        help="Pretrained summarizer name or path. Use `sshleifer/distilbart-cnn-6-6` when training on MSD long captions.",
    )
    parser.add_argument(
        "--num_images_per_validation",
        type=int,
        default=4,
        help="Number of images to be generated per validation image",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to generate. Default: all. Set no. for quick testing")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="data/blenderproc",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="The input directory (relative to train_data_dir from where the hdf5 files for predicting are present. Default: None",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference",
        help="The output directory name where the model predictions will be written. Default: `inference` in `args.brushnet_path`.",
    )
    parser.add_argument("--csv", type=str, default="test.csv")
    parser.add_argument("--caption_column", type=str, default="auto_caption", choices=["caption", "auto_caption"])
    parser.add_argument("--blended", action="store_true", help="Whether to use blended operation")
    parser.add_argument("--all_ckpt", action="store_true", help="Whether to evaluate all checkpoints in a directory. In this case, `brushnet_path` Is the root directory of checkpoints.")
    parser.add_argument("--ckpt_modulo", type=int, default=None, help="modulo of the ckpt folders to evaluate")
    parser.add_argument("--image_mode", action="store_true", help="If true then changes dataloader to read images and mask directly from the path")
    parser.add_argument(
        "--depth_conditioning_mode",
        type=str,
        default=None,
        help=("Depth conditioning mode. Choose between [None, 'concat', 'latents']."),
    )
    parser.add_argument(
        "--normals_conditioning_mode",
        type=str,
        default=None,
        help=("Normals conditioning mode. Choose between [None, 'concat', 'latents']."),
    )
    #TO DO : Use OmegaConf or hydra to maintain config file. There are way too many parameters
    parser.add_argument(
        "--geometric_input_data_dir",
        type=str,
        default="data/geometric_data",
        help=(
            "A folder containing the depth and normal predictions. If dataset is somewhere else, create symlink"
            )
    )
    parser.add_argument(
        "--depth_source",
        type=str,
        default='gt',
        help=("Depth Source. Choose between ['gt', 'marigold', 'depthanythingv2']."),
    )
    parser.add_argument(
        "--normal_source",
        type=str,
        default='gt',
        help=("Normal Source. Choose between ['gt', 'geowizard']."),
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
    args = parser.parse_args()
    assert args.num_images_per_validation % 2 == 0, "Number of images per validation must be even."
    main(args)
