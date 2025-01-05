import json
import random
from pathlib import Path

import os
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class HDF5Dataset(Dataset):
    """
    Dataset class to iterate over the blenderproc generated hdf5 synthetic dataset
    """

    def __init__(
        self,
        data_root: str,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        resolution: int = 512,
        proportion_empty_prompts: float = 0.1,
        mirror_prompt: str = "A perfect plane mirror reflection of ",
        caption_column: str = "auto_caption",
        random_flip: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.data_root = Path(data_root)
        self.df = df
        self.proportion_empty_prompts = proportion_empty_prompts
        self.mirror_prompt = mirror_prompt
        self.caption_column = caption_column
        self.random_flip = random_flip
        self.kwargs = kwargs

    def __len__(self):
        return self.df.shape[0]

    def tokenize_caption(self, caption: str):
        if random.random() < self.proportion_empty_prompts:
            caption = ""
        elif isinstance(caption, str):
            caption = self.mirror_prompt + caption
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    @staticmethod
    def get_masked_image(image, mask, invert=True):
        masked_image = image.copy()
        if invert:
            masked_image[mask == 255] = 0
        else:
            masked_image[mask == 0] = 0
        return masked_image

    @staticmethod
    def apply_transforms_rgb(image: np.ndarray, resolution=512):
        image = np.copy(image)  # Make a copy to ensure positive strides
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        image = image_transforms(image)
        return image

    @staticmethod
    def apply_transforms_mask(mask: np.ndarray, resolution=512):
        mask = np.copy(mask)  # Make a copy to ensure positive strides
        mask = torch.tensor(mask, dtype=torch.float32) / 255.0
        mask = mask.unsqueeze(0)
        mask_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
            ]
        )
        mask = mask_transforms(mask)
        return mask

    @staticmethod
    def apply_transforms_depth(
        depth_map: np.ndarray,
        mask: np.ndarray = None,
        normalization_method="max_scene_depth",
        max_scene_depth=5.0,
        norm_range=[-1, 1],
        delta=0.5,
        resolution=512,
        **kwargs,
    ):
        depth_map = np.copy(depth_map)  # Make a copy to ensure positive strides

        # Ensure mask is 2D by taking the first channel if it's 3D
        if mask is not None and mask.ndim == 3:
            mask = mask[:, :, 0]

        if normalization_method == "percentile":
            # Calculate the 2% and 98% percentiles
            d_2 = np.percentile(depth_map, 2)
            d_98 = np.percentile(depth_map, 98)

            # Clip the depth_map map to the range [d_2, d_98]
            clipped_depth_map = np.clip(depth_map, d_2, d_98)

            # Normalize to the specified range
            if norm_range == [0, 1]:
                normalized_depth = (clipped_depth_map - d_2) / (d_98 - d_2)
            elif norm_range == [-1, 1]:
                normalized_depth = 2.0 * (clipped_depth_map - d_2) / (d_98 - d_2) - 1.0
            else:
                raise ValueError("Unsupported normalization range. Use [0, 1] or [-1, 1].")
        elif normalization_method == "max_scene_depth":
            if mask is not None:
                # Convert mask to boolean where non-zero values are True
                bool_mask = mask > 0
                # Calculate the maximum depth value over the mask
                max_depth_over_mask = np.max(depth_map[bool_mask])
                max_scene_depth = max_depth_over_mask + delta

            # Clip the depth map to the range [0, max_scene_depth]
            clipped_depth = np.clip(depth_map, 0, max_scene_depth)

            # Normalize to the specified range
            if norm_range == [0, 1]:
                normalized_depth = clipped_depth / max_scene_depth
            elif norm_range == [-1, 1]:
                normalized_depth = 2.0 * (clipped_depth / max_scene_depth) - 1.0
            else:
                raise ValueError("Unsupported normalization range. Use [0, 1] or [-1, 1].")
        else:
            raise ValueError("Unsupported normalization method. Use 'percentile' or 'max_scene_depth'.")

        # Convert to tensor and apply transformations
        normalized_depth = torch.tensor(normalized_depth, dtype=torch.float32).unsqueeze(0)
        # Resize while keeping aspect ratio, then center crop
        resize_transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BICUBIC
                ),  # Resize height to maintain width of 512
                transforms.CenterCrop((resolution, resolution)),  # Crop to the target resolution
            ]
        )

        # Apply transforms
        normalized_depth = resize_transform(normalized_depth)

        return normalized_depth

    @staticmethod
    def apply_transforms_normals(
        normals_map: np.ndarray, resolution=512, mask=None, normals_conditioning_mode="ip_adapter", **kwargs
    ):
        normals_map = np.copy(normals_map)  # Make a copy to ensure positive strides
        if normals_conditioning_mode == "ip_adapter":
            # Convert mask to boolean where non-zero values are True
            bool_mask = mask > 0
            # Calculate the mean normals vector over the mask region and normalise
            mean_normals_over_mask = np.mean(normals_map[bool_mask], axis=0)  # (3,)
            # Normalize the mean normals vector
            normalized_mean_normals = mean_normals_over_mask / np.linalg.norm(mean_normals_over_mask)
            return torch.tensor(normalized_mean_normals, dtype=torch.float32).unsqueeze(0)  # (1, 3)

        else:
            normals_map = torch.tensor(normals_map, dtype=torch.float32).permute(2, 0, 1)
            normals_transforms = transforms.Compose(
                [
                    transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(resolution),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            normals_map = normals_transforms(normals_map)
            return normals_map

    @staticmethod
    def decode_cam_states(cam_states):
        """return cam2world, cam_K matrices from cam_states both as lists"""
        array = np.array(cam_states)
        json_str = array.tobytes().decode("utf-8")
        data = json.loads(json_str)
        cam2world = data["cam2world"]
        cam_K = data["cam_K"]
        return cam2world, cam_K

    @staticmethod
    def extract_data_from_hdf5(hdf5_data, random_flip=False):
        data = {
            "image": np.array(hdf5_data["colors"], dtype=np.uint8),
            "mask": (np.array(hdf5_data["category_id_segmaps"], dtype=np.uint8) == 1).astype(np.uint8) * 255,
            "object_mask": (np.array(hdf5_data["category_id_segmaps"], dtype=np.uint8) == 2).astype(np.uint8) * 255,
            "masked_image": None,  # Placeholder for masked_image
            "depth": np.array(hdf5_data["depth"]),
            "normals": np.array(hdf5_data["normals"]),
            "cam_states": np.array(hdf5_data["cam_states"]),
        }

        data["masked_image"] = HDF5Dataset.get_masked_image(data["image"], data["mask"])

        if random_flip:
            # TODO: add appropriate flip for cam_states if used
            for key in ["image", "mask", "object_mask", "masked_image", "depth", "normals"]:
                data[key] = np.fliplr(data[key])

        return data

    def __getitem__(self, index):
        example = {}
        row = self.df.iloc[index]
        caption = str(row[self.caption_column])
        hdf5_path = self.data_root / str(row["path"])
        hdf5_data = h5py.File(hdf5_path, "r")

        # generate common random_flip flag for all images
        flip_horizontal = self.random_flip and random.random() < 0.5
        data = self.extract_data_from_hdf5(hdf5_data, random_flip=flip_horizontal)

        image = self.apply_transforms_rgb(data["image"], resolution=self.resolution)
        mask = self.apply_transforms_mask(data["mask"], resolution=self.resolution)

        if self.kwargs.get("hint_map_dir", False): ### This can be used if using a hint image as a conditioning input
            hint_image_path = str(row["path"]).replace("hdf5", "png")
            hint_image_path = os.path.join(str(self.data_root), self.kwargs["hint_map_dir"], hint_image_path)
            hint_image = np.array(Image.open(hint_image_path))
            hint_image = self.apply_transforms_rgb(hint_image, resolution=self.resolution)
            example["conditioning_pixel_values"] = hint_image
        else:
            masked_image = self.apply_transforms_rgb(data["masked_image"], resolution=self.resolution)
            example["conditioning_pixel_values"] = masked_image

        example["input_ids"] = self.tokenize_caption(caption)[0]
        example["pixel_values"] = image
        example["masks"] = mask

        if self.kwargs.get("depth", False):
            depth = self.apply_transforms_depth(
                data["depth"], mask=data["mask"], resolution=self.resolution, **self.kwargs
            )
            example["depths"] = depth

        if self.kwargs.get("normals_conditioning_mode", False):
            normals = self.apply_transforms_normals(
                data["normals"], resolution=self.resolution, mask=data["mask"], **self.kwargs
            )
            example["normals"] = normals

        if self.kwargs.get("cam_states", False):
            # TODO: add transforms if used in training
            cam2world, cam_K = self.decode_cam_states(data["cam_states"])
            example["cam2world"] = cam2world
            example["cam_K"] = cam_K

        return example


class MSDDataset(HDF5Dataset):
    """
    Dataset class to iterate over the MSD dataset
    """

    def __init__(
        self,
        data_root: str,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        mirror_prompt: str = "",  # default caption already contains "mirror"
        caption_column: str = "auto_caption",
        path_column: str = "path",
        **kwargs,
    ):
        super().__init__(
            data_root, df, tokenizer, mirror_prompt=mirror_prompt, caption_column=caption_column, **kwargs
        )
        self.path_column = path_column
        self.images_dir = self.data_root / "images"
        self.masks_dir = self.data_root / "masks"
        self.depth_dir = self.data_root / "depth"

    def __getitem__(self, index):
        example = {}
        row = self.df.iloc[index]
        caption = str(row[self.caption_column])
        image_path = str(row[self.path_column])

        image = np.array(Image.open(str(self.images_dir / image_path)))
        orig_mask = np.array(Image.open(str(self.masks_dir / image_path)))
        masked_image = self.get_masked_image(image, orig_mask)

        image = self.apply_transforms_rgb(image, resolution=self.resolution)
        mask = self.apply_transforms_mask(orig_mask, resolution=self.resolution)
        masked_image = self.apply_transforms_rgb(masked_image, resolution=self.resolution)

        example["input_ids"] = self.tokenize_caption(caption)[0]
        example["pixel_values"] = image
        example["conditioning_pixel_values"] = masked_image
        example["masks"] = mask

        if self.kwargs.get("depth", False): # depth for MSD dataset generated using Depth-Pro
            depth = np.load(self.depth_dir / image_path.replace("png", "npz"))["depth"]
            depth = self.apply_transforms_depth(depth, mask=orig_mask, resolution=self.resolution, **self.kwargs)
            example["depths"] = depth

        return example


if __name__ == "__main__":
    pass
