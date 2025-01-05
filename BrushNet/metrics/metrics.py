import autoroot
import autorootcwd
import os
import json
from urllib.request import urlretrieve

import hpsv2
import ImageReward as RM
import numpy as np
import open_clip
import torch
from examples.brushnet.dataset.dataset import HDF5Dataset
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from torchmetrics.functional.multimodal.clip_score import clip_score
from torchvision import transforms

from metrics.object_metrics import segment_image
from metrics.segment_reflection import SegmentPoints


to_pil = transforms.ToPILImage()

def get_normalised_tensor(image, range=[-1, 1], device=None):
    """Normalize the image tensor
    Args:
        image (torch.Tensor or np.ndarray): Image tensor to normalize
        range (list): Normalization range
        device (torch.device): Device to move the tensor to
    Returns:
        torch.Tensor: Normalized image tensor
        torch.Tensor: Original image tensor
    """
    if isinstance(image, np.ndarray):
        image = torch.tensor(image, device=device).permute(2, 0, 1).unsqueeze(0).float()

    # Copy the original image
    original_image = image.clone()

    # Normalize the image
    if range == [-1, 1]:
        image = image / 127.5 - 1
    elif range == [0, 1]:
        image = image / 255.0
    else:
        raise ValueError("Unsupported normalization range. Use [-1, 1] or [0, 1].")

    return image, original_image


def compute_metrics(pred, gt, norm_range=[-1, 1]):
    """
    Internally calls the torchmetrics functional API (via `MetricsCalculator`) to compute LPIPS, SSIM, PSNR
    between the predicted(generated) and ground truth images.
    Args:
        pred (torch.Tensor or np.ndarray): Predicted image tensor
        gt (torch.Tensor or np.ndarray): Ground truth image tensor
        norm_range (list): Normalization range required for LPIPS. Default is [-1, 1]
    """
    pred_normalized, pred_original = get_normalised_tensor(pred, norm_range)
    gt_normalized, gt_original = get_normalised_tensor(gt, norm_range)

    return {
        "lpips": MetricsCalculator.calculate_lpips(pred_normalized, gt_normalized),
        "ssim": MetricsCalculator.calculate_ssim(pred_original, gt_original),
        "psnr": MetricsCalculator.calculate_psnr(pred_original, gt_original)
    }


class MetricsCalculator:
    def __init__(self, metrics_to_compute, device, data_dir, cache_dir, ckpt_path="data/ckpt", norm_range=[-1, 1]) -> None:
        self.device = device
        self.metrics_to_compute = metrics_to_compute
        self.norm_range = norm_range
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cam_pose_map = None

        check_if_obj = any("obj" in metric for metric in self.metrics_to_compute)
        check_if_iou = any("IoU" in metric for metric in self.metrics_to_compute)
        if check_if_obj or check_if_iou:
            self.segmenter = SegmentPoints(version='vit_h', checkpoint_folder=ckpt_path)
            with open('metrics/cam_pose_map.json', 'r') as json_file:
                self.cam_pose_map = json.load(json_file)

        if "Aesthetic_Score" in self.metrics_to_compute:
            # aesthetic model
            self.aesthetic_model = torch.nn.Linear(768, 1, device=device)
            aesthetic_model_url = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
            )
            os.makedirs(ckpt_path, exist_ok=True)
            aesthetic_model_ckpt_path = os.path.join(ckpt_path, "sa_0_4_vit_l_14_linear.pth")
            urlretrieve(aesthetic_model_url, aesthetic_model_ckpt_path)
            self.aesthetic_model.load_state_dict(torch.load(aesthetic_model_ckpt_path))
            self.aesthetic_model.eval()
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=device
            )
            # remove PIL related transforms
            self.clip_preprocess.transforms.pop(2)
            self.clip_preprocess.transforms.pop(2)

        if "Image_Reward" in self.metrics_to_compute:
            # image reward model
            self.imagereward_model = RM.load("ImageReward-v1.0", download_root=ckpt_path, device=device)

    def compute_metric(self, metric_name, gen_image, gt_data, caption):
        gen_image = np.array(gen_image)
        gt_image = gt_data["image"]
        if "obj" in metric_name:
            rel_path = gt_data["file_path"].split(".")[0] # ex. abo_v3/B/B07HSLGGDB/0
            gt_sam_cache = os.path.join(self.data_dir, self.cache_dir, f"{rel_path}.png")
            _, gt_image, _, gen_image = segment_image(
                gt_data=gt_data,
                gen_image=gen_image,
                segmenter=self.segmenter,
                cam_pose_map=self.cam_pose_map,
                gt_sam_cache=gt_sam_cache,
                use_floor_mask=True,  # use segmentation over the floor + object
                use_gt_mask=True,  # use the sam_mask of gt for both the ground truth and generated image
            )

        elif "IoU" in metric_name:
            rel_path = gt_data["file_path"].split(".")[0]  # ex. abo_v3/B/B07HSLGGDB/0
            gt_sam_cache = os.path.join(self.data_dir, self.cache_dir, f"{rel_path}.png")
            gt_sam_mask, _, gen_sam_mask, _ = segment_image(
                gt_data=gt_data,
                gen_image=gen_image,
                segmenter=self.segmenter,
                cam_pose_map=self.cam_pose_map,
                gt_sam_cache=gt_sam_cache,
                use_floor_mask=False,  # use only segmentation over the object
                use_gt_mask=False,  # use sam for both the ground truth and generated image
            )

            return self.calculate_iou(gen_sam_mask, gt_sam_mask)

        elif "mask" in metric_name:
            gt_image = gt_data["masked_image"]
            gen_image = HDF5Dataset.get_masked_image(gen_image, gt_data["mask"])

        elif "mirror" in metric_name:
            gt_image = HDF5Dataset.get_masked_image(gt_data["image"], gt_data["mask"], invert=False)
            gen_image = HDF5Dataset.get_masked_image(gen_image, gt_data["mask"], invert=False)

        pred_normalized, pred_original = get_normalised_tensor(gen_image, self.norm_range, self.device)
        gt_normalized, gt_original = get_normalised_tensor(gt_image, self.norm_range, self.device)

        if "LPIPS" in metric_name:
            return self.calculate_lpips(pred_normalized, gt_normalized)
        elif "PSNR" in metric_name:
            return self.calculate_psnr(pred_original, gt_original)
        elif "SSIM" in metric_name:
            return self.calculate_ssim(pred_original, gt_original)
        elif "CLIP_Similarity" in metric_name:
            return self.calculate_clip_similarity(pred_original, caption)
        elif "Aesthetic_Score" in metric_name:
            return self.calculate_aesthetic_score(pred_original)
        elif "Image_Reward" in metric_name:
            return self.calculate_image_reward(pred_original, caption)
        elif "HPS_V2.1" in metric_name:
            return self.calculate_hpsv21_score(pred_original, caption)
        else:
            raise ValueError(f"Unsupported metric {metric_name}")

    def calculate_image_reward(self, image, prompt):
        reward = self.imagereward_model.score(prompt, to_pil(image.squeeze()))
        return reward

    def calculate_hpsv21_score(self, image, prompt):
        image = to_pil(image.squeeze())
        result = hpsv2.score(image, prompt, hps_version="v2.1")[0]
        return result.item()

    def calculate_aesthetic_score(self, img):
        image = self.clip_preprocess(img)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = self.aesthetic_model(image_features)
        return prediction.item()

    @staticmethod
    def calculate_iou(gen_mask, gt_mask):
        intersection = np.logical_and(gen_mask, gt_mask)
        union = np.logical_or(gen_mask, gt_mask)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    @staticmethod
    def calculate_clip_similarity(img, txt):
        score = clip_score(img, txt, model_name_or_path="openai/clip-vit-large-patch14")
        return score.item()

    @staticmethod
    def calculate_psnr(pred_img, gt_img):
        psnr_score = peak_signal_noise_ratio(pred_img, gt_img)
        return psnr_score.item()

    @staticmethod
    def calculate_lpips(pred_img, gt_img, net_type="squeeze"):
        lpips_score = learned_perceptual_image_patch_similarity(pred_img, gt_img, net_type=net_type)
        return lpips_score.item()

    @staticmethod
    def calculate_ssim(pred_img, gt_img):
        ssim_score = structural_similarity_index_measure(pred_img, gt_img)
        return ssim_score.item()


if __name__ == "__main__":
    pass
