import autoroot
import autorootcwd
import os
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


class SegmentPoints:
    def __init__(self, checkpoint_folder, version='vit_b', device='cuda'):
        self.checkpoint_folder = checkpoint_folder
        self.version = version
        self.sam = self.load_checkpoint()
        self.predictor = SamPredictor(self.sam)
        if torch.cuda.is_available():
            self.predictor.model = self.predictor.model.to(device)

    def load_checkpoint(self):
        if "vit_b" in self.version:
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            checkpoint_file = os.path.join(self.checkpoint_folder, "sam_vit_b_01ec64.pth")
            if not os.path.exists(checkpoint_file):
                self.save_checkpoint(url, checkpoint_file)
            return sam_model_registry["vit_b"](checkpoint=checkpoint_file)
        elif "vit_l" in self.version:
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
            checkpoint_file = os.path.join(self.checkpoint_folder, "sam_vit_l_0b3195.pth")
            if not os.path.exists(checkpoint_file):
                self.save_checkpoint(url, checkpoint_file)
            return sam_model_registry["vit_l"](checkpoint=checkpoint_file)
        elif "vit_h" in self.version:
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            checkpoint_file = os.path.join(self.checkpoint_folder, "sam_vit_h_4b8939.pth")
            if not os.path.exists(checkpoint_file):
                self.save_checkpoint(url, checkpoint_file)
            return sam_model_registry["vit_h"](checkpoint=checkpoint_file)
        else:
            raise ValueError("Unsupported checkpoint type")

    def save_checkpoint(self, url, checkpoint_file):
        subprocess.run(["wget", "-O", checkpoint_file, url])

    def set_image(self, image):
        if isinstance(image, str):
            image = self.load_image(image)
        elif isinstance(image, np.ndarray):
            pass  # image is already a numpy array
        elif isinstance(image, Image.Image):
            image = np.array(image)
        else:
            print("Unsupported image type")
            return
        self.predictor.set_image(image)

    def give_mask(self, bbox):
        input_box = np.array([bbox])
        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=True,
        )
        return masks, scores, logits

    def give_mask_from_point(self, point):
        point_coords = np.array([point])
        point_labels = np.array([1])  # Assuming label 1 for the given point
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        return masks, scores, logits

    def get_mask(self, masks, scores):
        return masks[np.argmax([np.sum(mask) for mask in masks]),:,:]

    def load_image(self, image_path):
        return cv2.imread(image_path)

    def segment_whole_object(self, bbox):
        masks, scores, logits = self.give_mask(bbox)
        selected_mask = self.get_mask(masks, scores)
        return selected_mask


def create_bbox_from_point(point, width, height):
    width = max(width, 50)
    height = max(height, 50)
    x, y = point
    x1 = max(0, x - width // 2)
    y1 = max(0, y - height // 2)
    x2 = x + width // 2
    y2 = y + height // 2
    return (x1, y1, x2, y2)


def get_bbox_from_mask(mask):
    # Find the contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    # Get the bounding box that encloses all contours
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    return (x_min, y_min, x_max, y_max)


if __name__ == "__main__":
    pass
