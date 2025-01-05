import autoroot
import autorootcwd
import os
import json
import cv2
import numpy as np

from metrics.segment_reflection import (
    create_bbox_from_point,
    get_bbox_from_mask,
)


def create_sign_vector(vector):
    return np.where(vector != 0, np.sign(vector), 1).astype(int)


def get_point_from_cam_states(data, cam_pose_map):
    cam_states_array = np.array(data["cam_states"])
    json_str = cam_states_array.tobytes().decode("utf-8")
    cam_states_dict = json.loads(json_str)
    cam2world = cam_states_dict['cam2world']

    if isinstance(cam2world, list) and all(isinstance(row, list) for row in cam2world):
        cam2world_arr = np.array(cam2world)
        translational_component = cam2world_arr[:3, 3]
        translational_norm = np.linalg.norm(translational_component)
        sign_vector = create_sign_vector(translational_component)
        directed_norm = translational_norm *sign_vector[0] *sign_vector[1] *sign_vector[2]
        key = round(directed_norm, 3)
    else:
        raise ValueError("cam2world is not in the expected format")

    if str(key) in cam_pose_map.keys():
        bbox_data = cam_pose_map[str(key)]
        if isinstance(bbox_data, list):
            bbox_data = bbox_data[0]
        point, ratio_w, ratio_h, floor_path = bbox_data["point"], bbox_data["ratio_w"], bbox_data["ratio_h"], bbox_data["floor_path"]
    else:
        print("Cam2world not matching any key in cam-pose-map! Choosing the nearest point!")
        try:
            cam_values = [float(num) for num in cam_pose_map.keys()]
            nearest_value = min(cam_values, key=lambda x: abs(x - key))
            bbox_data = cam_pose_map[str(nearest_value)]
            if isinstance(bbox_data, list):
                bbox_data = bbox_data[0]
            point, ratio_w, ratio_h, floor_path = bbox_data["point"], bbox_data["ratio_w"], bbox_data["ratio_h"], bbox_data["floor_path"]

        except Exception as e:
            point, ratio_w, ratio_h, floor_path = [80, 250], 0.9, 0.9, "0.png"

    return point, ratio_w, ratio_h, floor_path


def visualize_mask_and_bbox(image, mask, bbox):
    # Create a green mask
    green_mask = np.zeros_like(image)
    green_mask[:, :, 1] = mask  # Apply the mask to the green channel

    # Create a translucent mask
    translucent_mask = cv2.addWeighted(image, 0.7, green_mask, 0.3, 0)

    # Add a red thin border around the bounding box
    border_thickness = 2
    red_color = (0, 0, 255)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(translucent_mask, (x1, y1), (x2, y2), red_color, border_thickness)

    # Convert image to RGB if it's in BGR format
    if translucent_mask.shape[2] == 3:
        translucent_mask = cv2.cvtColor(translucent_mask, cv2.COLOR_BGR2RGB)

    return translucent_mask


def get_sam_mask(segmenter, image, bbox):
    """return the sam mask given the image and bbox"""
    segmenter.set_image(image)
    masks, scores, logits = segmenter.give_mask(bbox)
    # masks, scores, logits = segmenter.give_mask_from_point(point)
    sam_mask = masks[np.argmax([np.sum(mask) for mask in masks]), :, :]
    sam_mask = (sam_mask * 255).astype(np.uint8)
    return sam_mask


def segment_image(
    gt_data,
    gen_image,
    segmenter,
    cam_pose_map,
    gt_sam_cache="",
    save_cache=True,
    use_floor_mask=False,
    use_gt_mask=False,
):

    """
    Segments the given generated image based on the ground truth file path and segmenter provided.

    Parameters:
    - gt_file_path (str): Path to the ground truth HDF5 file.
    - gen_image (str or np.ndarray): Path to the generated image or the image itself as a numpy array.
    - segmenter (SegmentPoints): An instance of the SegmentPoints class used for segmentation.
    - cam_pose_map (dict): A dictionary mapping the camera pose to the bounding box data.
    - gt_sam_cache (str, optional): Path to a cached segmented image. Defaults to a specific path.
    - save_cache (bool, optional): If True, saves the segmented image to the cache. Defaults to True.
    - use_floor_mask (bool, optional): If True, uses segmentation over the object + floor. Defaults to False.
    - use_gt_mask (bool, optional): If True, uses the segmentation mask of the ground truth for both images. Defaults to False.
    Returns:
    - sam_mask_gt (np.ndarray): The sam segmentation mask of the ground truth image.
    - masked_img_gt (np.ndarray): The ground truth image with the segmentation mask applied.
    - sam_mask_gen (np.ndarray): The sam segmentation mask of the generated image.
    - masked_img_gen (np.ndarray): The generated image with the segmentation mask applied.
    """

    point, ratio_w, ratio_h, floor_path = get_point_from_cam_states(gt_data, cam_pose_map)
    floor_mask = np.zeros_like(gt_data['mask'])
    if use_floor_mask:
        floor_mask = cv2.imread(os.path.join("metrics/floor_masks", floor_path), cv2.IMREAD_GRAYSCALE)

    if isinstance(gen_image, str):
        gen_image = cv2.imread(gen_image)

    gt_img, mirror_mask, object_mask = gt_data["image"], gt_data["mask"], gt_data["object_mask"]

    gt_masked_image = cv2.bitwise_and(gt_img, gt_img, mask=mirror_mask) # mirror region of the ground truth image
    gen_masked_img = cv2.bitwise_and(gen_image, gen_image, mask=mirror_mask) # mirror region of the generated image

    bbox_from_mask = get_bbox_from_mask(object_mask)
    x1, y1, x2, y2 = bbox_from_mask
    width = x2 - x1
    height = y2 - y1

    bbox = create_bbox_from_point(point, int(width*ratio_w), int(height*ratio_h))

    if os.path.exists(gt_sam_cache):
        sam_mask_gt = cv2.imread(gt_sam_cache, cv2.IMREAD_GRAYSCALE)
    else:
        sam_mask_gt = get_sam_mask(segmenter, gt_masked_image, bbox)
        if save_cache:
            # save the sam object mask over the ground truth image
            os.makedirs(os.path.dirname(gt_sam_cache), exist_ok=True)
            cv2.imwrite(gt_sam_cache, sam_mask_gt)

    combined_mask_gt = cv2.bitwise_and(
        cv2.bitwise_or(floor_mask, sam_mask_gt), mirror_mask
    )  # floor + obj_in_mirror mask gt
    masked_img_gt = cv2.bitwise_and(gt_img, gt_img, mask=combined_mask_gt)

    sam_mask_gen = sam_mask_gt
    if not use_gt_mask:
        sam_mask_gen = get_sam_mask(segmenter, gen_masked_img, bbox)
    combined_mask_gen = cv2.bitwise_and(cv2.bitwise_or(floor_mask, sam_mask_gen), mirror_mask) # floor + obj_in_mirror mask gen
    masked_img_gen = cv2.bitwise_and(gen_image, gen_image, mask=combined_mask_gen)

    return sam_mask_gt, masked_img_gt, sam_mask_gen, masked_img_gen
