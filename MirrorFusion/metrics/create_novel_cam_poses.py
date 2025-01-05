import json
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np


# Add the dataset path to the system path
sys.path.append('../examples/brushnet/dataset')
from dataset import HDF5Dataset


def create_sign_vector(vector):
    return np.where(vector != 0, np.sign(vector), 1).astype(int)

def get_cam2world_key(hdf5_path):
    hdf5_data = h5py.File(hdf5_path, "r")
    data = HDF5Dataset.extract_data_from_hdf5(hdf5_data)
    cam_states_array = np.array(data["cam_states"])
    json_str = cam_states_array.tobytes().decode("utf-8")
    cam_states_dict = json.loads(json_str)
    cam2world = cam_states_dict['cam2world']
    if isinstance(cam2world, list) and all(isinstance(row, list) for row in cam2world):
        cam2world_tuple = tuple(tuple(element for element in row) for row in cam2world)
        translational_component = np.array(cam2world_tuple)[:3, 3]
        translational_norm = np.linalg.norm(translational_component)
        sign_vector = create_sign_vector(translational_component)
        directed_norm = translational_norm *sign_vector[0] *sign_vector[1] *sign_vector[2]
        return round(directed_norm, 3)
    else:
        raise ValueError("cam2world is not in the expected format")

map = {}
point_list = [
    (95, 180), (410, 180), (240, 80), (120, 170), (140, 160), (150, 150),
    (170, 140), (180, 130), (200, 120), (210, 110), (220, 110), (390, 170),
    (380, 160), (350, 150), (350, 150), (340, 140), (320, 120), (300, 110),
    (290, 100)
]

novel_views_dir = "/path/to/novel_views/R/B07B4D499R"
for i, point in enumerate(point_list):
    hdf5_file = os.path.join(novel_views_dir, f"{i}.hdf5")
    cam2world_key = get_cam2world_key(hdf5_file)
    print(cam2world_key)
    map[cam2world_key] = {"point": point, "ratio_w": 0.7, "ratio_h": 0.7, "floor_path":f"{i}.png"}

with open('cam_pose_map.json', 'w') as json_file:
    json.dump({str(k): v for k, v in map.items()}, json_file, indent=4)
