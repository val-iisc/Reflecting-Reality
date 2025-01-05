"""
This script is used to test the dataset for corrupt data. It checks for the following:
1. Null data in rgb, mask, depth, normals
2. Black images in rgb
3. Mask doesn't have mirror
4. Mask contains fewer than obj_pixels_threshold pixels
5. Depth map is constant
6. Normal map is constant
"""
import autoroot
import autorootcwd
import os
import time
import numpy as np
import h5py
from PIL import Image
import argparse
from multiprocessing import Pool, Manager
from examples.brushnet.dataset.dataset import HDF5Dataset


class ImageException(Exception):
    """Custom exception for null image data."""

    pass


class MaskException(Exception):
    """Custom exception for null image data."""

    pass


class DepthException(Exception):
    """Custom exception for null image data."""

    pass


class NormalsException(Exception):
    """Custom exception for null image data."""

    pass


def convert_to_uint8(data):
    if np.max(data) > 1:
        data = data / 255
    data = (data * 255).astype(np.uint8)
    return data


def handle_exception(exception, lock, cnt, corrupt_uids, cur_uid, data_to_save, output_path, args):
    with lock:
        corrupt_uids.append(cur_uid)
        cnt.value += 1
    print(f"{exception.__class__.__name__}: {exception}")
    if "Null" not in str(exception) and args.output_dir:
        if os.path.exists(output_path):
            print(f"File {output_path} already exists")
            return
        print(f"Extracting {output_path}")
        data_to_save = convert_to_uint8(data_to_save)
        Image.fromarray(data_to_save).save(output_path)


def process_file(args, hdf5_path, cur_uid, output_path, corrupt_uids, cnt, lock):
    try:
        with h5py.File(hdf5_path, "r") as f:
            data = HDF5Dataset.extract_data_from_hdf5(f)
            rgb = data["image"]
            depth = data["depth"]
            normals = data["normals"]
            mask = data["mask"]
            object_mask = data["object_mask"]

            # rgb checks
            if rgb is None:
                raise ImageException(f"rgb Null data found in {hdf5_path}")
            if np.all(rgb <= 5):
                raise ImageException(f"Black image found in {hdf5_path}")

            # mask checks
            if mask is None:
                raise MaskException(f"mask Null data found in {hdf5_path}")
            if np.all(mask == 0):
                raise MaskException(f"Mask doesn't have mirror in {hdf5_path}")
            if np.sum(object_mask == 255) < args.obj_pixels_threshold:
                raise MaskException(
                    f"Mask contains fewer than {args.obj_pixels_threshold} pixels in {hdf5_path}"
                )

            # depth checks
            if depth is None:
                raise DepthException(f"depth Null data found in {hdf5_path}")
            if np.std(depth.reshape(-1)) == 0:
                raise DepthException(f"Depth map is constant in {hdf5_path}")
            bool_mask = mask > 0
            if len(depth[bool_mask]) == 0:
                raise DepthException(f"len(depth[bool_mask]) == 0 in {hdf5_path}")

            # normals checks
            if normals is None:
                raise NormalsException(f"normals Null data found in {hdf5_path}")
            if np.std(normals.reshape(-1)) == 0:
                raise NormalsException(f"Normal map is constant in {hdf5_path}")

    except ImageException as nie:
        handle_exception(nie, lock, cnt, corrupt_uids, cur_uid, rgb, output_path, args)
    except MaskException as mie:
        handle_exception(mie, lock, cnt, corrupt_uids, cur_uid, mask, output_path, args)
    except DepthException as die:
        handle_exception(die, lock, cnt, corrupt_uids, cur_uid, depth, output_path, args)
    except NormalsException as nie:
        handle_exception(nie, lock, cnt, corrupt_uids, cur_uid, normals, output_path, args)
    except Exception as e:
        with lock:
            cnt.value += 1
        print(f"Error {e}, opening {hdf5_path}")
        corrupt_uids.append(cur_uid)


def main(args):
    manager = Manager()
    corrupt_uids = manager.list()
    cnt = manager.Value("i", 0)
    lock = manager.Lock()  # Create a lock object
    start_time = time.perf_counter()

    file_list = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".hdf5"):
                if len(args.ignore_dirs) > 0:
                    if any(dir in root for dir in args.ignore_dirs):
                        continue
                hdf5_path = os.path.join(root, file)
                cur_uid = hdf5_path.split("/")[-2]

                output_path = None
                if args.output_dir:
                    output_subdir = os.path.join(
                        args.output_dir,
                        os.path.relpath(root, args.input_dir),
                    )
                    os.makedirs(output_subdir, exist_ok=True)
                    output_path = os.path.join(output_subdir, f"{file.split('.')[0]}.png")

                file_list.append((args, hdf5_path, cur_uid, output_path, corrupt_uids, cnt, lock))

    num_processes = os.cpu_count()  # Default to the number of CPU cores
    if args.processes:
        num_processes = args.processes
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_file, file_list)

    end_time = time.perf_counter()
    corrupt_uids = set(corrupt_uids)
    print(f"Count of corrupt uids: {len(corrupt_uids)}")
    print(f"Time taken: {end_time - start_time} seconds")
    pwd = os.getcwd()

    if len(corrupt_uids) > 0:
        with open(os.path.join(pwd, args.corrupt_output_file), "w") as f:
            for uid in set(corrupt_uids):
                f.write(f"{uid}\n")


if __name__ == "__main__":
    # python test_dataset.py --input_dir /path/to/data/blenderproc/hf-objaverse-v3 --corrupt_output_file data/blenderproc/corrupt_uids_objaverse-v3.txt
    parser = argparse.ArgumentParser(description="Check HDF5 files for corrupt data")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory",
    )
    parser.add_argument(
        "--corrupt_output_file",
        type=str,
        default="data/blenderproc/corrupt_uids.txt",
        help="Output file with corrupt uids",
    )
    parser.add_argument("--obj_pixels_threshold", type=int, default=10, help="Threshold for object pixels")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes to use")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory of the png renderings")
    parser.add_argument("--ignore_dirs", type=str, nargs="+", default=[], help="Directories to ignore")
    args = parser.parse_args()
    main(args)
