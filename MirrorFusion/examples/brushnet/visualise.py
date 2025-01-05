"""
This script visualises the inference images and their metrics (after the csv files are generated) using FiftyOne.
"""
import autoroot
import autorootcwd
# import cudf.pandas
# cudf.pandas.install()
import argparse
import os
import json

import fiftyone as fo
import pandas as pd
from tqdm import tqdm


def transform_uid(df):
    """
    Transforms the uid column of df by appending the cam_id.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'uid' and 'path' columns.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    path = df["path"].str.split("/").str[-1]
    df["uid"] = df["uid"] + "_" + path.str.split(".").str[0]
    return df


def get_row_index_by_uid(df, uid):
    """
    Returns the row index of the DataFrame where the uid matches.
    """
    return df[df["uid"] == uid].index[0]


def add_metric_label(sample: fo.Sample, eval_df: pd.DataFrame, row_index: int, label_suffix):
    """
    Adds metric labels to the sample based on the evaluation DataFrame and returns the sample.
    """

    for col in eval_df.columns:
        # TODO handle issue with HPSV_2.1 (currently unused)
        if col in ["uid", "path", "select_img_index", "HPS_V2", "HPS_V2.1"]:
            continue
        label_name = f"{col}_{label_suffix}"  # eg. SSIM_0, LPIPS_1, PSNR_best
        value = (
            float(eval_df.iloc[row_index][col])
            if eval_df.iloc[row_index][col] not in [None, ""] and not pd.isna(eval_df.iloc[row_index][col])
            else float("nan")
        )
        sample.set_field(label_name, value)

    if "select_img_index" in eval_df.columns:
        sample.set_field("select_img_index", int(eval_df.iloc[row_index]["select_img_index"]))

    return sample


def declare_fields(dataset: fo.Dataset, eval_df, label_suffix):
    for col in eval_df.columns:
        if col in ["uid", "path", "select_img_index", "HPS_V2", "HPS_V2.1"]:
            continue
        label_name = f"{col}_{label_suffix}"
        if label_name not in dataset.get_field_schema():
            dataset.add_sample_field(label_name, fo.FloatField)

    if "select_img_index" not in dataset.get_field_schema():
        dataset.add_sample_field("select_img_index", fo.IntField)


def add_tags_and_labels(
    fiftyone_dataset: fo.Dataset,
    sample: fo.Sample,
    uid: str,
    uid_category_map,
    test_df: pd.DataFrame,
    eval_dfs=[],
    best_df=None,
):
    """add tags, labels and fields to the sample for better visualisation"""

    row = test_df[test_df["uid"] == uid].iloc[0]

    base_uid = uid.split("_")[0] # remove the cam_id
    sample.tags.append(uid_category_map[base_uid])

    # add sample fields
    sample["manual_caption"] = row["caption"]
    sample["automatic_caption"] = row["auto_caption"]

    is_novel = bool(row["is_novel"])
    is_small = "small_mirrors" in row["path"]
    is_abo = "abo" in row["path"]

    # add data-subset tags
    if is_novel:
        sample.tags.append("novel")
    if is_small:
        sample.tags.append("small-mirrors")
    if is_abo:
        sample.tags.append("abo")

    # add metric labels
    for i, eval_df in enumerate(eval_dfs):
        declare_fields(fiftyone_dataset, eval_df, i)
        row_index = get_row_index_by_uid(eval_df, uid)
        sample = add_metric_label(sample, eval_df, row_index, i)

    if best_df is not None:
        declare_fields(fiftyone_dataset, best_df, "best")
        row_index = get_row_index_by_uid(best_df, uid)
        sample = add_metric_label(sample, best_df, row_index, "best")

    return sample


def read_eval_csvs(args, infer_dir):
    eval_csv_basename = args.eval_csv
    eval_csv_paths = [os.path.join(infer_dir, f"{eval_csv_basename}_{i}.csv") for i in range(args.num_images_per_validation)]
    eval_dfs = []
    best_df = None
    for eval_csv_path in eval_csv_paths:
        if not os.path.exists(eval_csv_path):
            print(f"Warning: {eval_csv_path} does not exist. Skipping.")
            continue
        eval_df = pd.read_csv(eval_csv_path)
        eval_dfs.append(eval_df)

    if os.path.exists(os.path.join(infer_dir, f"{eval_csv_basename}_best.csv")):
        best_df = pd.read_csv(os.path.join(infer_dir, f"{eval_csv_basename}_best.csv"))

    if os.path.exists(os.path.join(infer_dir, f"{eval_csv_basename}_avg.csv")):
        avg_df = pd.read_csv(os.path.join(infer_dir, f"{eval_csv_basename}_avg.csv"))
        print(f'Average metrics (over best_df) for {infer_dir}:')
        print(avg_df.to_string(index=False))

    return eval_dfs, best_df


def remove_strings(main_string, strings_to_remove):
    for s in strings_to_remove:
        main_string = main_string.replace(s, "")
    # remove trainling slash and return main string
    if main_string.endswith("/"):
        main_string = main_string[:-1]
    return main_string


def main(args):
    test_df = pd.read_csv(os.path.join(args.train_data_dir, args.csv))
    test_df = transform_uid(test_df)

    uid_category_map = {}
    # read abo classes
    abo_classes_path = os.path.join(args.train_data_dir, args.abo_classes)
    with open(abo_classes_path, "r") as f:
        for line in f:
            line = line.strip().split(',')
            uid_category_map[line[0]] = line[1]

    # read objaverse classes json file
    objaverse_classes_path = os.path.join(args.train_data_dir, args.objaverse_classes)
    with open(objaverse_classes_path, "r") as f:
        objaverse_manual_captions = json.load(f)
        for uid, _ in objaverse_manual_captions.items():
            uid_category_map[uid] = objaverse_manual_captions[uid]["category"]

    infer_dirs = []
    if args.all_ckpt:
        root_dir = args.infer_dir[0]
        ckpts = os.listdir(root_dir)
        ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("checkpoint")]
        ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
        for ck in ckpts:
            infer_dir_path = os.path.join(root_dir, ck, "inference")
            if os.path.exists(infer_dir_path) and os.path.isdir(infer_dir_path):
                infer_dirs.append(infer_dir_path)
    else:
        infer_dirs = args.infer_dir

    for infer_dir in infer_dirs:
        counter = 0
        print(f'visualising {infer_dir}')
        eval_dfs, best_df = read_eval_csvs(args, infer_dir)
        dataset_name = remove_strings(infer_dir, ["runs/logs/", "inference"]) + f"_{args.port}"
        dataset = fo.Dataset(dataset_name, overwrite=args.overwrite)
        sorted_paths = sorted(os.listdir(infer_dir))
        for img_path in tqdm(sorted_paths):
            if not img_path.endswith(".png"):
                continue
            if args.limit is not None and counter >= args.limit:
                break
            uid = img_path.split(".")[0]
            sample = fo.Sample(filepath=os.path.join(infer_dir, img_path))
            sample = add_tags_and_labels(dataset, sample, uid, uid_category_map, test_df, eval_dfs, best_df)
            dataset.add_sample(sample)
            counter += 1

    session = fo.launch_app(port=args.port, remote=True) # select the dataset from the desktop app
    session.wait()


if __name__ == "__main__":
    # python examples/brushnet/visualise.py --infer_dir runs/logs/<ckpt_name>/inference dir2
    # python examples/brushnet/visualise.py --all_ckpt
    parser = argparse.ArgumentParser(description="Visualise inference images")
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
        "--num_images_per_validation",
        type=int,
        default=4,
        help="Number of seeds used in inference images per validation image. (default: 4)",
    )
    parser.add_argument("--csv", type=str, default="test.csv")
    parser.add_argument(
        "--infer_dir",
        type=str,
        nargs="+",
        default=["runs/logs/sd15_full"],
        help="Inference directories",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="port to run the FiftyOne app on",
    )
    parser.add_argument(
        "--all_ckpt",
        action="store_true",
        help="Whether to visualise all checkpoints in a directory. In this case, `infer_dir`[0] Is the root directory of checkpoints.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the existing dataset with the same name",
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default="eval",
        help="base name of eval csv files. Default: eval. files should be stored under the infer_dir. [eval_0.csv, eval_1.csv, ..., eval_best.csv]",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to visualise")
    parser.add_argument(
        "--abo_classes",
        type=str,
        default="abo_metadata/metadata/abo_classes_3d.txt",
        help="relative path to abo classes file from `train_data_dir`. \
        Get abo metadata from https://github.com/jazcollins/amazon-berkeley-objects/tree/main/metadata",
    )
    parser.add_argument(
        "--objaverse_classes",
        type=str,
        default="objaverse_cat_descriptions_64k.json",
        help="relative path to objaverse classes file from `train_data_dir`. \
        https://github.com/allenai/object-edit/blob/main/objaverse_cat_descriptions_64k.json",
    )
    args = parser.parse_args()
    main(args)
