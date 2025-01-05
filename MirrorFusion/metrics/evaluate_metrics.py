import autoroot
import autorootcwd
import os
import argparse
import h5py
import logging
import numpy as np
import pandas as pd
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from examples.brushnet.dataset.dataset import HDF5Dataset
from metrics.metrics import MetricsCalculator
from tqdm.auto import tqdm


logger = get_logger(__name__)


full_metrics = ["PSNR", "LPIPS", "SSIM"] # on entire image
object_metrics = ["obj_PSNR", "obj_LPIPS", "obj_SSIM"] # over the object reflection mask
mirror_metrics = ["mirror_PSNR", "mirror_LPIPS", "mirror_SSIM"]  # over the mirror region between gt and predicted
mask_metrics = ["mask_PSNR", "mask_LPIPS", "mask_SSIM"] # for image preservation
text_align_metrics = ["CLIP_Similarity"] # for text alignment
img_quality_metrics = ["Image_Reward", "HPS_V2.1", "Aesthetic_Score"] # for image quality
selection_metrics = ["mask_SSIM", "mask_PSNR", "mask_LPIPS"] # plausible metric that can be used for selecting best image
reflection_metrics = ["IoU"] # for checking the region of reflection
all_metrics = (
    full_metrics
    + object_metrics
    + mirror_metrics
    + mask_metrics
    + reflection_metrics
    + text_align_metrics
    + img_quality_metrics
)

# define columns of the eval csv's
columns = ["uid"]
columns += all_metrics


# a dict of {metric_name: metric lambda (max or min)} to return lambda upon metric name
metric_lambda_dict = {
    "PSNR": max,
    "LPIPS": min,
    "SSIM": max,
    "obj_PSNR": max,
    "obj_LPIPS": min,
    "obj_SSIM": max,
    "mirror_PSNR": max,
    "mirror_LPIPS": min,
    "mirror_SSIM": max,
    "mask_PSNR": max,
    "mask_LPIPS": min,
    "mask_SSIM": max,
    "IoU": max,
    "CLIP_Similarity": max,
    "Image_Reward": max,
    "HPS_V2.1": max,
    "Aesthetic_Score": max,
}


def get_uids_and_eval_df(args):
    """
    creates a list of df's of length args.num_images_per_validation with columns as columns
    and reads them if they exist in the infer_dir with index as suffix to args.output_csv name.
    returns uids and eval_dfs
    """
    eval_dfs = []
    uids = os.listdir(args.infer_dir)
    # filter those paths that end with .png and remove the extension
    uids = [uid.split(".")[0] for uid in uids if uid.endswith(".png")] # ex. 0ca5fee159e048a9abdab6d835137ff9_1
    for i in range(args.num_images_per_validation):
        eval_csv = os.path.join(args.infer_dir, f"{args.output_csv}_{i}.csv")
        if not args.overwrite and os.path.exists(eval_csv):
            print(f"Reading existing {eval_csv}")
            eval_df = pd.read_csv(eval_csv)
            # add missing columns with NaN values
            for col in columns:
                if col not in eval_df.columns:
                    eval_df[col] = float("nan")
        else:
            eval_df = pd.DataFrame(columns=columns)
            eval_df["uid"] = uids
        eval_dfs.append(eval_df)
    return uids, eval_dfs


def get_metrics_to_compute(args):
    """returns a list of metrics to compute in this run"""
    metrics_to_compute = []
    for m in args.metrics:
        if m == "all":
            metrics_to_compute = all_metrics
            break
        elif m == "full":
            metrics_to_compute += full_metrics
        elif m == "object":
            metrics_to_compute += object_metrics
        elif m == "mirror":
            metrics_to_compute += mirror_metrics
        elif m == "mask":
            metrics_to_compute += mask_metrics
        elif m == "text_align":
            metrics_to_compute += text_align_metrics
        elif m == "img_quality":
            metrics_to_compute += img_quality_metrics
        elif m in all_metrics:
            metrics_to_compute.append(m)
    return metrics_to_compute


def transform_uid(df):
    """
    Transforms the uid column of df by appending the cam_id.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'uid' and 'path' columns.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    path = df["path"].str.split("/").str[-1]
    df["uid"] = df["uid"] + '_' + path.str.split(".").str[0]
    return df


def split_generated_image(args, gen_image):
    """
    Splits the generated image into args.num_images_per_validation sub images.
    For args.num_images_per_validation = 4, the gen_image looks like this:
    0 1
    2 3
    we want [0, 1, 2, 3] as output sub images.
    for 6,
    0 1 2
    3 4 5
    we want [0, 1, 2, 3, 4, 5] as output sub images.
    so gen_image has two rows of images that are stacked.
    Args:
        args: arguments
        gen_image (PIL image): The generated image.

    Returns:
        list: A list of PIL sub images.
    """
    w, h = gen_image.size
    sub_images = []
    for i in range(args.num_images_per_validation):
        x = (i % 2) * w // 2
        y = (i // 2) * h // 2
        sub_images.append(gen_image.crop((x, y, x + w // 2, y + h // 2)))
    return sub_images


def save_dfs(args, eval_dfs, gpu_id):
    """
    Saves the eval_dfs to csv files in the infer_dir with index as suffix to args.output_csv name and gpu_id.
    Args:
        args
        eval_dfs: list of dfs to save
        gpu_id: the gpu id of the current process
    """
    for i, eval_df in enumerate(eval_dfs):
        eval_csv = os.path.join(args.infer_dir, f"{args.output_csv}_{i}_{gpu_id}.csv")
        eval_df.to_csv(eval_csv, index=False)


def merge_csv_files(args, delete_intermediate=False):
    """
    Merges the csv files with unique suffixes into the final output csv files.
    Args:
        args
    """
    for i in range(args.num_images_per_validation):
        final_csv = os.path.join(args.infer_dir, f"{args.output_csv}_{i}.csv")
        dfs = []
        for file in os.listdir(args.infer_dir):
            if file.startswith(f"{args.output_csv}_{i}_") and file.endswith(".csv"):
                dfs.append(pd.read_csv(os.path.join(args.infer_dir, file)))
                if delete_intermediate:
                    # delete the csv files with gpu_id suffix
                    os.remove(os.path.join(args.infer_dir, file))
        if dfs:
            final_df = dfs[0]
            for df in dfs[1:]:
                final_df = final_df.combine_first(df)
            final_df.to_csv(final_csv, index=False)


def get_row_index_by_uid(df, uid):
    """
    Returns the row index of the DataFrame where the uid matches.
    """
    return df[df["uid"] == uid].index[0]


def get_best_df_index(dfs, row_idx, select_metric_name):
    """
    Returns the index of the DataFrame with the best metric value for a given row_idx,
    """
    metric_values = [df.at[row_idx, select_metric_name] if not pd.isna(df.at[row_idx, select_metric_name]) \
                     else (float('-inf') if metric_lambda_dict[select_metric_name] == max else float('inf')) for df in dfs]

    # use argmax if metric_lambda_dict[select_metric_name] is max else use argmin
    best_idx = np.argmax(metric_values) if metric_lambda_dict[select_metric_name] == max else np.argmin(metric_values)

    return best_idx


def evaluate_metric_for_row_index(
    df,
    row_index,
    metrics_calculator,
    metric_name,
    gen_image,
    gt_data,
    caption,
):
    """
    Evaluates the metric for a given row index in the DataFrame.
    """
    metric_na = df.at[row_index, metric_name] is None or pd.isna(df.at[row_index, metric_name])
    if metric_na:
        metric_result = metrics_calculator.compute_metric(
            metric_name, gen_image, gt_data, caption
        )
        df.at[row_index, metric_name] = metric_result


def check_select_metric_exists(args, eval_dfs):
    """
    Check if select_metric exists and is not nan in all eval_dfs
    """
    for i, eval_df in enumerate(eval_dfs):
        if args.select_metric not in eval_df.columns:
            raise ValueError(f"{args.select_metric} not found in {args.output_csv}_{i}.csv")
        if eval_df[args.select_metric].isnull().values.any():
            # print the uid with NaN value
            nan_uids = eval_df[eval_df[args.select_metric].isnull()]["uid"].values
            raise ValueError(f"{args.select_metric} column has NaN values in {args.output_csv}_{i}.csv, on uids: {nan_uids}")


def calculate_best_metrics_df(args):
    """
    Calculates the best metrics for each uid based on the select_metric and saves the best metrics df
    """
    best_metrics_csv_path = os.path.join(args.infer_dir, f"{args.output_csv}_best.csv")

    eval_csv_files = [f"{args.output_csv}_{i}.csv" for i in range(args.num_images_per_validation)]

    if not all(os.path.exists(os.path.join(args.infer_dir, csv_file)) for csv_file in eval_csv_files):
        raise ValueError(f"Missing eval csv files in {args.infer_dir}")

    eval_dfs = [pd.read_csv(os.path.join(args.infer_dir, csv_file)) for csv_file in eval_csv_files]
    check_select_metric_exists(args, eval_dfs)
    best_df_cols = eval_dfs[0].columns.to_list() + ["select_img_index"]
    metric_cols = [col for col in best_df_cols if col in all_metrics]

    best_df = pd.DataFrame(columns=best_df_cols)
    uids = eval_dfs[0]["uid"].values

    for i, uid in enumerate(uids):
        best_df_idx = get_best_df_index(eval_dfs, i, args.select_metric)
        best_df.at[i, "select_img_index"] = int(best_df_idx)
        best_df.at[i, "uid"] = uid
        for metric_name in metric_cols:
            best_df.at[i, metric_name] = eval_dfs[best_df_idx].at[i, metric_name]

    best_df.to_csv(best_metrics_csv_path, index=False)
    return best_df


def calculate_avg_df(best_df=None):
    """
    Calculates the average of the dataset with best selected metrics among all eval dfs.
    """
    if best_df is None:
        best_csv_file = os.path.join(args.infer_dir, f"{args.output_csv}_best.csv")
        best_df = pd.read_csv(best_csv_file, usecols=all_metrics)
    else:
        best_df = best_df.reindex(columns=all_metrics, fill_value=0)
    avg_values = best_df.astype(float).mean()
    avg_stats_df = pd.DataFrame({"Metric": avg_values.index, "Dataset Average": avg_values.values})
    average_csv = os.path.join(args.infer_dir, f"{args.output_csv}_avg.csv")
    print(avg_stats_df.to_string(index=False))
    avg_stats_df.to_csv(average_csv, index=False)


def main(args):

    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    test_df = pd.read_csv(os.path.join(args.train_data_dir, args.csv))
    test_df = transform_uid(test_df)
    # which metrics to compute in this run.
    metrics_to_compute = get_metrics_to_compute(args)
    # get uids and eval_dfs
    uids, eval_dfs = get_uids_and_eval_df(args)

    logger.info(f"metrics to compute: {metrics_to_compute}")

    if args.mode == "best":
        check_select_metric_exists(args, eval_dfs)

    metrics_calculator = MetricsCalculator(
        metrics_to_compute=metrics_to_compute,
        device=accelerator.device,
        data_dir=args.train_data_dir,
        cache_dir=args.cache_dir
    )

    gpu_id = accelerator.local_process_index
    progress_bar = tqdm(range(len(uids)), disable=not accelerator.is_local_main_process)
    with accelerator.split_between_processes(uids) as uids_split:
        for uid in uids_split:
            try:
                row = test_df[test_df["uid"] == uid].iloc[0]
                rel_path = str(row["path"])
                caption = args.mirror_prompt + str(row[args.captions_column])
                hdf5_path = os.path.join(args.train_data_dir, rel_path)
                gen_image_path = os.path.join(args.infer_dir, f"{uid}.png")
                hdf5_data = h5py.File(hdf5_path, "r")
                gt_data = HDF5Dataset.extract_data_from_hdf5(hdf5_data)
                gt_data["file_path"] = rel_path # ex. abo_v3/B/B07HSLGGDB/0.hdf5

                gen_image = Image.open(gen_image_path)
                gen_images = split_generated_image(args, gen_image)

                for metric_name in metrics_to_compute:
                    # compute non selection metrics only for image with best args.select_metric from eval_dfs among gen_images
                    if (
                        args.mode == "best"
                    ):  # This assumes that `select_metric` has been computed before this run for all eval dfs
                        row_index = get_row_index_by_uid(eval_dfs[0], uid)
                        best_df_index = get_best_df_index(eval_dfs, row_index, args.select_metric)
                        evaluate_metric_for_row_index(
                            eval_dfs[best_df_index],
                            row_index,
                            metrics_calculator,
                            metric_name,
                            gen_images[best_df_index],
                            gt_data,
                            caption,
                        )
                        continue

                    for i, gen_image in enumerate(gen_images):
                        row_index = get_row_index_by_uid(eval_dfs[i], uid)
                        evaluate_metric_for_row_index(
                            eval_dfs[i],
                            row_index,
                            metrics_calculator,
                            metric_name,
                            gen_image,
                            gt_data,
                            caption,
                        )
            except FileNotFoundError:
                logger.error(f"Inference Image {gen_image_path} not found. Skipping evaluation.")
            except Exception as e:
                logger.error(f"Error processing image {gen_image_path}. Skipping evaluation.")
                logger.error(e)
            progress_bar.update(accelerator.num_processes)

    # Save the eval_dfs with a unique suffix for each process
    save_dfs(args, eval_dfs, gpu_id)

    # wait for all processes to finish
    accelerator.wait_for_everyone()

    # Merge the CSV files after all processes have finished
    if accelerator.is_main_process:
        logger.info("Merging CSV files")
        merge_csv_files(args, delete_intermediate=True)


def parser():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--csv", type=str, default="test.csv", help="The csv file containing the test data")
    parser.add_argument(
        "--captions_column",
        type=str,
        default="auto_caption",
        help="The column to use for getting captions from test.csv",
        choices=["caption", "auto_caption"],
    )
    parser.add_argument(
        "--mirror_prompt", type=str, default="A perfect plane mirror reflection of ", help="The mirror prompt to use."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="sam_cache",
        help="Directory where the segmented images will be cached",
    )
    parser.add_argument(
        "--infer_dir",
        type=str,
        default="runs/logs/sd15_full/checkpoint-20000/inference",
        help="Directory of the output inference images",
    )
    parser.add_argument("--resolution", type=int, default=512, help="The resolution of pipeline image. (default: 512)")
    parser.add_argument(
        "--num_images_per_validation",
        type=int,
        default=4,
        help="Number of seeds used in inference images per validation image. (default: 4)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mask"],
        choices=["all", "full", "object", "mirror", "mask", "text_align", "img_quality"] + all_metrics,
        help="Metrics to calculate. Options: [all, full, object, mask, text_align, img_quality, or individual names]",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="calc",
        help="script mode to run on. calc: calculate for all inference image seeds. \
        best: only calculate based on `select_metric`, avg: calculate average metrics df. all: calculate all.",
        choices=["calc", "best", "avg"],
    )
    parser.add_argument("--select_metric", type=str, default="mask_SSIM", help="The metric to use for selecting best image")
    parser.add_argument("--output_csv", type=str, default="eval", help="base name of output csv file. Will be stored inside the infer_dir.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite the existing eval csv's.")
    return parser


if __name__ == "__main__":
    # Check the README for the usage of this script

    parser = parser()
    args = parser.parse_args()
    if args.mode == "best":
        if args.select_metric not in selection_metrics:
            raise ValueError(f"select_metric {args.select_metric} not in {selection_metrics}")
    if args.mode == "avg":
        best_df = calculate_best_metrics_df(args)
        calculate_avg_df(best_df)
    else:
        main(args)
