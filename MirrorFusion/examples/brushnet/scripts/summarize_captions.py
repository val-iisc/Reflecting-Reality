"""
Summarize the captions in a csv file using the distilbart-cnn-6-6 model from the transformers library.
"""
# %load_ext cudf.pandas
import autoroot
import autorootcwd
import argparse
from pathlib import Path

import torch
import pandas as pd
from transformers import pipeline


def main(args):
    print(f'Summarizing captions in {Path(args.data_root) / args.csv}')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(Path(args.data_root) / args.csv)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=device)
    df["auto_caption"] = df["auto_caption"].apply(
        lambda x: summarizer(x, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]
    )
    df.to_csv(Path(args.data_root) / args.csv, index=False)
    print(f'Summarized captions saved to {Path(args.data_root) / args.csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        help="directory containing the csv files",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="test.csv",
        help="csv file to summarize",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="auto_caption",
        help="column containing the caption to summarize",
    )
    args = parser.parse_args()
    main(args)
