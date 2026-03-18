"""Run PATE-GAN to generate synthetic data for both train and test splits."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pate_gan import pategan

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "thesis" / "data"
OUTPUT_DIR = DATA_DIR / "pategan"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_no", type=int, default=57214, help="Number of rows to use from input.")
    parser.add_argument("--n_s", type=int, default=1, help="Number of student iterations.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for student/generator.")
    parser.add_argument("--k", type=int, default=10, help="Number of teacher models.")
    parser.add_argument("--epsilon", type=float, default=5.0, help="DP epsilon.")
    parser.add_argument("--delta", type=float, default=0.00001, help="DP delta.")
    parser.add_argument("--lamda", type=float, default=1.0, help="PATE noise size.")
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"],
                        help="Which splits to generate (default: both).")
    return parser.parse_args()


def normalize(df):
    """Min-max normalize all columns to [0, 1].

    Returns the normalized DataFrame, column mins, and column maxs.
    Constant columns (range == 0) are set to 0.
    """
    mins = df.min()
    maxs = df.max()
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # avoid division by zero for constant columns
    return (df - mins) / ranges, mins, maxs


def denormalize(synthetic, cols, mins, maxs):
    """Inverse min-max transform: map [0, 1] synthetic data back to original scale.

    Integer columns are detected automatically and rounded.
    Age (discrete float with step 0.5) is snapped to {1.0, 1.5, 2.0}.
    """
    df = pd.DataFrame(np.clip(synthetic, 0, 1), columns=cols)
    df = df * (maxs - mins) + mins

    for col in cols:
        if maxs[col] == mins[col]:  # constant column — leave as-is
            continue
        if col == "age":
            # valid values: 1.0, 1.5, 2.0
            df[col] = (df[col] * 2).round() / 2
        elif float(mins[col]).is_integer() and float(maxs[col]).is_integer():
            # discrete integer column
            df[col] = df[col].round().astype(int).clip(int(mins[col]), int(maxs[col]))

    return df


def generate_split(split_name, args):
    print(f"\n=== Generating {split_name} synthetic data ===")

    input_path = DATA_DIR / f"diabetic_data_preprocessed_{split_name}.csv"
    output_path = OUTPUT_DIR / f"diabetic_data_pategan_{split_name}_synthetic_epsilon_{args.epsilon}.csv"

    df = pd.read_csv(input_path)
    use_rows = min(args.data_no, len(df))
    df = df.iloc[:use_rows].copy()

    parameters = {
        "n_s": args.n_s,
        "batch_size": args.batch_size,
        "k": args.k,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "lamda": args.lamda,
    }

    df_norm, mins, maxs = normalize(df)
    synthetic = pategan(df_norm.to_numpy(), parameters)
    decoded = denormalize(synthetic, df.columns, mins, maxs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    decoded.to_csv(output_path, index=False)

    print(f"Saved synthetic data to: {output_path}")
    print(f"Synthetic shape: {synthetic.shape}")


def main():
    args = parse_args()
    for split_name in args.splits:
        generate_split(split_name, args)


if __name__ == "__main__":
    main()
