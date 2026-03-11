"""Run PATE-GAN to generate synthetic data only (no downstream evaluation)."""

import argparse
from pathlib import Path

import pandas as pd

from pate_gan import pategan

REPO_ROOT = Path(__file__).resolve().parents[1]
DIABETIC_DATA_PATH = REPO_ROOT / "thesis" / "data" / "diabetic_data_preprocessed_test.csv"
OUTPUT_PATH = REPO_ROOT / "thesis" / "data" / "pategan" / "diabetic_data_pategan_test_synthetic.csv"


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_no", type=int, default=57214, help="Number of rows to use from input.")
  parser.add_argument("--n_s", type=int, default=1, help="Number of student iterations.")
  parser.add_argument("--batch_size", type=int, default=64, help="Batch size for student/generator.")
  parser.add_argument("--k", type=int, default=10, help="Number of teacher models.")
  parser.add_argument("--epsilon", type=float, default=1.0, help="DP epsilon.")
  parser.add_argument("--delta", type=float, default=0.00001, help="DP delta.")
  parser.add_argument("--lamda", type=float, default=1.0, help="PATE noise size.")
  return parser.parse_args()


def main():
  args = parse_args()

  df = pd.read_csv(DIABETIC_DATA_PATH)
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

  synthetic = pategan(df.to_numpy(), parameters)

  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(synthetic, columns=df.columns).to_csv(OUTPUT_PATH, index=False)

  print(f"Saved synthetic data to: {OUTPUT_PATH}")
  print(f"Synthetic shape: {synthetic.shape}")


if __name__ == "__main__":
  main()
