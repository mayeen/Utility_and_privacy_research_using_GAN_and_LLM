"""Run class-conditioned Pythia synthetic generation for train/test splits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

try:
    from pythia.pythia_tabular import (
        file_sha256,
        generate_synthetic_for_split,
        set_global_seed,
        stats_to_dict,
    )
except ImportError:
    from pythia_tabular import (  # type: ignore
        file_sha256,
        generate_synthetic_for_split,
        set_global_seed,
        stats_to_dict,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "thesis" / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "pythia"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic tabular data using Pythia + LoRA.")

    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])

    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--target-col", type=str, default="readmitted")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries-per-row", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=None)

    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Optional row cap per split for quick smoke runs.",
    )

    return parser.parse_args()


def resolve_split_paths(args: argparse.Namespace) -> Dict[str, Path]:
    default_train = DATA_DIR / "diabetic_data_preprocessed_train.csv"
    default_test = DATA_DIR / "diabetic_data_preprocessed_test.csv"
    return {
        "train": Path(args.train_csv) if args.train_csv else default_train,
        "test": Path(args.test_csv) if args.test_csv else default_test,
    }


def output_path_for_split(output_dir: Path, split: str) -> Path:
    return output_dir / f"diabetic_data_pythia_{split}_synthetic.csv"


def summarize_class_counts(df: pd.DataFrame, target_col: str) -> Dict[str, int]:
    target = pd.to_numeric(df[target_col], errors="coerce").round().astype(int)
    return {
        "0": int((target == 0).sum()),
        "1": int((target == 1).sum()),
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    split_paths = resolve_split_paths(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "pythia_lora_tabular_generation",
        "model_name": args.model_name,
        "parameters": {
            "splits": args.splits,
            "target_col": args.target_col,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_retries_per_row": args.max_retries_per_row,
            "generation_batch_size": args.generation_batch_size,
            "row_limit": args.row_limit,
        },
        "splits": {},
    }

    for idx, split_name in enumerate(args.splits):
        input_path = split_paths[split_name]
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV for split '{split_name}' not found: {input_path}")

        print(f"\n=== Running split: {split_name} ===")
        print(f"Source: {input_path}")

        source_df = pd.read_csv(input_path)
        if args.row_limit is not None:
            if args.row_limit <= 0:
                raise ValueError("--row-limit must be a positive integer.")
            source_df = source_df.head(args.row_limit).copy()
            print(f"Row limit applied: {len(source_df)}")

        if args.target_col not in source_df.columns:
            raise ValueError(f"Target column '{args.target_col}' missing from split '{split_name}'.")

        synthetic_df, split_stats = generate_synthetic_for_split(
            split_name=split_name,
            source_df=source_df,
            model_name=args.model_name,
            target_col=args.target_col,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            max_retries_per_row=args.max_retries_per_row,
            seed=args.seed + idx,
            generation_batch_size=args.generation_batch_size,
        )

        out_path = output_path_for_split(output_dir, split_name)
        synthetic_df.to_csv(out_path, index=False)

        print(f"Saved synthetic {split_name} split: {out_path}")
        print(f"Rows: {len(synthetic_df)}")
        print(f"Class counts: {summarize_class_counts(synthetic_df, args.target_col)}")

        metadata["splits"][split_name] = {
            "input_path": str(input_path),
            "input_sha256": file_sha256(input_path),
            "output_path": str(out_path),
            "output_sha256": file_sha256(out_path),
            "source_rows": int(len(source_df)),
            "source_class_counts": summarize_class_counts(source_df, args.target_col),
            "synthetic_rows": int(len(synthetic_df)),
            "synthetic_class_counts": summarize_class_counts(synthetic_df, args.target_col),
            "generation_stats": stats_to_dict(split_stats),
        }

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nMetadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
