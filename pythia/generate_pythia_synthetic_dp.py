"""Run class-conditioned Pythia synthetic generation for train/test splits."""
#comment
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Force line-buffered stdout so logs stream live on VMs / remote terminals.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

try:
    from pythia.pythia_tabular_dp import (
        DPConfig,
        file_sha256,
        generate_synthetic_for_split,
        set_global_seed,
        stats_to_dict,
    )
except ImportError:
    from pythia_tabular_dp import (  # type: ignore
        DPConfig,
        file_sha256,
        generate_synthetic_for_split,
        set_global_seed,
        stats_to_dict,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "thesis" / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "pythia"
DEFAULT_HF_TOKEN_FILE = REPO_ROOT / "pythia" / "token.txt"


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

    parser.add_argument(
        "--dp",
        action="store_true",
        help="Enable differentially-private LoRA fine-tuning (DP-SGD via Opacus).",
    )
    parser.add_argument("--target-epsilon", type=float, default=5.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dp-per-device-batch-size", type=int, default=32)
    parser.add_argument("--dp-grad-accum-steps", type=int, default=16)

    return parser.parse_args()


def build_dp_config(args: argparse.Namespace) -> Optional[DPConfig]:
    if not args.dp:
        return None
    return DPConfig(
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.max_grad_norm,
        per_device_batch_size=args.dp_per_device_batch_size,
        gradient_accumulation_steps=args.dp_grad_accum_steps,
    )


def resolve_split_paths(args: argparse.Namespace) -> Dict[str, Path]:
    default_train = DATA_DIR / "diabetic_data_preprocessed_train.csv"
    default_test = DATA_DIR / "diabetic_data_preprocessed_test.csv"
    return {
        "train": Path(args.train_csv) if args.train_csv else default_train,
        "test": Path(args.test_csv) if args.test_csv else default_test,
    }


def output_path_for_split(output_dir: Path, split: str, dp_enabled: bool = False) -> Path:
    suffix = "_dp" if dp_enabled else ""
    return output_dir / f"diabetic_data_pythia_{split}{suffix}_synthetic.csv"


def summarize_class_counts(df: pd.DataFrame, target_col: str) -> Dict[str, int]:
    target = pd.to_numeric(df[target_col], errors="coerce").round().astype(int)
    return {
        "0": int((target == 0).sum()),
        "1": int((target == 1).sum()),
    }


def load_hf_token(token_file: Path = DEFAULT_HF_TOKEN_FILE) -> Optional[str]:
    """Load Hugging Face token from token file, fallback to environment."""
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            return token

    env_token = os.environ.get("HF_TOKEN", "").strip()
    return env_token or None


def _print_env_info() -> None:
    """Print runtime / device info so logs make clear what hardware is in use."""
    try:
        import torch
    except Exception as e:
        print(f"[env] torch import failed: {e}", flush=True)
        return

    print(f"[env] python={sys.version.split()[0]} torch={torch.__version__}", flush=True)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[env] device=cuda ({name}, {total_gb:.1f} GiB), device_count={torch.cuda.device_count()}", flush=True)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[env] device=mps (Apple Silicon)", flush=True)
    else:
        print("[env] device=cpu (no GPU detected)", flush=True)


def main() -> None:
    args = parse_args()

    # Pin to GPU 1 before any CUDA calls so all allocations go there.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    set_global_seed(args.seed)
    hf_token = load_hf_token()

    _print_env_info()

    split_paths = resolve_split_paths(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dp_config = build_dp_config(args)

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
            "dp_enabled": bool(dp_config is not None),
            "dp": (
                {
                    "target_epsilon": dp_config.target_epsilon,
                    "target_delta": dp_config.target_delta,
                    "max_grad_norm": dp_config.max_grad_norm,
                    "per_device_batch_size": dp_config.per_device_batch_size,
                    "gradient_accumulation_steps": dp_config.gradient_accumulation_steps,
                    "effective_batch_size": dp_config.effective_batch_size,
                }
                if dp_config is not None
                else None
            ),
        },
        "splits": {},
    }

    for idx, split_name in enumerate(args.splits):
        input_path = split_paths[split_name]
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV for split '{split_name}' not found: {input_path}")

        print(f"\n=== Running split: {split_name} ===", flush=True)
        print(f"Source: {input_path}", flush=True)
        split_t0 = time.time()

        source_df = pd.read_csv(input_path)
        if args.row_limit is not None:
            if args.row_limit <= 0:
                raise ValueError("--row-limit must be a positive integer.")
            source_df = source_df.head(args.row_limit).copy()
            print(f"Row limit applied: {len(source_df)}", flush=True)

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
            hf_token=hf_token,
            dp_config=dp_config,
        )

        out_path = output_path_for_split(output_dir, split_name, dp_enabled=dp_config is not None)
        print(f"[split:{split_name}] writing CSV -> {out_path}", flush=True)
        synthetic_df.to_csv(out_path, index=False)

        print(f"Saved synthetic {split_name} split: {out_path}", flush=True)
        print(f"Rows: {len(synthetic_df)}", flush=True)
        print(f"Class counts: {summarize_class_counts(synthetic_df, args.target_col)}", flush=True)
        print(f"[split:{split_name}] total elapsed: {time.time() - split_t0:.1f}s", flush=True)

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

    metadata_filename = "run_metadata_dp.json" if dp_config is not None else "run_metadata.json"
    metadata_path = output_dir / metadata_filename
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nMetadata saved: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
