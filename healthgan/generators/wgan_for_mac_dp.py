"""DP-WGAN tabular generator (PyTorch + Opacus).

Differential privacy added to the discriminator (critic) via DP-SGD.
The generator never sees real data directly, so its updates inherit DP via
the post-processing theorem.

Gradient penalty (WGAN-GP) is dropped because it requires gradients through
mixed-batch inputs, which breaks Opacus per-sample gradient tracking.
Instead, classic WGAN weight clipping is used on the critic.

CLI mirrors `pythia/generate_pythia_synthetic_dp.py` so results are directly
comparable across pipelines.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "thesis" / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "healthgan"


@dataclass
class DPConfig:
    """Differential-privacy settings (mirrors pythia DPConfig)."""
    target_epsilon: float = 5.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    per_device_batch_size: int = 32
    gradient_accumulation_steps: int = 16

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _print_env_info(device: torch.device) -> None:
    print(f"[env] python={sys.version.split()[0]} torch={torch.__version__}", flush=True)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[env] device=cuda ({name}, {total_gb:.1f} GiB)", flush=True)
    elif device.type == "mps":
        print("[env] device=mps (Apple Silicon)", flush=True)
    else:
        print("[env] device=cpu (no GPU detected)", flush=True)


class Generator(nn.Module):
    def __init__(self, n_features: int, noise_dim: int = 100):
        super().__init__()
        h1 = 2 * n_features
        h2 = round(1.5 * n_features)
        self.net = nn.Sequential(
            nn.Linear(noise_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_features),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, n_features: int, base_nodes: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, base_nodes),
            nn.LeakyReLU(0.2),
            nn.Linear(base_nodes, 2 * base_nodes),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * base_nodes, 4 * base_nodes),
            nn.LeakyReLU(0.2),
            nn.Linear(4 * base_nodes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _weight_clip(module: nn.Module, clip_value: float) -> None:
    """Classic WGAN weight clipping (per parameter)."""
    with torch.no_grad():
        for p in module.parameters():
            p.data.clamp_(-clip_value, clip_value)


class DPWGAN:
    def __init__(
        self,
        train_filepath: str,
        test_filepath: str,
        critic_iters: int = 5,
        base_nodes: int = 64,
        noise_dim: int = 100,
        weight_clip: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        train_df = pd.read_csv(train_filepath)
        test_df = pd.read_csv(test_filepath)

        self.col_names = list(train_df.columns)
        self.n_features = train_df.shape[1]
        self.n_train = train_df.shape[0]
        self.train_data = torch.tensor(train_df.values, dtype=torch.float32)
        self.test_data = torch.tensor(test_df.values, dtype=torch.float32)

        self.critic_iters = critic_iters
        self.base_nodes = base_nodes
        self.noise_dim = noise_dim
        self.weight_clip = weight_clip
        self.device = device or _select_device()

        self.generator = Generator(self.n_features, noise_dim).to(self.device)
        self.discriminator = Discriminator(self.n_features, base_nodes).to(self.device)

        self.history: Dict[str, List[float]] = {
            "gen_loss": [],
            "disc_loss": [],
            "test_disc_loss": [],
            "epoch_time": [],
        }
        self.dp_stats: Optional[Dict] = None

    def _sample_noise(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.noise_dim, device=self.device)

    def train(
        self,
        epochs: int,
        learning_rate: float,
        dp_config: Optional[DPConfig],
        seed: int,
    ) -> None:
        set_global_seed(seed)

        gen_optim = torch.optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9)
        )
        disc_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9)
        )

        # Build DataLoader for the critic. With DP we use Poisson sampling via
        # Opacus; without DP we use a vanilla DataLoader.
        if dp_config is not None:
            logical_bs = dp_config.effective_batch_size
            physical_bs = dp_config.per_device_batch_size
        else:
            logical_bs = max(self.n_train // max(1, self.critic_iters), 1)
            logical_bs = (logical_bs // 100) * 100 if logical_bs >= 100 else logical_bs
            physical_bs = logical_bs

        dataset = TensorDataset(self.train_data)
        dataloader = DataLoader(
            dataset,
            batch_size=logical_bs,
            shuffle=True,
            drop_last=True,
        )

        privacy_engine = None
        memory_manager_ctx = None
        if dp_config is not None:
            from opacus import PrivacyEngine
            from opacus.utils.batch_memory_manager import BatchMemoryManager

            privacy_engine = PrivacyEngine()
            # WGAN runs `critic_iters` critic steps per generator step,
            # so total critic passes over data == epochs * critic_iters.
            dp_epochs = epochs * self.critic_iters
            self.discriminator, disc_optim, dataloader = privacy_engine.make_private_with_epsilon(
                module=self.discriminator,
                optimizer=disc_optim,
                data_loader=dataloader,
                target_epsilon=dp_config.target_epsilon,
                target_delta=dp_config.target_delta,
                max_grad_norm=dp_config.max_grad_norm,
                epochs=dp_epochs,
            )
            print(
                f"[train-dp] dp_epochs(critic)={dp_epochs} per_device_bs={physical_bs} "
                f"grad_accum={dp_config.gradient_accumulation_steps} "
                f"effective_bs={logical_bs} noise_multiplier={disc_optim.noise_multiplier:.4f}",
                flush=True,
            )
            BMM = BatchMemoryManager
        else:
            BMM = None
            print(f"[train] non-DP run, batch_size={logical_bs}", flush=True)

        t0 = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            disc_losses_epoch: List[float] = []
            gen_losses_epoch: List[float] = []

            # Iterate logical batches; each logical batch supplies one critic step.
            # We alternate: every `critic_iters` critic steps -> 1 generator step.
            critic_step_idx = 0

            if BMM is not None:
                loader_ctx = BMM(
                    data_loader=dataloader,
                    max_physical_batch_size=physical_bs,
                    optimizer=disc_optim,
                )
            else:
                from contextlib import nullcontext
                loader_ctx = nullcontext(dataloader)

            with loader_ctx as loader:
                for (real_batch,) in loader:
                    real_batch = real_batch.to(self.device)
                    bs = real_batch.shape[0]

                    # ---- Critic step ----
                    disc_optim.zero_grad(set_to_none=True)
                    noise = self._sample_noise(bs)
                    fake = self.generator(noise).detach()
                    d_real = self.discriminator(real_batch)
                    d_fake = self.discriminator(fake)
                    # WGAN critic loss: maximize E[D(real)] - E[D(fake)]
                    d_loss = d_fake.mean() - d_real.mean()
                    d_loss.backward()
                    disc_optim.step()

                    # Weight clipping (skipped under DP-SGD optimizer wrapper only
                    # if optimizer didn't actually step due to virtual batching).
                    # Opacus signals "real" steps internally; clipping every call
                    # is safe because it just clamps weights.
                    _weight_clip(self.discriminator, self.weight_clip)

                    disc_losses_epoch.append(float(d_loss.detach().cpu()))
                    critic_step_idx += 1

                    # ---- Generator step every `critic_iters` critic steps ----
                    if critic_step_idx % self.critic_iters == 0:
                        gen_optim.zero_grad(set_to_none=True)
                        noise = self._sample_noise(bs)
                        fake = self.generator(noise)
                        # Generator wants to maximize D(fake) -> minimize -D(fake)
                        g_loss = -self.discriminator(fake).mean()
                        g_loss.backward()
                        gen_optim.step()
                        gen_losses_epoch.append(float(g_loss.detach().cpu()))

            epoch_d = float(np.mean(disc_losses_epoch)) if disc_losses_epoch else float("nan")
            epoch_g = float(np.mean(gen_losses_epoch)) if gen_losses_epoch else float("nan")

            # Quick test-set critic loss (non-private read; informational only).
            with torch.no_grad():
                idx = torch.randperm(self.test_data.shape[0])[: min(1024, self.test_data.shape[0])]
                test_real = self.test_data[idx].to(self.device)
                test_noise = self._sample_noise(test_real.shape[0])
                test_fake = self.generator(test_noise)
                test_d_loss = float(
                    (self.discriminator(test_fake).mean() - self.discriminator(test_real).mean())
                    .detach().cpu()
                )

            self.history["disc_loss"].append(round(epoch_d, 4))
            self.history["gen_loss"].append(round(epoch_g, 4))
            self.history["test_disc_loss"].append(round(test_d_loss, 4))
            self.history["epoch_time"].append(round(time.time() - epoch_start, 2))

            print(
                f"[DP] Epoch {epoch + 1}/{epochs} "
                f"D={epoch_d:.4f} G={epoch_g:.4f} testD={test_d_loss:.4f} "
                f"t={self.history['epoch_time'][-1]:.1f}s",
                flush=True,
            )

        print(f"[train] complete in {time.time() - t0:.1f}s", flush=True)

        if privacy_engine is not None:
            try:
                achieved_eps = float(privacy_engine.get_epsilon(delta=dp_config.target_delta))
            except Exception:
                achieved_eps = None
            try:
                noise_mult = float(disc_optim.noise_multiplier)
            except Exception:
                noise_mult = None
            self.dp_stats = {
                "target_epsilon": dp_config.target_epsilon,
                "target_delta": dp_config.target_delta,
                "max_grad_norm": dp_config.max_grad_norm,
                "per_device_batch_size": dp_config.per_device_batch_size,
                "gradient_accumulation_steps": dp_config.gradient_accumulation_steps,
                "effective_batch_size": dp_config.effective_batch_size,
                "critic_iters": self.critic_iters,
                "dp_epochs_critic": epochs * self.critic_iters,
                "train_samples": self.n_train,
                "sample_rate": float(dp_config.effective_batch_size) / max(1, self.n_train),
                "achieved_epsilon": achieved_eps,
                "noise_multiplier": noise_mult,
                "epoch_disc_losses": self.history["disc_loss"],
                "epoch_gen_losses": self.history["gen_loss"],
            }
        else:
            self.dp_stats = {
                "dp_enabled": False,
                "epoch_disc_losses": self.history["disc_loss"],
                "epoch_gen_losses": self.history["gen_loss"],
            }

    def generate(self, n_rows: int) -> pd.DataFrame:
        self.generator.eval()
        with torch.no_grad():
            noise = self._sample_noise(n_rows)
            samples = self.generator(noise).cpu().numpy()
        self.generator.train()
        return pd.DataFrame(samples, columns=self.col_names)

    def save_checkpoint(self, path: Path) -> None:
        # Unwrap GradSampleModule if Opacus wrapped the critic.
        disc = getattr(self.discriminator, "_module", self.discriminator)
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": disc.state_dict(),
                "n_features": self.n_features,
                "base_nodes": self.base_nodes,
                "noise_dim": self.noise_dim,
                "col_names": self.col_names,
            },
            path,
        )
        print(f"[ckpt] saved {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DP-WGAN tabular generator (PyTorch + Opacus).")

    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--critic-iters", type=int, default=5)
    parser.add_argument("--base-nodes", type=int, default=64)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--weight-clip", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=10,
                        help="Generator-level epochs. Critic sees epochs*critic_iters passes.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-synth-files", type=int, default=10)
    parser.add_argument("--samples-per-file", type=int, default=None,
                        help="Rows per synthetic CSV. Defaults to test-set size.")

    parser.add_argument("--row-limit", type=int, default=None,
                        help="Optional row cap for quick smoke runs.")
    parser.add_argument("--train-only", action="store_true",
                        help="Train and save metadata only; skip synthetic generation.")

    parser.add_argument("--dp", action="store_true",
                        help="Enable DP-SGD on the discriminator (via Opacus).")
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


def resolve_split_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    train = Path(args.train_csv) if args.train_csv else DATA_DIR / "diabetic_data_preprocessed_train_sdv.csv"
    test = Path(args.test_csv) if args.test_csv else DATA_DIR / "diabetic_data_preprocessed_test_sdv.csv"
    return train, test


def output_path_for_synth(output_dir: Path, dp_enabled: bool, critic_iters: int, base_nodes: int, idx: int) -> Path:
    suffix = "_dp" if dp_enabled else ""
    return output_dir / f"synthetic_{critic_iters}_{base_nodes}{suffix}_test_{idx}.csv"


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = _select_device()
    _print_env_info(device)

    train_path, test_path = resolve_split_paths(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.row_limit is not None:
        if args.row_limit <= 0:
            raise ValueError("--row-limit must be a positive integer.")
        capped_train = output_dir / f"_tmp_train_rowcap_{args.row_limit}.csv"
        pd.read_csv(train_path).head(args.row_limit).to_csv(capped_train, index=False)
        train_path = capped_train
        print(f"[main] row limit applied: {args.row_limit}", flush=True)

    dp_config = build_dp_config(args)

    wgan = DPWGAN(
        train_filepath=str(train_path),
        test_filepath=str(test_path),
        critic_iters=args.critic_iters,
        base_nodes=args.base_nodes,
        noise_dim=args.noise_dim,
        weight_clip=args.weight_clip,
        device=device,
    )

    wgan.train(
        epochs=args.epochs,
        learning_rate=args.lr,
        dp_config=dp_config,
        seed=args.seed,
    )

    ckpt_name = f"wgan_dp_{args.critic_iters}_{args.base_nodes}.pt" if dp_config else f"wgan_{args.critic_iters}_{args.base_nodes}.pt"
    wgan.save_checkpoint(output_dir / ckpt_name)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "wgan_tabular_generation_dp",
        "parameters": {
            "critic_iters": args.critic_iters,
            "base_nodes": args.base_nodes,
            "noise_dim": args.noise_dim,
            "weight_clip": args.weight_clip,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "seed": args.seed,
            "row_limit": args.row_limit,
            "dp_enabled": dp_config is not None,
            "dp": (
                {
                    "target_epsilon": dp_config.target_epsilon,
                    "target_delta": dp_config.target_delta,
                    "max_grad_norm": dp_config.max_grad_norm,
                    "per_device_batch_size": dp_config.per_device_batch_size,
                    "gradient_accumulation_steps": dp_config.gradient_accumulation_steps,
                    "effective_batch_size": dp_config.effective_batch_size,
                }
                if dp_config is not None else None
            ),
        },
        "training_stats": wgan.dp_stats,
        "synthetic_files": [],
    }

    if not args.train_only:
        n_rows = args.samples_per_file or wgan.test_data.shape[0]
        for i in range(args.n_synth_files):
            df = wgan.generate(n_rows)
            out_path = output_path_for_synth(
                output_dir, dp_config is not None, args.critic_iters, args.base_nodes, i
            )
            df.to_csv(out_path, index=False)
            print(f"[gen] wrote {out_path} ({len(df)} rows)", flush=True)
            metadata["synthetic_files"].append(str(out_path))

    metadata_filename = "run_metadata_dp.json" if dp_config is not None else "run_metadata.json"
    metadata_path = output_dir / metadata_filename
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nMetadata saved: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
