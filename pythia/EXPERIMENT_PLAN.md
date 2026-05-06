# Pythia Synthetic Data Generation — Experiment Plan

## Goal

Generate non-DP and DP synthetic datasets that are directly comparable. Both runs use identical training hyperparameters so that any difference in output quality is attributable solely to the DP noise/clipping, not to different training setups.

---

## Phase 1 — Smoke Test (convergence check)

Determine where loss plateaus before committing to the full 57 000-row training run.

> **Important:** `--row-limit` cannot be used for the DP smoke test. DP privacy accounting
> depends on `sample_rate = effective_batch / n_training_rows`. At 5 000 rows and
> effective batch 512 the sample rate is ~10%, forcing Opacus to set an extremely high
> noise multiplier to hit ε=5.0. The model learns nothing and generates only invalid rows.
> The non-DP smoke test uses a row limit; the DP smoke test must train on the full dataset
> with a reduced epoch count instead.

### Non-DP smoke test (5 000-row subset, 20 epochs)

`--train-only` skips generation entirely and just saves the loss curve to metadata.

```bash
python -m pythia.generate_pythia_synthetic \
  --train-only \
  --row-limit 5000 \
  --epochs 20 \
  --batch-size 512 \
  --lr 1e-4 \
  --max-length 512 \
  --splits train
```

At 5 000 rows and effective batch 512, each epoch is ~10 steps — 20 epochs finishes in minutes.

### DP smoke test (full dataset, 5 epochs)

No `--row-limit` (would inflate sample rate and break privacy accounting).
`--train-only` skips generation so you get the loss curve without waiting for row generation.

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --dp \
  --train-only \
  --epochs 5 \
  --lr 1e-4 \
  --dp-per-device-batch-size 32 \
  --dp-grad-accum-steps 16 \
  --max-length 512 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --splits train
```

If loss is still clearly descending after 5 epochs, extend to 10 and repeat.

### What to look for in `run_metadata.json` → `generation_stats.training_stats.epoch_losses`

| Signal | Decision |
|---|---|
| Non-DP loss plateaus by epoch N | Use N epochs for both full runs |
| Non-DP loss still falling at epoch 20 | Increase to 30 epochs for full run |
| DP loss never meaningfully decreases | LR too low — try `--lr 2e-4` |
| DP loss diverges / spikes | LR too high — try `--lr 5e-5` |
| DP loss descending but slower than non-DP | Normal — use same epoch count |

**Target:** identify epoch N where `epoch_losses[N] - epoch_losses[N-1] < 0.005` for 2+ consecutive epochs. Use that N for both full runs.

---

## Phase 2 — Full Run

Fill in `<N>` from the smoke test. Both runs use the full dataset and identical hyperparameters.

### Full non-DP run

```bash
python -m pythia.generate_pythia_synthetic \
  --epochs <N> \
  --batch-size 512 \
  --lr 1e-4 \
  --max-length 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-retries-per-row 8
```

Output: `thesis/data/pythia/diabetic_data_pythia_{train,test}_synthetic.csv`

### Full DP run

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --dp \
  --epochs <N> \
  --lr 1e-4 \
  --dp-per-device-batch-size 32 \
  --dp-grad-accum-steps 16 \
  --max-length 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-retries-per-row 8 \
  --target-epsilon 5.0 \
  --target-delta 1e-5
```

Output: `thesis/data/pythia/diabetic_data_pythia_{train,test}_dp_synthetic.csv`

---

## Fixed parameters (both runs)

| Parameter | Value | Rationale |
|---|---|---|
| Model | `EleutherAI/pythia-70m` | consistent with prior runs |
| Effective batch size | 512 | matched between DP and non-DP for fair comparison |
| Learning rate | 1e-4 | scaled up from 2e-5 to match larger batch |
| Max length | 512 | covers full row serialisation |
| Temperature | 0.8 | generation diversity |
| Top-p | 0.95 | nucleus sampling |
| Max retries/row | 8 | generation quality floor |
| DP ε | 5.0 | moderate privacy guarantee |
| DP δ | 1e-5 | standard for this dataset size (~57k rows) |
| Max grad norm | 1.0 | standard DP-SGD clipping |
| Seed | 42 | reproducibility |

---

## Experimental design note

Both runs use the same effective batch size (512), learning rate (1e-4), and epoch count. The **only** difference is the `--dp` flag. This isolates the cost of differential privacy and makes the comparison defensible in the thesis methodology section.
