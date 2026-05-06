# Pythia Synthetic Data Generation — Experiment Plan

## Goal

Generate non-DP and DP synthetic datasets that are directly comparable. Both runs use identical training hyperparameters so that any difference in output quality is attributable solely to the DP noise/clipping, not to different training setups.

---

## Phase 1 — Smoke Test (5 000 rows)

Run both pipelines on a 5 000-row subset to determine where loss converges before committing to the full 57 000-row training run.

### Non-DP smoke test

```bash
python -m pythia.generate_pythia_synthetic \
  --row-limit 5000 \
  --epochs 20 \
  --batch-size 512 \
  --lr 1e-4 \
  --max-length 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-retries-per-row 8 \
  --splits train
```

### DP smoke test

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --dp \
  --row-limit 5000 \
  --epochs 20 \
  --lr 1e-4 \
  --dp-per-device-batch-size 32 \
  --dp-grad-accum-steps 16 \
  --max-length 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-retries-per-row 8 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --splits train
```

> At 5 000 rows and effective batch 512, each epoch is ~10 steps — fast enough to run 20 epochs in minutes.

### What to look for in `run_metadata.json` → `training_stats.epoch_losses`

| Signal | Decision |
|---|---|
| Loss still falling at epoch 20 | Increase epochs for full run (try 25–30) |
| Loss plateaus by epoch N | Use N epochs for full run |
| DP loss never meaningfully decreases | LR too low — try 2e-4 or 5e-4 |
| DP loss diverges / spikes | LR too high — try 5e-5 |
| Non-DP converges much earlier than DP | Normal; use separate epoch counts only if justified |

**Target:** identify the epoch N where `epoch_losses[N] - epoch_losses[N-1] < 0.005` for 2+ consecutive epochs. Use that N for the full run.

---

## Phase 2 — Full Run

Replace `--epochs` with the value determined from Phase 1. Remove `--row-limit`.

### Full non-DP run

```bash
python -m pythia.generate_pythia_synthetic \
  --epochs <N_from_smoke_test> \
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
  --epochs <N_from_smoke_test> \
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
| Learning rate | 1e-4 | scaled up from 2e-5 to match larger batch size |
| Max length | 512 | covers full row serialisation |
| Temperature | 0.8 | generation diversity |
| Top-p | 0.95 | nucleus sampling |
| Max retries/row | 8 | generation quality floor |
| DP ε | 5.0 | moderate privacy guarantee |
| DP δ | 1e-5 | standard for this dataset size |
| Max grad norm | 1.0 | standard DP-SGD clipping |
| Seed | 42 | reproducibility |

---

## Experimental design note

Both runs use the same effective batch size (512), learning rate (1e-4), and epoch count. The **only** difference is the `--dp` flag. This isolates the cost of differential privacy and makes the comparison defensible in the thesis methodology section.
