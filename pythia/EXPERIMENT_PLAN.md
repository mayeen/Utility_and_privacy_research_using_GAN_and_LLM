# Pythia Synthetic Data Generation - Experiment Plan

## Goal

Generate non-DP and DP synthetic datasets that are directly comparable for the thesis.
The main comparison must use the same model, train split, max sequence length,
effective batch size, learning rate, epoch count, generation settings, and seed.
The only intended methodological difference is DP-SGD noise/clipping.

The DP trainer now treats:

```text
effective_batch_size = --dp-per-device-batch-size * --dp-grad-accum-steps
```

For the recommended configuration:

```text
32 * 16 = 512
```

This means Opacus accounts privacy at logical batch size 512, while the GPU only
processes physical microbatches of 32.

---

## Fixed Dataset Rules

Use the full training split for every DP run.

Do not use `--row-limit` with `--dp`. DP privacy accounting depends on the sample
rate:

```text
sample_rate = effective_batch_size / n_training_rows
```

With about 57,214 training rows and effective batch 512:

```text
sample_rate ~= 512 / 57214 = 0.00895
```

If you use only 5,000 rows, the sample rate becomes about 0.1024, which forces
much stronger noise for the same epsilon and makes the DP result misleading.

---

## Phase 0 - Environment Check

Run this once before experiments:

```bash
python -m py_compile \
  pythia/pythia_tabular_dp.py \
  pythia/generate_pythia_synthetic_dp.py \
  pythia/generate_pythia_synthetic.py
```

Install/update the Pythia dependencies if needed:

```bash
pip install -r pythia/requirements.txt
```

Check the train/test row counts:

```bash
wc -l thesis/data/diabetic_data_preprocessed_train.csv \
      thesis/data/diabetic_data_preprocessed_test.csv
```

Expected CSV line counts include the header:

```text
train: 57215 lines -> 57214 data rows
test: 14305 lines -> 14304 data rows
```

---

## Phase 1 - DP Accounting Sanity Check

First verify that gradient accumulation is really active.

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --dp \
  --train-only \
  --epochs 1 \
  --lr 1e-4 \
  --max-length 512 \
  --dp-per-device-batch-size 32 \
  --dp-grad-accum-steps 16 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --splits train
```

Check the terminal log. It must contain values like:

```text
per_device_bs=32 grad_accum=16 effective_bs=512 logical_steps/epoch=112
```

Then check the metadata:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("thesis/data/pythia/run_metadata_dp.json")
m = json.loads(p.read_text())
s = m["splits"]["train"]["dp_stats"]

for k in [
    "train_samples",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "sample_rate",
    "target_epsilon",
    "achieved_epsilon_prv",
    "noise_multiplier",
    "epoch_losses",
]:
    print(k, s.get(k))
PY
```

Expected checks:

| Field | Expected |
|---|---|
| `train_samples` | about `57214` |
| `per_device_batch_size` | `32` |
| `gradient_accumulation_steps` | `16` |
| `effective_batch_size` | `512` |
| `sample_rate` | about `0.00895` |
| `noise_multiplier` | not `None` |
| `epoch_losses` | one value |

If these checks fail, do not continue to full experiments.

---

## Phase 2 - Training Sweep

The previous DP losses were produced before gradient accumulation was actually
used, so do not use them to choose the final configuration. Re-run a small,
controlled sweep with the corrected DP trainer.

Use full training data and `--train-only`. This avoids generation time while
still measuring the actual training behavior.

### DP run A: conservative learning rate

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --dp \
  --train-only \
  --epochs 5 \
  --lr 5e-5 \
  --max-length 512 \
  --dp-per-device-batch-size 32 \
  --dp-grad-accum-steps 16 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --splits train
```

Record:

```text
lr=5e-5, epoch_losses=[...]
```

### DP run B: default learning rate

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --dp \
  --train-only \
  --epochs 5 \
  --lr 1e-4 \
  --max-length 512 \
  --dp-per-device-batch-size 32 \
  --dp-grad-accum-steps 16 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --splits train
```

Record:

```text
lr=1e-4, epoch_losses=[...]
```

### DP run C: aggressive learning rate, only if A/B are flat

Run this only if both `5e-5` and `1e-4` barely improve.

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --dp \
  --train-only \
  --epochs 5 \
  --lr 2e-4 \
  --max-length 512 \
  --dp-per-device-batch-size 32 \
  --dp-grad-accum-steps 16 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --splits train
```

Do not use `3e-4` unless all lower rates fail. Your earlier `3e-4` run increased
loss after epoch 1, so it is a high-risk setting.

---

## Phase 3 - Choose Final LR and Epoch Count

Pick the configuration from the DP sweep using this rule:

| DP loss pattern | Decision |
|---|---|
| Loss decreases for 2-5 epochs, then flattens | Use the first flat epoch |
| Lowest loss is epoch 1 and later epochs rise | Use `epochs=1`, lower LR if needed |
| Loss rises every epoch | LR is too high; choose a lower LR |
| Loss barely changes from epoch 1 | LR may be too low; try the next higher LR |
| Loss is noisy but final loss is best or near best | Run more epochs freely — best epoch is auto-restored |

> **Note — auto best-epoch restore:** The code automatically snapshots the
> lowest-loss epoch's LoRA weights during training and restores them before
> generation. You do not need to manually set `--epochs` to the best epoch
> number. Run a longer sweep and the best checkpoint is used automatically.
> Use `--disable-best-epoch-restore` only if you explicitly want the final
> epoch instead.

Recommended default if the sweep is ambiguous:

```text
epochs = 5
lr = 5e-5 or 1e-4, whichever has the lower best-epoch loss
effective_batch_size = 512
epsilon = 5.0
delta = 1e-5
max_grad_norm = 1.0
```

For the main thesis comparison, use the same final `epochs` and `lr` for both
non-DP and DP. This keeps the comparison defensible.

You may also report a secondary non-DP "best utility" baseline if non-DP keeps
improving for many epochs, but do not mix that with the main DP-vs-non-DP matched
comparison.

---

## Phase 4 - Matched Non-DP Train-Only Check

After choosing `<LR>` and `<N>` from the DP sweep, run the matched non-DP training
check:

```bash
python -m pythia.generate_pythia_synthetic \
  --train-only \
  --epochs <N> \
  --batch-size 512 \
  --lr <LR> \
  --max-length 512 \
  --splits train
```

Check:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("thesis/data/pythia/run_metadata.json")
m = json.loads(p.read_text())
print(m["splits"]["train"]["training_stats"]["epoch_losses"])
PY
```

If `--batch-size 512` causes a CUDA out-of-memory error in the non-DP run, do not
silently change only the non-DP setup. Either:

1. add non-DP gradient accumulation support and keep effective batch 512, or
2. document a secondary non-DP baseline with smaller batch as not perfectly
   matched.

The preferred thesis setup is option 1 or a successful direct `--batch-size 512`
run.

---

## Phase 5 - Full Synthetic Generation

Fill in `<LR>` and `<N>` from Phase 3.

### Full non-DP generation

```bash
python -m pythia.generate_pythia_synthetic \
  --epochs <N> \
  --batch-size 512 \
  --lr <LR> \
  --max-length 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-retries-per-row 8 \
  --splits train test
```

Expected outputs:

```text
thesis/data/pythia/diabetic_data_pythia_train_synthetic.csv
thesis/data/pythia/diabetic_data_pythia_test_synthetic.csv
thesis/data/pythia/run_metadata.json
```

### Full DP generation

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --dp \
  --epochs <N> \
  --lr <LR> \
  --max-length 512 \
  --dp-per-device-batch-size 32 \
  --dp-grad-accum-steps 16 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --max-grad-norm 1.0 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-retries-per-row 8 \
  --splits train test
```

Expected outputs:

```text
thesis/data/pythia/diabetic_data_pythia_train_dp_synthetic.csv
thesis/data/pythia/diabetic_data_pythia_test_dp_synthetic.csv
thesis/data/pythia/run_metadata_dp.json
```

---

## Phase 6 - Output Quality Checks

Run this after each full generation:

```bash
python - <<'PY'
import json
import pandas as pd
from pathlib import Path

base = Path("thesis/data/pythia")
files = [
    base / "diabetic_data_pythia_train_synthetic.csv",
    base / "diabetic_data_pythia_test_synthetic.csv",
    base / "diabetic_data_pythia_train_dp_synthetic.csv",
    base / "diabetic_data_pythia_test_dp_synthetic.csv",
]

for f in files:
    if not f.exists():
        print("missing", f)
        continue
    df = pd.read_csv(f)
    print()
    print(f.name)
    print("rows:", len(df))
    print("cols:", len(df.columns))
    print("readmitted counts:")
    print(df["readmitted"].value_counts(dropna=False).sort_index())
    print("missing values:", int(df.isna().sum().sum()))

for meta_name in ["run_metadata.json", "run_metadata_dp.json"]:
    p = base / meta_name
    if not p.exists():
        print("missing", p)
        continue
    m = json.loads(p.read_text())
    print()
    print(meta_name)
    for split, s in m["splits"].items():
        stats = s.get("generation_stats", {})
        print(split, "source_rows=", s.get("source_rows"), "synthetic_rows=", s.get("synthetic_rows"))
        print(split, "source_class_counts=", s.get("source_class_counts"))
        print(split, "synthetic_class_counts=", s.get("synthetic_class_counts"))
        if stats.get("dp_stats"):
            print(split, "dp_stats=", stats["dp_stats"])
PY
```

Check these points:

| Check | Desired result |
|---|---|
| Synthetic train rows | matches source train rows |
| Synthetic test rows | matches source test rows |
| Column count | matches source column count |
| `readmitted` classes | both classes present |
| Class counts | close to requested source split counts |
| Missing values | ideally zero after postprocessing |
| DP `achieved_epsilon_prv` | near or below target epsilon |
| DP `noise_multiplier` | recorded and not `None` |
| DP `sample_rate` | about `0.00895` for train |

---

## Phase 7 - Optional Pythia-410M Extension

Treat `EleutherAI/pythia-410m` as a model-size experiment, not as a replacement
for the 70M baseline. The 410M model can improve row syntax and utility because
it has more capacity, but it also changes runtime, GPU memory, and DP optimization
difficulty.

Recommended reporting structure:

```text
Primary experiment:
  Pythia-70M non-DP vs Pythia-70M DP

Model-size extension:
  Pythia-410M non-DP vs Pythia-410M DP
  Optional: compare 70M DP vs 410M DP at same epsilon
```

Do not overwrite the 70M outputs. Use a separate output directory:

```text
thesis/data/pythia_410m
```

### How 410M affects the experiment

| Area | Expected effect |
|---|---|
| Memory | Much higher, especially for DP per-sample gradients |
| Runtime | Slower training and slower generation |
| Non-DP quality | Usually better syntax and schema following |
| DP quality | May improve from larger pretrained capacity, but DP noise/clipping can still dominate |
| Privacy accounting | Same epsilon/delta if sample rate, epochs, clipping, and accountant settings match |
| DP noise multiplier | Mostly driven by epsilon, delta, sample rate, and epochs, not model size |
| Optimization | Larger LoRA parameter set receives DP noise, so LR may need to be lower |

The privacy guarantee is still `(epsilon, delta)` for the training corpus. A larger
model does not by itself weaken the formal DP bound. However, training can become
less stable because DP-SGD adds noise to a larger trainable LoRA update.

### 410M DP memory strategy

Start with the same effective batch size 512, but reduce the physical microbatch
if memory is tight:

| Physical batch | Grad accumulation | Effective batch |
|---|---:|---:|
| `32` | `16` | `512` |
| `16` | `32` | `512` |
| `8` | `64` | `512` |
| `4` | `128` | `512` |

Use the largest physical batch that fits GPU memory. The effective batch should
remain 512 so the sample rate stays comparable to the 70M experiment.

### 410M DP accounting sanity check

Try `32 x 16` first:

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --model-name EleutherAI/pythia-410m \
  --output-dir thesis/data/pythia_410m \
  --dp \
  --train-only \
  --epochs 1 \
  --lr 5e-5 \
  --max-length 512 \
  --dp-per-device-batch-size 32 \
  --dp-grad-accum-steps 16 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --splits train
```

If that causes CUDA out-of-memory, retry with `16 x 32`:

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --model-name EleutherAI/pythia-410m \
  --output-dir thesis/data/pythia_410m \
  --dp \
  --train-only \
  --epochs 1 \
  --lr 5e-5 \
  --max-length 512 \
  --dp-per-device-batch-size 16 \
  --dp-grad-accum-steps 32 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --splits train
```

If needed, use `8 x 64`:

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --model-name EleutherAI/pythia-410m \
  --output-dir thesis/data/pythia_410m \
  --dp \
  --train-only \
  --epochs 1 \
  --lr 5e-5 \
  --max-length 512 \
  --dp-per-device-batch-size 8 \
  --dp-grad-accum-steps 64 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --splits train
```

Check the terminal log. For all of these, the important part is:

```text
effective_bs=512 logical_steps/epoch=112
```

The `per_device_bs` and `grad_accum` may change, but `effective_bs` should stay
512.

### 410M DP learning-rate sweep

For 410M, start lower than 70M. Recommended sweep:

```text
2e-5, 5e-5, 1e-4
```

Run this first:

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --model-name EleutherAI/pythia-410m \
  --output-dir thesis/data/pythia_410m \
  --dp \
  --train-only \
  --epochs 5 \
  --lr 2e-5 \
  --max-length 512 \
  --dp-per-device-batch-size <PHYSICAL_BS> \
  --dp-grad-accum-steps <ACCUM_STEPS> \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --splits train
```

Then repeat with:

```text
--lr 5e-5
--lr 1e-4
```

Choose the 410M LR and epoch count independently from 70M. Do not force 410M to
use the 70M LR if its loss curve says otherwise.

### Final 410M 512-batch train-only test

Use this pair when you want to test the final 410M configuration with matched
effective batch size 512 before spending time on full row generation.

DP command:

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --model-name EleutherAI/pythia-410m \
  --output-dir thesis/data/pythia_410m \
  --dp \
  --train-only \
  --epochs 5 \
  --lr 4e-4 \
  --max-length 512 \
  --dp-per-device-batch-size 16 \
  --dp-grad-accum-steps 32 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --max-grad-norm 1.0 \
  --splits train
```

Expected DP log:

```text
per_device_bs=16 grad_accum=32 effective_bs=512 logical_steps/epoch=112
```

Equivalent non-DP command:

```bash
python -m pythia.generate_pythia_synthetic \
  --model-name EleutherAI/pythia-410m \
  --output-dir thesis/data/pythia_410m \
  --train-only \
  --epochs 5 \
  --batch-size 512 \
  --lr 4e-4 \
  --max-length 512 \
  --splits train
```

Use these results to decide whether `lr=4e-4` and `epochs=5` are stable. If DP
loss rises after an earlier epoch, use the best-loss epoch count instead of
blindly keeping all 5 epochs.

### 410M non-DP matched run

The ideal matched non-DP 410M run uses effective batch 512:

```bash
python -m pythia.generate_pythia_synthetic \
  --model-name EleutherAI/pythia-410m \
  --output-dir thesis/data/pythia_410m \
  --train-only \
  --epochs <N_410M> \
  --batch-size 512 \
  --lr <LR_410M> \
  --max-length 512 \
  --splits train
```

If 410M non-DP cannot fit `--batch-size 512`, this codebase currently needs a
non-DP gradient accumulation option to keep the 410M comparison perfectly matched.
Until that is added, a smaller non-DP batch can be used only as a clearly labeled
secondary baseline.

### 410M full generation

After choosing `<LR_410M>`, `<N_410M>`, `<PHYSICAL_BS>`, and `<ACCUM_STEPS>`,
run:

```bash
python -m pythia.generate_pythia_synthetic \
  --model-name EleutherAI/pythia-410m \
  --output-dir thesis/data/pythia_410m \
  --epochs <N_410M> \
  --batch-size 512 \
  --lr <LR_410M> \
  --max-length 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-retries-per-row 8 \
  --splits train test
```

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --model-name EleutherAI/pythia-410m \
  --output-dir thesis/data/pythia_410m \
  --dp \
  --epochs <N_410M> \
  --lr <LR_410M> \
  --max-length 512 \
  --dp-per-device-batch-size <PHYSICAL_BS> \
  --dp-grad-accum-steps <ACCUM_STEPS> \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --max-grad-norm 1.0 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-retries-per-row 8 \
  --splits train test
```

Expected 410M outputs:

```text
thesis/data/pythia_410m/diabetic_data_pythia_train_synthetic.csv
thesis/data/pythia_410m/diabetic_data_pythia_test_synthetic.csv
thesis/data/pythia_410m/diabetic_data_pythia_train_dp_synthetic.csv
thesis/data/pythia_410m/diabetic_data_pythia_test_dp_synthetic.csv
thesis/data/pythia_410m/run_metadata.json
thesis/data/pythia_410m/run_metadata_dp.json
```

### 410M decision rule

Proceed to full 410M generation only if:

| Check | Requirement |
|---|---|
| DP sanity check | `effective_bs=512` and `noise_multiplier` is recorded |
| Memory | training completes without OOM |
| Loss | does not rise steadily after epoch 1 |
| Runtime | acceptable for both train and generation |
| Thesis scope | 70M primary experiment is already complete or scheduled |

If 410M DP trains but generation is too slow, generate only the train split first:

```bash
--splits train
```

Then run the test split separately after confirming the train output is valid.

---

## Phase 8 - Final Paper Run (Fixed Parameters)

These are the fixed parameters from the paper. Run these after the sweep phases
are complete and you are ready to produce the final thesis outputs.

Parameters:

```text
lr = 4e-4
DP physical batch size = 32
DP gradient accumulation steps = 16
effective batch size = 512
epochs = 5  (best epoch auto-restored)
epsilon = 5.0
delta = 1e-5
max_grad_norm = 1.0
```

### Final non-DP generation

```bash
python -m pythia.generate_pythia_synthetic \
  --epochs 5 \
  --batch-size 512 \
  --lr 4e-4 \
  --max-length 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-retries-per-row 8 \
  --splits train test
```

### Final DP generation

```bash
python -m pythia.generate_pythia_synthetic_dp \
  --dp \
  --epochs 5 \
  --lr 4e-4 \
  --max-length 512 \
  --dp-per-device-batch-size 32 \
  --dp-grad-accum-steps 16 \
  --target-epsilon 5.0 \
  --target-delta 1e-5 \
  --max-grad-norm 1.0 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-retries-per-row 8 \
  --splits train test
```

Expected outputs:

```text
thesis/data/pythia/diabetic_data_pythia_train_synthetic.csv
thesis/data/pythia/diabetic_data_pythia_test_synthetic.csv
thesis/data/pythia/diabetic_data_pythia_train_dp_synthetic.csv
thesis/data/pythia/diabetic_data_pythia_test_dp_synthetic.csv
thesis/data/pythia/run_metadata.json
thesis/data/pythia/run_metadata_dp.json
```

---

## Final Configuration Template

Use this template in the thesis methods section:

```text
Model: EleutherAI/pythia-70m
Training data: full diabetic_data_preprocessed_train.csv
Target column: readmitted
Max sequence length: 512
Effective batch size: 512
Non-DP batch size: 512
DP physical batch size: 32
DP gradient accumulation steps: 16
Learning rate: 4e-4
Epochs: 5 (best-loss epoch weights auto-restored before generation)
Optimizer: AdamW
LoRA rank: 8
LoRA alpha: 16
LoRA dropout: 0.0
DP epsilon: 5.0
DP delta: 1e-5
DP max grad norm: 1.0
Generation temperature: 0.8
Generation top-p: 0.95
Max retries per row: 8
Seed: 42
```

Main result to report:

```text
Matched non-DP Pythia vs DP Pythia, same LR, same epoch count, same effective
batch size, same generation settings.
```

Secondary result, optional:

```text
Best non-DP utility baseline, clearly labeled as not privacy matched if it uses
more epochs or different optimization settings.
```
