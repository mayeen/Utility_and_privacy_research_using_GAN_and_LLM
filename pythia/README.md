# Pythia Synthetic Tabular Pipeline

This folder contains a generation-only synthetic data pipeline using pretrained `EleutherAI/pythia-70m` with LoRA fine-tuning.

## What it does
- Trains on `thesis/data/diabetic_data_preprocessed_{split}.csv`
- Uses class-conditioned prompts (`Class_0` / `Class_1`)
- Generates synthetic rows per class to match real class ratio
- Validates and postprocesses outputs to match schema and dtypes
- Writes outputs to `thesis/data/pythia/`

## Install
```bash
pip install -r pythia/requirements.txt
```

## Run (default train + test)
```bash
python pythia/generate_pythia_synthetic.py
```

## Important arguments
```bash
python pythia/generate_pythia_synthetic.py \
  --model-name EleutherAI/pythia-70m \
  --splits train test \
  --epochs 10 \
  --batch-size 8 \
  --lr 2e-5 \
  --max-length 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42 \
  --max-retries-per-row 8
```

## Outputs
- `thesis/data/pythia/diabetic_data_pythia_train_synthetic.csv`
- `thesis/data/pythia/diabetic_data_pythia_test_synthetic.csv`
- `thesis/data/pythia/run_metadata.json`

## Notes
- This version does **not** implement formal differential privacy.
- If generation under-fills valid rows after retries, the pipeline resamples accepted rows per class and logs this behavior.
