# Synthetic Data Generation Workflows (Slide-Ready)

This document summarizes the **actual pipeline in this repository** for:
- **PATE-GAN**
- **HealthGAN (WGAN-GP)**
- **Pythia (LoRA fine-tuned LLM)**

---

## 1) PATE-GAN workflow

```mermaid
flowchart TD
    A[Input split CSV\n`thesis/data/diabetic_data_preprocessed_{train|test}.csv`] --> B[Min-max normalize all columns to [0,1]]
    B --> C[Run `pategan(x_train, params)`]
    C --> D[Partition training data into k teacher subsets]
    D --> E[Generator samples synthetic batch]
    E --> F[Train k logistic teacher models\n(real subset=1 vs generated=0)]
    F --> G[Aggregate teacher votes + Laplace noise\n(PATE label)]
    G --> H[Train student discriminator on noisy labels]
    H --> I[Update moments accountant]
    I --> J{epsilon_hat < epsilon?}
    J -- yes --> E
    J -- no --> K[Sample final synthetic table from generator]
    K --> L[Denormalize to original scale\n(age snapped; integer columns rounded/clipped)]
    L --> M[Save\n`thesis/data/pategan/diabetic_data_pategan_{split}_synthetic_epsilon_{eps}.csv`]
```

**Default params used in script:** `data_no=57214`, `n_s=1`, `batch_size=64`, `k=10`, `epsilon=5.0`, `delta=1e-5`, `lamda=1.0`.

---

## 2) HealthGAN workflow

```mermaid
flowchart TD
    A[Input real train/test CSV] --> B[SDV encode to [0,1]\n`sdv_converter.py encode`]
    B --> C[Train WGAN-GP\n`wgan_for_mac.py`]
    C --> D[Generator: noise(100) -> 2F -> 1.5F -> F (sigmoid)]
    C --> E[Discriminator: F -> 64 -> 128 -> 256 -> 1]
    D --> F[Adversarial training loop\ncritic_iters times D, then 1 G]
    E --> F
    F --> G[Gradient penalty added to critic loss]
    G --> H[At final epoch, generate 10 synthetic CSVs\n(in SDV/normalized space)]
    H --> I[Decode back to original schema\n`sdv_converter.py decode`]
    I --> J[Post-process numeric columns\n(snap-to-nearest valid values, clip/round)]
    J --> K[Save decoded files\n`thesis/data/healthgan/*_decoded.csv`]
```

**Default training config in code:** `num_epochs=100000`, `critic_iters=5`, `lambda=10`, `base_nodes=64`, `samples_per_file=57214`.

---

## 3) Pythia workflow

```mermaid
flowchart TD
    A[Input split CSV\n`thesis/data/diabetic_data_preprocessed_{train|test}.csv`] --> B[Derive table schema\n(dtype, bounds, discrete support, categories)]
    B --> C[Serialize each row as text\n`Class_{label} | col=value | ...`]
    C --> D[Load pretrained `EleutherAI/pythia-70m`]
    D --> E[Apply LoRA adapters and fine-tune on split text corpus]
    E --> F[Compute target class counts from real split]
    F --> G[For each class (0/1), generate rows with class-conditioned prompt]
    G --> H[Parse generated text -> key/value row]
    H --> I[Coerce to schema\n(range clip, nearest discrete value, category fallback)]
    I --> J{Enough valid rows?}
    J -- no --> G
    J -- still short --> R[Resample accepted rows or fallback schema-valid random rows]
    R --> K[Merge classes and shuffle]
    J -- yes --> K
    K --> L[Final cast + strict validation\n(columns/order, row count, no NA, binary target)]
    L --> M[Save split synthetic CSV]
    M --> N[Write run metadata JSON\n(parameters, hashes, class stats)]
```

**Default run config in script:** `epochs=10`, `batch_size=8`, `lr=2e-5`, `max_length=512`, `temperature=0.8`, `top_p=0.95`, `max_retries_per_row=8`.

**Important:** this implementation explicitly states it **does not provide formal differential privacy**.

---

## 4) One-slide comparison (quick)

| Model | Core idea | Privacy mechanism | Main output path |
|---|---|---|---|
| PATE-GAN | GAN with teacher-student aggregation | PATE noisy voting + privacy accountant until epsilon budget | `thesis/data/pategan/` |
| HealthGAN | WGAN-GP on SDV-normalized tabular data | No formal DP in this code path (privacy by synthetic generation + evaluation) | `thesis/data/healthgan/` |
| Pythia | LoRA-finetuned causal LLM generating row text | No formal DP (schema-constrained generation + validation) | `thesis/data/pythia/` |

---

## 5) Minimal command view (for methodology slide)

```bash
# PATE-GAN
python Pategan/generate_pategan_synthetic.py --splits train test --epsilon 5.0

# HealthGAN (typical sequence)
python healthgan/generators/sdv_converter.py thesis/data/diabetic_data_preprocessed_train.csv encode
python healthgan/generators/sdv_converter.py thesis/data/diabetic_data_preprocessed_test.csv encode
python healthgan/generators/wgan_for_mac.py 5 64
python healthgan/generators/sdv_converter.py thesis/data/diabetic_data_preprocessed_train.csv decode thesis/data/healthgan/samples_99999_5_64_synthetic_0.csv

# Pythia
python pythia/generate_pythia_synthetic.py --splits train test --model-name EleutherAI/pythia-70m
```
