"""
Nearest-Neighbor Distance Privacy Evaluation
=============================================
For each real training record, find the closest synthetic record (L2 distance
in min-max-normalised feature space) and report summary statistics.

A large mean/median NND indicates the generator is NOT copying real records
verbatim, i.e. the synthetic data is privacy-preserving.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

REAL_TRAIN = DATA / "diabetic_data_preprocessed_train.csv"
MIN_MAX    = DATA / "diabetic_data_preprocessed_train.min_max"

SYNTHETIC = {
    "HealthGAN": [
        DATA / "healthgan" / f"samples_99999_5_64_synthetic_{i}.csv"
        for i in range(10)
    ],
    "PATEGAN_train": [DATA / "pategan" / "diabetic_data_pategan_train_synthetic.csv"],
    "PATEGAN_test":  [DATA / "pategan" / "diabetic_data_pategan_test_synthetic.csv"],
    "Pythia_train":  [DATA / "pythia"  / "diabetic_data_pythia_train_synthetic.csv"],
    "Pythia_test":   [DATA / "pythia"  / "diabetic_data_pythia_test_synthetic.csv"],
}

# Column to drop before computing distances (identifier, not a feature)
DROP_COLS = ["encounter_id"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_min_max(path: Path) -> dict:
    """Load the min-max JSON produced during preprocessing."""
    with open(path) as f:
        return json.load(f)


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in DROP_COLS]


def normalise(df: pd.DataFrame, min_max: dict, feature_cols: list) -> np.ndarray:
    """
    Min-max normalise `df[feature_cols]` using stored min/max values.
    Columns absent from min_max are assumed already in [0,1] (e.g. binary flags).
    """
    arr = df[feature_cols].copy().astype(float)
    for col in feature_cols:
        if col in min_max:
            lo, hi = min_max[col][0], min_max[col][1]
            rng = hi - lo
            if rng > 0:
                arr[col] = (arr[col] - lo) / rng
    return arr.values


def nearest_neighbor_distances(real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
    """
    For every real record find its nearest synthetic neighbour.
    Returns an array of distances (one per real record).
    """
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean", n_jobs=-1)
    nn.fit(synthetic)
    distances, _ = nn.kneighbors(real)
    return distances.ravel()


def summarise(distances: np.ndarray) -> dict:
    return {
        "mean":   float(np.mean(distances)),
        "median": float(np.median(distances)),
        "std":    float(np.std(distances)),
        "min":    float(np.min(distances)),
        "max":    float(np.max(distances)),
        "p5":     float(np.percentile(distances, 5)),
        "p25":    float(np.percentile(distances, 25)),
        "p75":    float(np.percentile(distances, 75)),
        "p95":    float(np.percentile(distances, 95)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading real training data …")
    real_df  = pd.read_csv(REAL_TRAIN)
    min_max  = load_min_max(MIN_MAX)
    feat_cols = get_feature_cols(real_df)

    real_norm = normalise(real_df, min_max, feat_cols)
    print(f"  Real records : {len(real_norm):,}  |  features: {real_norm.shape[1]}")

    results = {}

    for model_name, paths in SYNTHETIC.items():
        # Concatenate all files for this model
        frames = []
        for p in paths:
            if not p.exists():
                print(f"  [SKIP] {p} not found")
                continue
            df = pd.read_csv(p)
            # Keep only columns present in real data
            df = df[[c for c in feat_cols if c in df.columns]]
            frames.append(df)

        if not frames:
            print(f"[{model_name}] No data found, skipping.")
            continue

        synth_df  = pd.concat(frames, ignore_index=True)

        # HealthGAN outputs are already normalised; others need normalisation
        if model_name.startswith("HealthGAN"):
            synth_norm = synth_df.values.astype(float)
        else:
            synth_norm = normalise(synth_df, min_max, feat_cols)

        print(f"\n[{model_name}]  synthetic records: {len(synth_norm):,}")
        distances = nearest_neighbor_distances(real_norm, synth_norm)
        stats     = summarise(distances)
        results[model_name] = {"distances": distances, "stats": stats}

        print(f"  Mean NND  : {stats['mean']:.4f}")
        print(f"  Median NND: {stats['median']:.4f}")
        print(f"  Std NND   : {stats['std']:.4f}")
        print(f"  Min NND   : {stats['min']:.6f}")
        print(f"  Max NND   : {stats['max']:.4f}")
        print(f"  P5 / P95  : {stats['p5']:.4f} / {stats['p95']:.4f}")

    # -----------------------------------------------------------------------
    # Save summary table
    # -----------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for model_name, data in results.items():
        row = {"model": model_name}
        row.update(data["stats"])
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("model")
    out_csv = out_dir / "nnd_summary.csv"
    summary_df.to_csv(out_csv)
    print(f"\nSummary saved → {out_csv}")

    # Save per-record distances for each model
    for model_name, data in results.items():
        safe_name = model_name.replace(" ", "_")
        dist_path = out_dir / f"nnd_{safe_name}.npy"
        np.save(dist_path, data["distances"])

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
