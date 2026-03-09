from pathlib import Path

from load_data import load_data
from feature_engineering import engineer_features
from sklearn.model_selection import train_test_split


def main():
    test_size = 0.2
    random_state = 42

    print("Loading raw data ...")
    data = load_data(processed=False)
    print(f"Raw shape: {data.shape}")

    print("Running feature engineering ...")
    data_processed = engineer_features(data, target="readmitted")
    print(f"Processed shape: {data_processed.shape}")

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "diabetic_data_preprocessed.csv"
    data_processed.to_csv(output_path, index=False)
    print(f"Saved preprocessed data to: {output_path}")

    stratify_col = data_processed["readmitted"] if "readmitted" in data_processed.columns else None
    train_df, test_df = train_test_split(
        data_processed,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    train_path = output_dir / "diabetic_data_preprocessed_train.csv"
    test_path = output_dir / "diabetic_data_preprocessed_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train split to: {train_path} (shape: {train_df.shape})")
    print(f"Saved test split to: {test_path} (shape: {test_df.shape})")


if __name__ == "__main__":
    main()
