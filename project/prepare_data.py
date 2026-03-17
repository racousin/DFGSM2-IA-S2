"""
Prepare SUPPORT2 dataset for the student project.
Downloads the data, selects features, creates train/test splits,
and exports CSVs for two targets:
  - hospdead (binary classification)
  - sfdm2 (multi-class classification)

IDs are shuffled and reassigned so students cannot match rows to the original dataset.
"""

import argparse
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUPPORT2_URL = "https://biostat.app.vumc.org/wiki/pub/Main/SupportDesc/support2csv.zip"

FEATURES = [
    "age", "sex", "race", "income", "edu",
    "dzgroup", "dzclass", "num.co", "ca",
    "meanbp", "hrt", "resp", "temp", "wblc",
    "pafi", "alb", "bili", "crea", "sod",
    "ph", "glucose", "bun", "urine",
    "adlp", "adls", "scoma", "dnr", "avtisst", "aps",
]

TARGETS = {
    "hospdead": {"type": "binary"},
    "sfdm2": {"type": "multiclass"},
}

SEED = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_support2(cache_dir: Path) -> pd.DataFrame:
    """Download SUPPORT2 and return as DataFrame."""
    cache_file = cache_dir / "support2.csv"
    if cache_file.exists():
        print(f"Using cached file: {cache_file}")
        return pd.read_csv(cache_file)

    print(f"Downloading SUPPORT2 from {SUPPORT2_URL} …")
    df = pd.read_csv(SUPPORT2_URL, compression="zip")
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    print(f"Saved raw data to {cache_file}")
    return df


def scramble_ids(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle rows and assign new opaque IDs so the original row order is lost."""
    rng = np.random.default_rng(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Create a hash-based ID so students can't just use the row index
    df.insert(
        0,
        "patient_id",
        [
            hashlib.sha256(f"support2_{seed}_{i}".encode()).hexdigest()[:10]
            for i in range(len(df))
        ],
    )
    return df


def prepare_target(df: pd.DataFrame, target: str, target_info: dict, out_dir: Path, seed: int):
    """Select features + target, drop NaN in target, split, and save."""
    cols = FEATURES + [target]
    sub = df[["patient_id"] + cols].copy()

    # Drop rows where the target is missing
    before = len(sub)
    sub = sub.dropna(subset=[target])
    after = len(sub)
    if before != after:
        print(f"  Dropped {before - after} rows with missing '{target}'")

    # For multi-class, drop rare / blank classes
    if target_info["type"] == "multiclass":
        sub = sub[sub[target].astype(str).str.strip() != ""]

    # Re-scramble per target so train/test splits differ between tasks
    target_seed = seed + hash(target) % 10_000
    sub = sub.sample(frac=1, random_state=target_seed).reset_index(drop=True)

    # Split
    train_df, test_df = train_test_split(
        sub, test_size=TEST_SIZE, random_state=target_seed, stratify=sub[target]
    )

    # Save
    task_dir = out_dir / target
    task_dir.mkdir(parents=True, exist_ok=True)

    # Train set: features + target
    train_df.to_csv(task_dir / "train.csv", index=False)

    # Test set: features only (students must predict)
    test_features = test_df.drop(columns=[target])
    test_features.to_csv(task_dir / "test.csv", index=False)

    # Ground truth (for evaluation, not distributed to students)
    test_df[["patient_id", target]].to_csv(task_dir / "test_ground_truth.csv", index=False)

    print(f"  [{target}] train={len(train_df)}, test={len(test_df)}")
    print(f"  Saved to {task_dir}/")

    # Print class distribution
    print(f"  Class distribution (train):")
    dist = train_df[target].value_counts()
    for cls, cnt in dist.items():
        print(f"    {cls}: {cnt} ({cnt/len(train_df)*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare SUPPORT2 project datasets")
    parser.add_argument(
        "--out", type=str, default="project/data", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out
    cache_dir = root / "project" / ".cache"

    # 1. Download
    df = download_support2(cache_dir)
    print(f"Raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Verify requested columns exist
    missing = [c for c in FEATURES + list(TARGETS.keys()) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # 3. Scramble IDs
    df = scramble_ids(df, seed=args.seed)

    # 4. Prepare each target
    for target, info in TARGETS.items():
        print(f"\nPreparing target: {target} ({info['type']})")
        prepare_target(df, target, info, out_dir, seed=args.seed)

    print("\nDone!")


if __name__ == "__main__":
    main()
