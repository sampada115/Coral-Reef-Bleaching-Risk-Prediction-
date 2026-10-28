"""
stack_images.py
===============
Stacks 4 PNG images (SST, SSTA, Hotspot, DHW) per date into
a single numpy array and splits into train/test sets.

Output:
    3_dataset/train/stacked_YYYY_MM_DD.npy   <- shape (4, H, W)
    3_dataset/test/stacked_YYYY_MM_DD.npy    <- shape (4, H, W)
    3_dataset/train_labels.csv               <- date, label
    3_dataset/test_labels.csv                <- date, label
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from datetime import date, timedelta

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMAGES_ROOT = "2_images"
DATASET_DIR = "3_dataset"
LABELS_CSV  = "baa_labels.csv"

TRAIN_YEARS = list(range(2018, 2024))   # 2018-2023
TEST_YEARS  = [2024, 2025]

# (attribute_key, folder_name, filename_prefix)
ATTRIBUTES = [
    ("sst",     "sst",         "sst"),
    ("ssta",    "sst_anomaly", "sst_anomaly"),
    ("hotspot", "hotspot",     "hotspot"),
    ("dhw",     "dhw",         "dhw"),
]

IMAGE_SIZE = (224, 224)  # resize all images to this (standard for CNNs)
# ──────────────────────────────────────────────


def date_range(start_year, end_year):
    d   = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    while d <= end:
        yield d
        d += timedelta(days=1)


def get_image_path(folder, prefix, year, month, day):
    """Build image path for a given attribute and date."""
    return f"{IMAGES_ROOT}/{folder}/{year}/{month:02d}/{prefix}_{year}_{month:02d}_{day:02d}.png"


def load_and_stack(year, month, day):
    """
    Load 4 PNG images for a date and stack into (4, H, W) array.
    Returns None if any image is missing.
    """
    channels = []
    for attr_key, folder, prefix in ATTRIBUTES:
        path = get_image_path(folder, prefix, year, month, day)

        if not os.path.exists(path):
            return None  # skip if any channel is missing

        img  = Image.open(path).convert("RGB")
        img  = img.resize(IMAGE_SIZE, Image.BILINEAR)  # resize to standard size
        arr  = np.array(img)                            # (H, W, 3)
        gray = arr.mean(axis=2)                         # (H, W) - RGB to grayscale
        channels.append(gray)

    stacked = np.stack(channels, axis=0)                # (4, H, W)
    stacked = stacked / 255.0                           # normalize to 0-1
    return stacked.astype(np.float32)


def process_split(years, split_name, labels_df):
    """Process all dates for a split (train or test)."""

    out_dir = f"{DATASET_DIR}/{split_name}"
    os.makedirs(out_dir, exist_ok=True)

    records     = []
    saved       = 0
    skipped_img = 0
    skipped_lbl = 0

    print(f"\n{'='*50}")
    print(f"Processing {split_name.upper()} | Years: {years[0]}-{years[-1]}")
    print(f"{'='*50}")

    for d in date_range(years[0], years[-1]):

        if d.year not in years:
            continue

        yr, mo, day = d.year, d.month, d.day
        date_str    = str(d)

        # Get label
        row = labels_df[labels_df["date"] == date_str]
        if row.empty:
            skipped_lbl += 1
            continue

        label = int(row["baa_label"].values[0])

        # Check if already processed
        out_path = f"{out_dir}/stacked_{yr}_{mo:02d}_{day:02d}.npy"
        if os.path.exists(out_path):
            records.append({"date": date_str, "label": label, "npy": out_path})
            saved += 1
            continue

        # Load and stack images
        stacked = load_and_stack(yr, mo, day)
        if stacked is None:
            skipped_img += 1
            continue

        # Save
        np.save(out_path, stacked)
        records.append({"date": date_str, "label": label, "npy": out_path})
        saved += 1

        print(f"  Saved: {date_str}  label={label}  shape={stacked.shape}")

    # Save labels CSV
    df = pd.DataFrame(records)
    csv_path = f"{DATASET_DIR}/{split_name}_labels.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    print(f"\n{'─'*40}")
    print(f"  Saved:            {saved} samples")
    print(f"  Missing images:   {skipped_img} dates skipped")
    print(f"  Missing labels:   {skipped_lbl} dates skipped")
    print(f"  Labels saved to:  {csv_path}")

    if saved > 0:
        print(f"\n  Label distribution:")
        for lbl, count in df["label"].value_counts().sort_index().items():
            bar = "█" * (count // 20)
            print(f"    Class {lbl}: {count:4d} days  {bar}")

    return df


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # Load BAA labels
    print("Loading BAA labels...")
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df["date"] = pd.to_datetime(labels_df["date"]).dt.strftime("%Y-%m-%d")
    labels_df = labels_df.dropna(subset=["baa_label"])
    print(f"  Total labels loaded: {len(labels_df)}")

    # Sanity check - print whether first image paths exist
    print("\nSanity check...")
    for attr_key, folder, prefix in ATTRIBUTES:
        test_path = get_image_path(folder, prefix, 2018, 1, 1)
        exists    = os.path.exists(test_path)
        status    = "OK" if exists else "NOT FOUND"
        print(f"  [{status}] {test_path}")

    # Process train and test splits
    train_df = process_split(TRAIN_YEARS, "train", labels_df)
    test_df  = process_split(TEST_YEARS,  "test",  labels_df)

    # Final summary
    print(f"\n{'='*50}")
    print("DATASET READY!")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples:  {len(test_df)}")
    print(f"  Image shape:   (4, {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]})")
    print(f"  Num classes:   5  (BAA 0-4)")
    print(f"\n  Next step: run your CNN model scripts!")
    print(f"{'='*50}")
