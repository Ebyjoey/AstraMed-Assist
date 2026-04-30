"""
AstraMed Assist - Dataset Preparation Script
=============================================
Merges NIH ChestX-ray14, CheXpert, RSNA Pneumonia, TBX11K
into a unified CSV: (path, pneumonia, tb, normal)

Usage:
    python scripts/prepare_dataset.py \
        --nih_path data/raw/NIH \
        --chexpert_path data/raw/CheXpert \
        --rsna_path data/raw/RSNA \
        --tbx_path data/raw/TBX11K \
        --output data/processed \
        --target_per_class 6667
"""

import os
import argparse
import hashlib
import shutil
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Label Normalisation Maps ────────────────────────────────────────────────

NIH_PNEUMONIA_LABELS = {"Pneumonia", "Consolidation", "Infiltration"}
NIH_NORMAL_LABELS = {"No Finding"}

CHEXPERT_PNEUMONIA_LABELS = {"Pneumonia", "Consolidation", "Lung Opacity"}
CHEXPERT_NORMAL_LABELS = {"No Finding"}

# ─── Utility Functions ───────────────────────────────────────────────────────

def file_hash(path: str) -> str:
    """MD5 hash of first 8KB of file — fast duplicate detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(8192))
    return h.hexdigest()


def validate_image(path: str) -> bool:
    """Return True if the image is valid and non-corrupted."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, Exception):
        return False


def convert_to_rgb_and_resize(src: str, dst: str, size: int = 224) -> bool:
    """
    Preprocess image: convert to RGB (3-channel), resize to size×size.
    Returns True on success.
    """
    try:
        with Image.open(src) as img:
            # Handle RGBA, L (grayscale), P (palette) modes
            if img.mode == "RGBA":
                img = img.convert("RGB")
            elif img.mode in ("L", "P", "1"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((size, size), Image.LANCZOS)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            img.save(dst, "JPEG", quality=95)
        return True
    except Exception as e:
        logger.debug(f"Failed to convert {src}: {e}")
        return False


# ─── Source Dataset Parsers ──────────────────────────────────────────────────

def parse_nih(nih_path: str, processed_dir: str, max_per_class: int = 8000) -> pd.DataFrame:
    """Parse NIH ChestX-ray14 dataset."""
    logger.info("Parsing NIH ChestX-ray14...")
    label_csv = os.path.join(nih_path, "Data_Entry_2017.csv")
    if not os.path.exists(label_csv):
        logger.warning(f"NIH label CSV not found at {label_csv}. Skipping.")
        return pd.DataFrame()

    df = pd.read_csv(label_csv)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="NIH"):
        fname = row["Image Index"]
        labels_str = row["Finding Labels"]
        labels = set(labels_str.split("|"))

        # Multi-label assignment
        pneumonia = int(bool(labels & NIH_PNEUMONIA_LABELS))
        tb = 0  # NIH doesn't have TB labels
        normal = int("No Finding" in labels)

        # Ensure at least one label
        if pneumonia == 0 and tb == 0 and normal == 0:
            continue

        # Find image in nested folders
        img_src = None
        for root, _, files in os.walk(nih_path):
            if fname in files:
                img_src = os.path.join(root, fname)
                break

        if img_src is None:
            continue

        img_dst = os.path.join(processed_dir, "images", "nih", fname)
        if not os.path.exists(img_dst):
            if not convert_to_rgb_and_resize(img_src, img_dst):
                continue

        records.append({
            "path": img_dst,
            "pneumonia": pneumonia,
            "tb": tb,
            "normal": normal,
            "source": "NIH"
        })

    df_out = pd.DataFrame(records)
    logger.info(f"NIH: {len(df_out)} valid records parsed.")
    return df_out


def parse_chexpert(chexpert_path: str, processed_dir: str) -> pd.DataFrame:
    """Parse CheXpert dataset."""
    logger.info("Parsing CheXpert...")
    records = []

    for split in ["train", "valid"]:
        label_csv = os.path.join(chexpert_path, f"{split}.csv")
        if not os.path.exists(label_csv):
            continue

        df = pd.read_csv(label_csv)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"CheXpert-{split}"):
            img_src = os.path.join(chexpert_path, row["Path"])
            if not os.path.exists(img_src):
                continue

            # CheXpert uses NaN for unmentioned, -1 for uncertain, 1 for positive
            def chex_label(col):
                v = row.get(col, 0)
                if pd.isna(v):
                    return 0
                return 1 if float(v) == 1.0 else 0

            pneumonia = max(chex_label("Pneumonia"), chex_label("Consolidation"), chex_label("Lung Opacity"))
            normal = chex_label("No Finding")
            tb = 0  # CheXpert has no TB label

            if pneumonia == 0 and normal == 0:
                continue

            fname = Path(img_src).name
            img_dst = os.path.join(processed_dir, "images", "chexpert", fname)
            if not os.path.exists(img_dst):
                if not convert_to_rgb_and_resize(img_src, img_dst):
                    continue

            records.append({
                "path": img_dst,
                "pneumonia": pneumonia,
                "tb": 0,
                "normal": normal,
                "source": "CheXpert"
            })

    df_out = pd.DataFrame(records)
    logger.info(f"CheXpert: {len(df_out)} valid records parsed.")
    return df_out


def parse_rsna(rsna_path: str, processed_dir: str) -> pd.DataFrame:
    """Parse RSNA Pneumonia Detection dataset."""
    logger.info("Parsing RSNA Pneumonia Detection...")
    label_csv = os.path.join(rsna_path, "stage_2_train_labels.csv")
    if not os.path.exists(label_csv):
        logger.warning(f"RSNA label CSV not found at {label_csv}. Skipping.")
        return pd.DataFrame()

    df = pd.read_csv(label_csv)
    # RSNA: Target=1 means pneumonia
    df_grouped = df.groupby("patientId")["Target"].max().reset_index()
    records = []

    for _, row in tqdm(df_grouped.iterrows(), total=len(df_grouped), desc="RSNA"):
        pid = row["patientId"]
        target = int(row["Target"])

        # Try both DICOM and PNG
        img_src = None
        for ext in [".dcm", ".png", ".jpg"]:
            candidate = os.path.join(rsna_path, "stage_2_train_images", f"{pid}{ext}")
            if os.path.exists(candidate):
                img_src = candidate
                break

        if img_src is None:
            continue

        img_dst = os.path.join(processed_dir, "images", "rsna", f"{pid}.jpg")
        if not os.path.exists(img_dst):
            if not convert_to_rgb_and_resize(img_src, img_dst):
                continue

        records.append({
            "path": img_dst,
            "pneumonia": target,
            "tb": 0,
            "normal": 1 - target,
            "source": "RSNA"
        })

    df_out = pd.DataFrame(records)
    logger.info(f"RSNA: {len(df_out)} valid records parsed.")
    return df_out


def parse_tbx11k(tbx_path: str, processed_dir: str) -> pd.DataFrame:
    """Parse TBX11K Tuberculosis dataset."""
    logger.info("Parsing TBX11K...")
    # TBX11K has subfolders: TB-train, TB-test, NonTB-train, NonTB-test, etc.
    records = []

    label_map = {
        "TB": {"tb": 1, "pneumonia": 0, "normal": 0},
        "tb": {"tb": 1, "pneumonia": 0, "normal": 0},
        "Normal": {"tb": 0, "pneumonia": 0, "normal": 1},
        "normal": {"tb": 0, "pneumonia": 0, "normal": 1},
        "NonTB": {"tb": 0, "pneumonia": 0, "normal": 1},
        "sick": {"tb": 0, "pneumonia": 1, "normal": 0},
    }

    # Try to find annotation CSV
    ann_csv = os.path.join(tbx_path, "annotations.csv")
    if os.path.exists(ann_csv):
        df = pd.read_csv(ann_csv)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="TBX11K-csv"):
            img_src = os.path.join(tbx_path, row.get("filename", row.get("path", "")))
            label = row.get("label", row.get("class", "Normal"))
            lmap = label_map.get(str(label), {"tb": 0, "pneumonia": 0, "normal": 1})
            if not os.path.exists(img_src):
                continue
            fname = Path(img_src).name
            img_dst = os.path.join(processed_dir, "images", "tbx11k", fname)
            if not os.path.exists(img_dst):
                if not convert_to_rgb_and_resize(img_src, img_dst):
                    continue
            records.append({"path": img_dst, **lmap, "source": "TBX11K"})
    else:
        # Infer labels from folder names
        img_dir = os.path.join(tbx_path, "imgs") if os.path.exists(os.path.join(tbx_path, "imgs")) else tbx_path
        for folder in os.listdir(img_dir):
            folder_path = os.path.join(img_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            lmap = None
            for key in label_map:
                if key.lower() in folder.lower():
                    lmap = label_map[key]
                    break
            if lmap is None:
                continue
            for fname in tqdm(os.listdir(folder_path), desc=f"TBX11K-{folder}"):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                img_src = os.path.join(folder_path, fname)
                img_dst = os.path.join(processed_dir, "images", "tbx11k", fname)
                if not os.path.exists(img_dst):
                    if not convert_to_rgb_and_resize(img_src, img_dst):
                        continue
                records.append({"path": img_dst, **lmap, "source": "TBX11K"})

    df_out = pd.DataFrame(records)
    logger.info(f"TBX11K: {len(df_out)} valid records parsed.")
    return df_out


# ─── Dataset Assembly ────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate images using file hashes."""
    logger.info("Removing duplicate images...")
    hashes = {}
    keep = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Dedup"):
        if not os.path.exists(row["path"]):
            continue
        h = file_hash(row["path"])
        if h not in hashes:
            hashes[h] = idx
            keep.append(idx)
    df_clean = df.loc[keep].reset_index(drop=True)
    logger.info(f"After dedup: {len(df_clean)} records (removed {len(df) - len(df_clean)})")
    return df_clean


def balance_classes(df: pd.DataFrame, target_per_class: int = 6667) -> pd.DataFrame:
    """
    Balance dataset to target_per_class per class.
    Assigns each sample to a primary class for balancing purposes.
    """
    logger.info(f"Balancing classes to ~{target_per_class} per class...")

    # Assign primary class (TB > Pneumonia > Normal in clinical priority)
    def primary_class(row):
        if row["tb"] == 1:
            return "tb"
        elif row["pneumonia"] == 1:
            return "pneumonia"
        else:
            return "normal"

    df = df.copy()
    df["_primary"] = df.apply(primary_class, axis=1)

    balanced_parts = []
    for cls in ["pneumonia", "tb", "normal"]:
        subset = df[df["_primary"] == cls]
        n = min(len(subset), target_per_class)
        balanced_parts.append(subset.sample(n=n, random_state=42))

    df_bal = pd.concat(balanced_parts, ignore_index=True)
    df_bal = df_bal.drop(columns=["_primary"])
    df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)

    # Log distribution
    for cls in ["pneumonia", "tb", "normal"]:
        logger.info(f"  {cls}: {df_bal[cls].sum()} samples")

    return df_bal


def split_dataset(df: pd.DataFrame, train=0.70, val=0.15, test=0.15) -> pd.DataFrame:
    """Add split column (train/val/test) to dataframe."""
    logger.info("Splitting dataset 70/15/15...")
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(indices, test_size=(val + test), random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test / (val + test), random_state=42)

    splits = np.empty(len(df), dtype=object)
    splits[train_idx] = "train"
    splits[val_idx] = "val"
    splits[test_idx] = "test"
    df = df.copy()
    df["split"] = splits

    for s in ["train", "val", "test"]:
        logger.info(f"  {s}: {(df['split'] == s).sum()} samples")

    return df


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AstraMed Dataset Preparation")
    parser.add_argument("--nih_path", default="data/raw/NIH")
    parser.add_argument("--chexpert_path", default="data/raw/CheXpert")
    parser.add_argument("--rsna_path", default="data/raw/RSNA")
    parser.add_argument("--tbx_path", default="data/raw/TBX11K")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--target_per_class", type=int, default=6667)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)

    all_dfs = []

    if os.path.exists(args.nih_path):
        all_dfs.append(parse_nih(args.nih_path, args.output))
    if os.path.exists(args.chexpert_path):
        all_dfs.append(parse_chexpert(args.chexpert_path, args.output))
    if os.path.exists(args.rsna_path):
        all_dfs.append(parse_rsna(args.rsna_path, args.output))
    if os.path.exists(args.tbx_path):
        all_dfs.append(parse_tbx11k(args.tbx_path, args.output))

    if not all_dfs:
        logger.error("No datasets found. Please check the provided paths.")
        return

    df_all = pd.concat([d for d in all_dfs if not d.empty], ignore_index=True)
    logger.info(f"Combined: {len(df_all)} total records from {len(all_dfs)} sources")

    df_all = remove_duplicates(df_all)
    df_all = balance_classes(df_all, args.target_per_class)
    df_all = split_dataset(df_all)

    # Save master CSV
    out_csv = os.path.join(args.output, "master_dataset.csv")
    df_all.to_csv(out_csv, index=False)
    logger.info(f"\n✅ Master dataset saved: {out_csv}")
    logger.info(f"   Total samples: {len(df_all)}")
    logger.info(f"   Columns: {list(df_all.columns)}")

    # Save split CSVs
    for split in ["train", "val", "test"]:
        sub = df_all[df_all["split"] == split]
        sub.to_csv(os.path.join(args.output, f"{split}.csv"), index=False)
        logger.info(f"   {split}.csv: {len(sub)} samples")


if __name__ == "__main__":
    main()
