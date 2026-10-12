import os
import shutil

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import IMAGES_DIR, LABELS_DIR, DOCS_DIR, YOLO_DIR, IMAGE_EXTS, SEED

SPLIT_CSV = DOCS_DIR / "split_assignment.csv"
SPLIT_NAMES = ["train", "val", "test"]


def make_split(photo_df, seed=SEED, n_splits=7):
    """Photographer-grouped, rarest-class-stratified split.
    photo_df needs the columns 'rarest' and 'photographer_id'.
    Fold 0 -> test, fold 1 -> val, the rest -> train."""
    df = photo_df.reset_index(drop=True)
    fold = pd.Series(-1, index=df.index)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (_, test_idx) in enumerate(
            sgkf.split(df, y=df["rarest"], groups=df["photographer_id"])):
        fold.iloc[test_idx] = f
    if (fold < 0).any():
        raise ValueError("Some images were not assigned a fold")
    return fold.map({0: "test", 1: "val"}).fillna("train").values


def _link_or_copy(src, dst, use_links):
    """Hard link when possible (no extra disk space), otherwise copy."""
    if use_links:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def build_yolo_folders(use_links=True, overwrite=False):
    """Build data/yolo/{images,labels}/{train,val,test} from the saved split CSV."""
    df = pd.read_csv(SPLIT_CSV)

    if not {"stem", "photographer_id", "split"} <= set(df.columns):
        raise ValueError("split_assignments.csv is missing a required column")
    if df["split"].isna().any() or not set(df["split"]) <= set(SPLIT_NAMES):
        raise ValueError("split column has missing or unexpected values")
    if df["stem"].duplicated().any():
        raise ValueError("duplicate stems in split_assignments.csv")

    if YOLO_DIR.exists():
        if not overwrite:
            raise FileExistsError(f"{YOLO_DIR} exists; pass overwrite=True to rebuild it")
        shutil.rmtree(YOLO_DIR)  # generated data only, safe to rebuild

    for kind in ("images", "labels"):
        for s in SPLIT_NAMES:
            (YOLO_DIR / kind / s).mkdir(parents=True)

    image_files = {f.stem: f for f in IMAGES_DIR.iterdir()
                   if f.suffix.lower() in IMAGE_EXTS}

    missing_images, missing_labels = [], []
    for stem, split in zip(df["stem"], df["split"]):
        img = image_files.get(stem)
        lbl = LABELS_DIR / f"{stem}.txt"
        if img is None:
            missing_images.append(stem)
            continue
        if not lbl.exists():
            missing_labels.append(stem)
            continue
        _link_or_copy(img, YOLO_DIR / "images" / split / img.name, use_links)
        _link_or_copy(lbl, YOLO_DIR / "labels" / split / lbl.name, use_links)

    # --- Verify what was built ---
    expected = df["split"].value_counts().to_dict()
    summary = {"missing_images": len(missing_images),
               "missing_labels": len(missing_labels)}
    all_stems = []
    for s in SPLIT_NAMES:
        img_stems = {f.stem for f in (YOLO_DIR / "images" / s).iterdir()}
        lbl_stems = {f.stem for f in (YOLO_DIR / "labels" / s).iterdir()}
        summary[f"{s}_images"] = len(img_stems)
        summary[f"{s}_labels"] = len(lbl_stems)
        summary[f"{s}_expected"] = expected.get(s, 0)
        summary[f"{s}_unpaired"] = len(img_stems ^ lbl_stems)
        all_stems += list(img_stems)
    summary["stems_in_two_splits"] = len(all_stems) - len(set(all_stems))
    return summary