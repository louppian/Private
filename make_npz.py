"""
Build Phase 2A mini-sequence NPZ files for CXR_GUI_LabelOnly.

Input
-----
Result/Phase2A_v2/phase2A_master_1155.csv
  - 1,155 presentations
  - 165 mini-sequences
  - contains ms_id, seq2, uid2, uid, and image_path

Output
------
One NPZ per mini-sequence:
  P2A_M001.npz ... P2A_M165.npz

Each NPZ is compatible with D:/CXR_GUI_LabelOnly/app.py:
  patient_id, user_id, timestamp, gallery, original_images,
  inference, inference_edit, inference_posthoc

Image source
------------
2026 presentations are read from RAW_IMAGE (untouched JPEG), not NPZ_IMAGE.
The NPZ_IMAGE copies carry a burned-in cyan frame plus a fixed 1024 size,
which lets a reader tell the acquisition year from the image alone.
2024 presentations have no RAW counterpart and stay on image_original.

All images are stored as uint8 1024x1024 single channel, stacked into one
(7, 1024, 1024) array. GUNet takes 1 channel at a fixed 1024 input size and STN
takes 1 channel, so that is what the GUI needs; every source is R==G==B anyway.
`gallery` holds uid2 labels only: app.py builds its gallery from
`original_images`, so storing the pixels twice just doubled the file size.

Non-disclosure rule
-------------------
The NPZ files contain only the presentation identifiers, images, and blank
score frames. They do not contain original uid, pid, year, prior grades,
selection role, or ROI role columns.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MAPPING_CSV = BASE_DIR / "Result" / "Phase2A_v2" / "phase2A_master_1155.csv"
OUTPUT_DIR = Path(r"D:\CXR_GUI_LabelOnly\phase2a_npz")
LOCAL_CXR_ROOT = Path(r"D:\InhaUH_CXR")

IMAGE_EXTS = (".dcm", ".dicom", ".png", ".jpg", ".jpeg")
EXPECTED_PRESENTATIONS = 1155
EXPECTED_SEQUENCES = 165
EXPECTED_SEQUENCE_LEN = 7
BLANK_SCORE = 0
SERVER_CXR_PREFIX = "/shared/home/mai/JeongGeon/Private/CXR/"
TARGET_SIZE = 1024
RAW_EXTS = (".jpg", ".jpeg", ".png")


def load_mapping() -> pd.DataFrame:
    df = pd.read_csv(MAPPING_CSV, encoding="utf-8-sig")
    required = {
        "read_order", "ms_id", "seq2", "uid2", "uid", "image_path",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"mapping_key.csv missing columns: {sorted(missing)}")

    df = df.sort_values(["ms_id", "seq2", "read_order"]).reset_index(drop=True)

    n_presentations = len(df)
    n_seq = df["ms_id"].nunique()
    seq_sizes = df.groupby("ms_id").size()
    bad_sizes = seq_sizes[seq_sizes != EXPECTED_SEQUENCE_LEN]

    if n_presentations != EXPECTED_PRESENTATIONS:
        raise ValueError(f"Expected {EXPECTED_PRESENTATIONS} presentations, got {n_presentations}")
    if n_seq != EXPECTED_SEQUENCES:
        raise ValueError(f"Expected {EXPECTED_SEQUENCES} sequences, got {n_seq}")
    if not bad_sizes.empty:
        raise ValueError(f"Sequences not length {EXPECTED_SEQUENCE_LEN}: {bad_sizes.to_dict()}")

    return df


def resolve_raw_image(npz_image_path: Path) -> Path | None:
    """NPZ_IMAGE/01_a0001.png -> RAW_IMAGE/A0001.jpg

    The two trees use different naming: NPZ_IMAGE prefixes a sequence number and
    lowercases the id, RAW_IMAGE keeps the plain uppercase id. Cases with split
    studies have NPZ_IMAGE1..4 paired with RAW_IMAGE1..4, but some pair every
    NPZ_IMAGEn with a single RAW_IMAGE, so try the numbered dir first.
    """
    stem = re.sub(r"^\d+_", "", npz_image_path.stem).upper()
    parent = npz_image_path.parent
    candidate_dirs = [
        parent.parent / parent.name.replace("NPZ_IMAGE", "RAW_IMAGE"),
        parent.parent / "RAW_IMAGE",
    ]
    for directory in candidate_dirs:
        for ext in RAW_EXTS:
            candidate = directory / (stem + ext)
            if candidate.exists():
                return candidate
    return None


def remap_image_path(server_path: str) -> Path:
    path = str(server_path).strip()
    if path.startswith(SERVER_CXR_PREFIX):
        rel = path[len(SERVER_CXR_PREFIX):]
        if rel.startswith("2026 CXR/"):
            rel = "2026.05 CXRs/Dataset/" + rel[len("2026 CXR/"):]
        local = LOCAL_CXR_ROOT / Path(rel)
        if local.parent.name.startswith("NPZ_IMAGE"):
            raw = resolve_raw_image(local)
            if raw is None:
                raise FileNotFoundError(f"No RAW_IMAGE counterpart for {local}")
            return raw
        return local
    return Path(path)


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix in {".png", ".jpg", ".jpeg"}:
        img = Image.open(path).convert("L")

    elif suffix in {".dcm", ".dicom"}:
        try:
            import pydicom
        except ImportError as exc:
            raise RuntimeError("pydicom is required to load DICOM files") from exc

        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr).convert("L")

    else:
        raise ValueError(f"Unsupported image file extension: {path}")

    # RAW_IMAGE files are full detector resolution and not square. app.py's
    # preprocess() does the same non-aspect-preserving resize to 1024, and
    # GUNet's fc_mu is wired to inputsize 1024, so 1024 is not negotiable.
    if img.size != (TARGET_SIZE, TARGET_SIZE):
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)

    # Single channel: GUNet takes 1 channel and STN takes 1 channel. Every
    # source is R==G==B, so dropping to L loses nothing.
    return np.array(img, dtype=np.uint8)


def build_npz_for_sequence(ms_id: str, rows: pd.DataFrame, image_map: dict[str, Path]) -> Path:
    images = []
    gallery_labels = []

    for row in rows.sort_values("seq2").itertuples():
        uid2 = str(row.uid2)
        path = image_map.get(uid2)
        if path is None or not path.exists():
            raise FileNotFoundError(f"No image file for uid2={uid2}")

        images.append(load_image(path))
        gallery_labels.append(uid2)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path = OUTPUT_DIR / f"{ms_id}.npz"
    blank_scores = [[BLANK_SCORE, BLANK_SCORE, BLANK_SCORE, BLANK_SCORE] for _ in images]
    np.savez(
        out_path,
        patient_id=ms_id,
        user_id="Phase2A",
        timestamp=now,
        gallery=np.array(gallery_labels),
        original_images=np.stack(images),
        inference=np.array(blank_scores, dtype=object),
        inference_edit=np.array(blank_scores, dtype=object),
        inference_posthoc=np.array(None, dtype=object),
    )
    return out_path


def verify_npz(path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    required = {
        "patient_id", "user_id", "timestamp", "gallery",
        "original_images", "inference", "inference_edit", "inference_posthoc",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{path.name} missing keys: {sorted(missing)}")
    if len(data["original_images"]) != EXPECTED_SEQUENCE_LEN:
        raise ValueError(f"{path.name} image count is not {EXPECTED_SEQUENCE_LEN}")


def main() -> int:
    mapping = load_mapping()
    print(f"mapping ok: {len(mapping)} presentations, {mapping.ms_id.nunique()} sequences")

    image_map = {
        str(r.uid2): remap_image_path(str(r.image_path))
        for r in mapping.itertuples()
    }
    missing_uid2 = sorted(k for k, v in image_map.items() if not v.exists())
    if missing_uid2:
        print(f"missing image files for {len(missing_uid2)} presentation id(s)")
        print("first missing:", ", ".join(missing_uid2[:20]))
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for ms_id, rows in mapping.groupby("ms_id", sort=True):
        out_path = build_npz_for_sequence(ms_id, rows, image_map)
        verify_npz(out_path)
        manifest_rows.append({
            "npz_file": out_path.name,
            "ms_id": ms_id,
            "presentations": len(rows),
            "uid2_list": "|".join(rows.sort_values("seq2")["uid2"].astype(str)),
            "read_order_min": int(rows["read_order"].min()),
            "read_order_max": int(rows["read_order"].max()),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUTPUT_DIR / "phase2a_npz_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"wrote {len(manifest)} npz files to {OUTPUT_DIR}")
    print(f"wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
