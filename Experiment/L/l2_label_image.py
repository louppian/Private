
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_components
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core as B  # noqa: E402


REPO = Path(B._REPO)
ROI = ["RT", "LT", "RB", "LB"]
RESULT_L = REPO / "Result" / "L"
FEATURES_PATH = RESULT_L / "l2_features.csv"
AUC_PATH = RESULT_L / "l2_auc.csv"

# CLI 대신 아래 상수를 직접 수정한다.
REBUILD = True
N_PERM = 1500
SEED = 0
BIN_WIDTH = 25.0

# B._load_img가 0~255 영상을 반환하면 1.0을 유지한다.
# 실제 입력이 0~1로 저장된 영상이라면 255.0으로 바꾼다.
INTENSITY_SCALE = 1.0

# PyRadiomics는 단일 voxel ROI를 허용하지 않는다.
MIN_ROI_PIXELS = 2

EXTRACTOR_TAG = "numpy-pyradiomics-like-2d-six-v2"
FEATS = [
    "mean",
    "median",
    "entropy",
    "uniformity",
    "glrlm_SRE",
    "glszm_SAE",
]

# distance=1인 2D GLRLM 방향. 방향 반대는 같은 run을 중복하므로 한쪽만 사용한다.
GLRLM_DIRECTIONS = (
    (0, 1),   # 0°: horizontal
    (1, 0),   # 90°: vertical
    (1, 1),   # 45° diagonal
    (1, -1),  # 135° diagonal
)

GLSZM_STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


def _year(pid):
    return {"24": 2024, "26": 2026}.get(str(pid)[:2])


def _prepare_intensity(img):
    """원본 intensity 단위를 유지하고 float64로 변환한다."""
    arr = np.asarray(img, dtype=np.float64) * INTENSITY_SCALE
    if arr.ndim != 2:
        raise ValueError(f"2D image required, got shape={arr.shape}")
    return arr


def _pyradiomics_bin_edges(values, bin_width=BIN_WIDTH):
    """
    PyRadiomics fixed-bin-width getBinEdges 규칙을 재현한다.

    lowBound = minimum - (minimum % binWidth)
    highBound = maximum + 2 * binWidth
    edges = arange(lowBound, highBound, binWidth)
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("ROI contains no finite intensity")
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    low_bound = minimum - (minimum % bin_width)
    high_bound = maximum + 2.0 * bin_width
    edges = np.arange(low_bound, high_bound, bin_width, dtype=np.float64)

    # PyRadiomics의 flat-region 방어 로직과 동일한 의미다.
    if edges.size == 1:
        edges = np.asarray([edges[0] - 0.5, edges[0] + 0.5], dtype=np.float64)
    return edges


def discretize_pyradiomics(img, roi_mask, bin_width=BIN_WIDTH):
    """
    ROI 내부만 PyRadiomics fixed-bin-width 규칙으로 1-based gray level로 변환한다.
    ROI 외부 및 non-finite voxel은 -1이다.
    """
    image = np.asarray(img, dtype=np.float64)
    mask = np.asarray(roi_mask, dtype=bool)
    if image.shape != mask.shape:
        raise ValueError(f"image/mask shape mismatch: {image.shape} vs {mask.shape}")

    valid_mask = mask & np.isfinite(image)
    quantized = np.full(image.shape, -1, dtype=np.int32)
    values = image[valid_mask]
    if values.size == 0:
        return quantized, valid_mask

    edges = _pyradiomics_bin_edges(values, bin_width=bin_width)
    quantized[valid_mask] = np.digitize(values, edges).astype(np.int32)
    return quantized, valid_mask


def first_order_features(img, valid_mask, quantized):
    """PyRadiomics 방식의 Mean, Median, Entropy, Uniformity를 계산한다."""
    values = np.asarray(img, dtype=np.float64)[valid_mask]
    if values.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "entropy": np.nan,
            "uniformity": np.nan,
        }

    levels = quantized[valid_mask]
    _, counts = np.unique(levels, return_counts=True)
    probabilities = counts.astype(np.float64) / float(counts.sum())
    eps = np.spacing(1.0)

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "entropy": float(-np.sum(probabilities * np.log2(probabilities + eps))),
        "uniformity": float(np.sum(probabilities ** 2)),
    }


def _glrlm_sre_one_direction(quantized, valid_mask, direction):
    """한 방향의 GLRLM을 암묵적으로 세어 Short Run Emphasis를 계산한다."""
    height, width = quantized.shape
    dr, dc = direction
    run_length_counts = {}

    rows, cols = np.nonzero(valid_mask)
    for row, col in zip(rows.tolist(), cols.tolist()):
        gray = int(quantized[row, col])

        # 같은 gray의 이전 voxel이 있으면 현재 voxel은 run 시작점이 아니다.
        prev_row = row - dr
        prev_col = col - dc
        if (
            0 <= prev_row < height
            and 0 <= prev_col < width
            and valid_mask[prev_row, prev_col]
            and int(quantized[prev_row, prev_col]) == gray
        ):
            continue

        length = 1
        next_row = row + dr
        next_col = col + dc
        while (
            0 <= next_row < height
            and 0 <= next_col < width
            and valid_mask[next_row, next_col]
            and int(quantized[next_row, next_col]) == gray
        ):
            length += 1
            next_row += dr
            next_col += dc

        run_length_counts[length] = run_length_counts.get(length, 0) + 1

    n_runs = int(sum(run_length_counts.values()))
    if n_runs == 0:
        return np.nan

    numerator = sum(count / (length ** 2) for length, count in run_length_counts.items())
    return float(numerator / n_runs)


def glrlm_short_run_emphasis(quantized, valid_mask):
    """
    PyRadiomics 기본처럼 각 2D 방향에서 SRE를 따로 계산한 뒤 산술평균한다.
    """
    direction_values = [
        _glrlm_sre_one_direction(quantized, valid_mask, direction)
        for direction in GLRLM_DIRECTIONS
    ]
    direction_values = np.asarray(direction_values, dtype=np.float64)
    if np.all(~np.isfinite(direction_values)):
        return np.nan
    return float(np.nanmean(direction_values))


def glszm_small_area_emphasis(quantized, valid_mask):
    """2D 8-connectivity GLSZM의 Small Area Emphasis를 계산한다."""
    gray_levels = np.unique(quantized[valid_mask])
    zone_sizes = []

    for gray in gray_levels:
        component_map, n_components = connected_components(
            valid_mask & (quantized == gray),
            structure=GLSZM_STRUCTURE_8,
        )
        if n_components == 0:
            continue
        sizes = np.bincount(component_map.ravel())[1:]
        zone_sizes.extend(int(size) for size in sizes if size > 0)

    n_zones = len(zone_sizes)
    if n_zones == 0:
        return np.nan

    sizes_array = np.asarray(zone_sizes, dtype=np.float64)
    return float(np.sum(1.0 / (sizes_array ** 2)) / n_zones)


def roi_feats(img, roi_mask):
    """한 ROI에서 6개 특징을 계산한다."""
    out = {name: np.nan for name in FEATS}
    quantized, valid_mask = discretize_pyradiomics(img, roi_mask, BIN_WIDTH)

    if int(np.count_nonzero(valid_mask)) < MIN_ROI_PIXELS:
        return out

    out.update(first_order_features(img, valid_mask, quantized))
    out["glrlm_SRE"] = glrlm_short_run_emphasis(quantized, valid_mask)
    out["glszm_SAE"] = glszm_small_area_emphasis(quantized, valid_mask)
    return out


def roi_pixel_masks(mask_bin):
    coords = B.split_lungs_to_four(mask_bin) or [
        (0, 0, 0.5, 0.5),
        (0, 0.5, 0.5, 1),
        (0.5, 0, 1, 0.5),
        (0.5, 0.5, 1, 1),
    ]
    height, width = mask_bin.shape
    out = np.zeros((4, height, width), dtype=bool)

    for i, (y0, x0, y1, x1) in enumerate(coords):
        yy0, yy1 = int(y0 * height), int(y1 * height)
        xx0, xx1 = int(x0 * width), int(x1 * width)
        out[i, yy0:yy1, xx0:xx1] = mask_bin[yy0:yy1, xx0:xx1] > 0
    return out


def build():
    df = pd.read_csv(B.CSV_PATH)
    has_image_path = "image_path" in df.columns
    records = []

    for n, (_, row) in enumerate(df.iterrows(), 1):
        uid = str(row[B.UID_COL])
        year = _year(row[B.PATIENT_COL])

        raw_img = B._load_img(uid, row["image_path"] if has_image_path else None)
        img = _prepare_intensity(raw_img)
        mask = np.asarray(B._load_mask_np(uid)) > 0
        if img.shape != mask.shape:
            raise ValueError(f"{uid}: image/mask shape mismatch: {img.shape} vs {mask.shape}")
        roi_masks = roi_pixel_masks(mask)

        record = {
            "uid": uid,
            "year": year,
            "patient": row[B.PATIENT_COL],
            "extractor": EXTRACTOR_TAG,
            "bin_width": BIN_WIDTH,
            "intensity_scale": INTENSITY_SCALE,
            "RT": int(row.RT),
            "LT": int(row.LT),
            "RB": int(row.RB),
            "LB": int(row.LB),
        }

        for i, roi in enumerate(ROI):
            for feature_name, value in roi_feats(img, roi_masks[i]).items():
                record[f"{roi}_{feature_name}"] = value

        records.append(record)
        if n % 100 == 0:
            print(f"  {n} imgs", end="\r")

    output = pd.DataFrame(records)
    RESULT_L.mkdir(parents=True, exist_ok=True)
    output.to_csv(FEATURES_PATH, index=False)
    print(f"\n[save] {FEATURES_PATH}  {output.shape}")
    return output


def _auc_directionless(pos, neg):
    """Mann–Whitney U 기반 방향 무관 AUC: max(AUC, 1-AUC)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]

    if len(pos) < 5 or len(neg) < 5:
        return np.nan

    ranks = rankdata(np.concatenate([pos, neg]), method="average")
    u_value = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    auc_value = u_value / (len(pos) * len(neg))
    return max(auc_value, 1.0 - auc_value)


def perm_maxauc(sub, roi, features, n_perm=N_PERM, seed=SEED):
    """3→4 경계에서 6개 특징의 최대 방향 무관 AUC와 순열 p값을 계산한다."""
    columns = [f"{roi}_{name}" for name in features]
    missing = [column for column in columns if column not in sub.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")

    positive = sub.loc[sub[roi] == 4, columns].to_numpy(float)
    negative = sub.loc[sub[roi] == 3, columns].to_numpy(float)
    if len(positive) < 5 or len(negative) < 5:
        return np.nan, "", np.nan

    observed = np.asarray(
        [_auc_directionless(positive[:, j], negative[:, j]) for j in range(len(columns))]
    )
    if np.all(np.isnan(observed)):
        return np.nan, "", np.nan

    observed_max = float(np.nanmax(observed))
    best_index = int(np.nanargmax(observed))
    best_feature = features[best_index]

    values = np.vstack([positive, negative])
    n_positive = len(positive)
    rng = np.random.default_rng(seed)
    exceed_count = 0
    valid_permutations = 0

    for _ in range(n_perm):
        index = rng.permutation(len(values))
        perm_positive = values[index[:n_positive]]
        perm_negative = values[index[n_positive:]]
        perm_auc = np.asarray(
            [
                _auc_directionless(perm_positive[:, j], perm_negative[:, j])
                for j in range(values.shape[1])
            ]
        )
        if np.all(np.isnan(perm_auc)):
            continue
        valid_permutations += 1
        if float(np.nanmax(perm_auc)) >= observed_max:
            exceed_count += 1

    if valid_permutations == 0:
        return observed_max, best_feature, np.nan

    p_value = (exceed_count + 1) / (valid_permutations + 1)
    return observed_max, best_feature, p_value


def export_l2(features_df, n_perm=N_PERM):
    """연도×ROI별 3→4 최대 AUC와 6-feature 순열 p값을 저장한다."""
    rows = []
    for year in (2024, 2026):
        subset = features_df[features_df.year == year]
        for roi in ROI:
            auc_value, feature_name, p_value = perm_maxauc(
                subset,
                roi,
                FEATS,
                n_perm=n_perm,
                seed=SEED,
            )
            rows.append(
                {
                    "year": year,
                    "roi": roi,
                    "boundary": "3to4",
                    "auc": round(auc_value, 4) if np.isfinite(auc_value) else "",
                    "feature": feature_name,
                    "perm_p": round(p_value, 4) if np.isfinite(p_value) else "",
                    "bonferroni_0p0125": (
                        int(p_value < 0.0125) if np.isfinite(p_value) else ""
                    ),
                    "extractor": EXTRACTOR_TAG,
                }
            )

    RESULT_L.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "year",
        "roi",
        "boundary",
        "auc",
        "feature",
        "perm_p",
        "bonferroni_0p0125",
        "extractor",
    ]
    with AUC_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[save] {AUC_PATH}")
    for row in rows:
        passed = row["bonferroni_0p0125"]
        status = "통과" if passed == 1 else ("미통과" if passed == 0 else "-")
        print(
            f"  {row['year']} {row['roi']:<3} 3→4 "
            f"AUC {row['auc']} ({row['feature']})  "
            f"perm p {row['perm_p']} [{status}]"
        )


def _cache_is_compatible(features_df):
    required = {
        "uid",
        "year",
        "patient",
        "extractor",
        "bin_width",
        "intensity_scale",
        *ROI,
        *(f"{roi}_{feature}" for roi in ROI for feature in FEATS),
    }
    if not required.issubset(features_df.columns):
        return False

    tags = features_df["extractor"].dropna().astype(str).unique()
    if len(tags) != 1 or tags[0] != EXTRACTOR_TAG:
        return False

    bin_widths = pd.to_numeric(features_df["bin_width"], errors="coerce").dropna().unique()
    scales = pd.to_numeric(features_df["intensity_scale"], errors="coerce").dropna().unique()
    return (
        len(bin_widths) == 1
        and np.isclose(bin_widths[0], BIN_WIDTH)
        and len(scales) == 1
        and np.isclose(scales[0], INTENSITY_SCALE)
    )


def main():
    if FEATURES_PATH.exists() and not REBUILD:
        cached = pd.read_csv(FEATURES_PATH)
        features_df = cached if _cache_is_compatible(cached) else build()
    else:
        features_df = build()

    export_l2(features_df, n_perm=N_PERM)


if __name__ == "__main__":
    main()