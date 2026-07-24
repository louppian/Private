from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from skimage.measure import label, regionprops


IMAGE_DIR = Path("/shared/home/mai/JeongGeon/Private/CXR/Merged/images_normalize")
MASK_DIR = Path("/shared/home/mai/JeongGeon/Private/CXR/Merged/masks")
OUTPUT_DIR = Path("/shared/home/mai/JeongGeon/Private/roi_overlay_samples")

NUM_SAMPLES = 8
MIN_AREA = 1000
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def split_lungs_to_four(mask_bin, min_area: int = 1000):
    """
    폐 마스크(2D)를 좌/우 폐 bbox 기준으로 상/하 2등분한다.

    반환 순서:
        [RT, LT, RB, LB]

    box 형식:
        normalized (y0, x0, y1, x1)
    """
    comps = [
        p
        for p in regionprops(
            label((mask_bin > 0).astype(np.uint8))
        )
        if p.area >= min_area
    ]

    if len(comps) < 2:
        return None

    # 영상 좌측 = 환자 우폐
    R = min(comps, key=lambda p: p.centroid[1])
    # 영상 우측 = 환자 좌폐
    L = max(comps, key=lambda p: p.centroid[1])

    H, W = mask_bin.shape

    def halves(reg):
        y0, x0, y1, x1 = reg.bbox
        y_mid = int(round((y0 + y1) / 2))

        top = (y0, y_mid)
        bot = (y_mid, y1)

        return (x0, x1), top, bot

    (Rx0, Rx1), R_top, R_bot = halves(R)
    (Lx0, Lx1), L_top, L_bot = halves(L)

    return [
        (R_top[0] / H, Rx0 / W, R_top[1] / H, Rx1 / W),  # RT
        (L_top[0] / H, Lx0 / W, L_top[1] / H, Lx1 / W),  # LT
        (R_bot[0] / H, Rx0 / W, R_bot[1] / H, Rx1 / W),  # RB
        (L_bot[0] / H, Lx0 / W, L_bot[1] / H, Lx1 / W),  # LB
    ]


def load_image(path):
    """이미지를 RGB uint8로 불러온다."""
    image = np.asarray(Image.open(path))

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]

    image = image.astype(np.float32)

    # float 이미지 또는 고비트 영상도 시각화 가능하도록 min-max 정규화
    min_val = np.nanmin(image)
    max_val = np.nanmax(image)

    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)

    return np.clip(image * 255, 0, 255).astype(np.uint8)


def load_mask(path):
    """마스크를 2D binary array로 불러온다."""
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = np.asarray(Image.open(path))

    if mask.ndim == 3:
        mask = mask[..., 0]

    return mask > 0


def find_file_pairs(image_dir, mask_dir):
    """확장자와 관계없이 파일 stem으로 이미지와 마스크를 매칭한다."""
    valid_exts = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"
    }

    image_files = {
        p.stem: p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    }

    mask_files = {
        p.stem: p
        for p in mask_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    }

    common_stems = sorted(set(image_files) & set(mask_files))

    return [
        (stem, image_files[stem], mask_files[stem])
        for stem in common_stems
    ]


def make_region_overlay(image, mask, boxes, alpha=0.4):
    """
    bbox 전체가 아니라 bbox와 폐 마스크가 겹치는 부분만 색칠한다.
    """
    H, W = mask.shape

    if image.shape[:2] != mask.shape:
        image = np.asarray(
            Image.fromarray(image).resize(
                (W, H),
                resample=Image.Resampling.BILINEAR,
            )
        )

    overlay = image.astype(np.float32).copy()

    region_names = ["RT", "LT", "RB", "LB"]

    # RT=빨강, LT=초록, RB=파랑, LB=노랑
    colors = np.array([
        [255,   0,   0],
        [  0, 255,   0],
        [  0, 128, 255],
        [255, 220,   0],
    ], dtype=np.float32)

    pixel_boxes = []

    for name, color, box in zip(region_names, colors, boxes):
        y0 = int(round(box[0] * H))
        x0 = int(round(box[1] * W))
        y1 = int(round(box[2] * H))
        x1 = int(round(box[3] * W))

        y0 = np.clip(y0, 0, H)
        y1 = np.clip(y1, 0, H)
        x0 = np.clip(x0, 0, W)
        x1 = np.clip(x1, 0, W)

        region_mask = np.zeros((H, W), dtype=bool)
        region_mask[y0:y1, x0:x1] = True

        # 실제 폐 마스크 내부만 색칠
        region_mask &= mask

        overlay[region_mask] = (
            (1 - alpha) * overlay[region_mask]
            + alpha * color
        )

        pixel_boxes.append((name, x0, y0, x1, y1, color / 255.0))

    return overlay.astype(np.uint8), pixel_boxes


pairs = find_file_pairs(IMAGE_DIR, MASK_DIR)

if not pairs:
    raise RuntimeError(
        "이미지와 마스크의 공통 파일명을 찾지 못했습니다.\n"
        "이미지와 마스크의 stem이 같은지 확인하세요."
    )

print(f"매칭된 이미지-마스크 쌍: {len(pairs)}개")

random.seed(SEED)
random.shuffle(pairs)

valid_samples = []

for stem, image_path, mask_path in pairs:
    image = load_image(image_path)
    mask = load_mask(mask_path)

    boxes = split_lungs_to_four(mask, min_area=MIN_AREA)

    if boxes is None:
        print(f"[SKIP] 폐 성분 2개 미만: {stem}")
        continue

    overlay, pixel_boxes = make_region_overlay(
        image=image,
        mask=mask,
        boxes=boxes,
        alpha=0.45,
    )

    valid_samples.append(
        (stem, image, mask, overlay, pixel_boxes)
    )

    if len(valid_samples) >= NUM_SAMPLES:
        break

if not valid_samples:
    raise RuntimeError("조건을 만족하는 폐 마스크가 없습니다.")


ncols = 2
nrows = int(np.ceil(len(valid_samples) / ncols))

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(12, 6 * nrows),
)

axes = np.asarray(axes).reshape(-1)

for ax, sample in zip(axes, valid_samples):
    stem, image, mask, overlay, pixel_boxes = sample

    ax.imshow(overlay)

    for name, x0, y0, x1, y1, color in pixel_boxes:
        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)

        ax.text(
            x0 + 5,
            y0 + 18,
            name,
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox={
                "facecolor": color,
                "alpha": 0.8,
                "edgecolor": "none",
                "pad": 2,
            },
        )

    ax.set_title(stem)
    ax.axis("off")

for ax in axes[len(valid_samples):]:
    ax.axis("off")

fig.tight_layout()

grid_output_path = OUTPUT_DIR / "random_four_lung_regions.png"
fig.savefig(
    grid_output_path,
    dpi=180,
    bbox_inches="tight",
)

plt.show()

print(f"Grid 저장 완료: {grid_output_path}")


# 샘플별 개별 이미지도 저장
for stem, image, mask, overlay, pixel_boxes in valid_samples:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(overlay)

    for name, x0, y0, x1, y1, color in pixel_boxes:
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )

        ax.text(
            x0 + 5,
            y0 + 18,
            name,
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox={
                "facecolor": color,
                "alpha": 0.8,
                "edgecolor": "none",
                "pad": 2,
            },
        )

    ax.set_title(stem)
    ax.axis("off")
    fig.tight_layout()

    save_path = OUTPUT_DIR / f"{stem}_four_regions.png"
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

print(f"개별 결과 저장 폴더: {OUTPUT_DIR}")