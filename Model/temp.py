import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from SegSTN.seg import LungSegmenter
from SegSTN.stn import STN
from BSNet.BSNet import make_boxes
"""
정렬 마스크 사전 저장.

파이프라인(pipeline.py)과 동일한 순서로 원본에서 마스크를 만들고, STN theta 를
그 마스크에 적용해 '정렬 마스크'를 저장한다. 학습은 정렬 이미지(images_normalize)를
쓰므로 마스크도 같은 theta 로 정렬돼 있어야 정합이 맞는다.

순서:
  원본 -> seg(1024) -> mask_1024 -> mask_512(nearest)
       -> STN(mask_512) = theta
       -> aligned_mask = STN.transform(mask_512, theta)   # 이미지와 동일 theta
  저장: MASK_DIR/<uid>.png  (0/255 uint8, 512x512)

파일명(uid) 은 images_normalize 와 동일 규칙(원본 상대경로의 stem)이라
dataset 이 그대로 짝짓는다.

실행:
    python cache_masks.py
    python cache_masks.py --overwrite
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SEG_W = "/shared/home/mai/JeongGeon/Private/Model/SegSTN/weights/finetuned_9601.pt"
STN_W = "/shared/home/mai/JeongGeon/Private/Model/SegSTN/weights/stn_weights.pth"
SRC   = "/shared/home/mai/JeongGeon/Private/CXR/Merged/images_original"
MASK_DIR = "/shared/home/mai/JeongGeon/Private/CXR/Merged/masks"

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
OUT_SIZE = 512


def list_images(src):
    out = []
    for root, _, names in os.walk(src):
        for n in names:
            if n.lower().endswith(EXTS):
                out.append(os.path.join(root, n))
    return sorted(out)


def uid_of(path, src):
    # images_normalize 와 동일: 원본 상대경로의 확장자 제거 stem
    rel = os.path.relpath(path, src)
    return os.path.splitext(rel)[0]


def load_gray(path):
    arr = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, None]            # [1,1,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for w in (SEG_W, STN_W):
        if not os.path.exists(w):
            sys.exit(f"[중단] 가중치 없음: {w}")
    os.makedirs(MASK_DIR, exist_ok=True)

    seg = LungSegmenter(SEG_W, device=device)
    stn = STN(in_shape=(1, 512, 512))
    stn.load_state_dict(torch.load(STN_W, map_location=device, weights_only=False))
    stn.to(device).eval()

    files = list_images(SRC)
    done = skip = fail = 0
    failed = []

    for f in files:
        uid = uid_of(f, SRC)
        out_path = os.path.join(MASK_DIR, uid + ".png")
        if os.path.exists(out_path) and not args.overwrite:
            skip += 1
            continue
        try:
            img = load_gray(f).to(device)
            img_1024 = F.interpolate(img, size=(1024, 1024), mode="bilinear",
                                     align_corners=False)
            with torch.no_grad():
                mask_1024 = seg(img_1024).to(device)           # [1,1,1024,1024] {0,1}
                mask_512 = F.interpolate(mask_1024.float(), size=(512, 512),
                                         mode="nearest")
                theta = stn(mask_512)                          # 마스크로 theta
                aligned_mask = STN.transform(mask_512, theta)  # 이미지와 동일 theta 적용

            m = (aligned_mask[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.fromarray(m, mode="L").save(out_path)
            done += 1
            if done % 50 == 0:
                print(f"  {done} done", end="\r")
        except Exception as e:
            fail += 1
            failed.append(f"{f}\t{type(e).__name__}: {e}")

    print(f"\n정렬 마스크 저장: {done}개 생성, {skip}개 skip, {fail}개 실패 -> {MASK_DIR}")

    if failed:
        log = os.path.join(MASK_DIR, "failed.log")
        with open(log, "w", encoding="utf-8") as fp:
            fp.write("\n".join(failed))
        print(f"[경고] 실패 {fail}건 -> {log}")

    # 검증: 첫 성공 마스크의 폐 면적 비율
    ok = [uid_of(f, SRC) for f in files
          if os.path.exists(os.path.join(MASK_DIR, uid_of(f, SRC) + ".png"))]
    if ok:
        m0 = np.asarray(Image.open(os.path.join(MASK_DIR, ok[0] + ".png"))) / 255.0
        frac = m0.mean()
        print(f"[검증] {ok[0]} 폐 면적 {frac:.3f} "
              f"({'정상' if 0.03 < frac < 0.5 else '이상 - 정렬/좌표 확인'})")


if __name__ == "__main__":
    main()