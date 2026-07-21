#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline.py — GUNet 폐분할 + STN 정렬 정규화 파이프라인 & 폴더 변환 실행기

역할
  (1) PreprocessPipeline : 원본 이미지 → 폐분할(1024) → STN 정렬(512) → aligned 512
      = 프로젝트의 dorga_train_2024to2026.py::build_2026_cache 와 동일 연산.
  (2) __main__ 실행기 : images_original 폴더의 이미지를 정규화해 images_normalize 에 저장.

의존(로컬, 같은 폴더): stn.py, seg.py
의존(서드파티): torch, torchvision, torch_geometric, cv2, scipy, numpy, PIL, tqdm

저장 대상 = STN 정렬된 512×512 grayscale uint8 png.
  * photometric Normalize([0.56],[0.17])는 학습 로더에서 적용 → 여기서 저장하지 않는다.
  * lung mask는 --save-mask 로 별도 저장(정렬 이미지와 같은 STN 좌표계, 다운스트림 4-ROI split 용).
    기본 저장 폴더 = CXR/Merged/masks (train.py MASK_DIR 와 동일).

사용법
    python pipeline.py                      # DRY-RUN(기본): 개수·매핑·skip만 출력
    python pipeline.py --execute            # 실제 변환
    python pipeline.py --execute --save-mask
    python pipeline.py --seg <finetuned_9601.pt> --stn <stn_weights.pth> \\
                       --src <images_original> --dst <images_normalize>
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from tqdm import tqdm
except Exception:                                   # tqdm 없으면 무동작 래퍼
    def tqdm(x, **k): return x

# 같은 폴더의 로컬 모듈
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stn import STN                                  # noqa: E402
from seg import LungSegmenter                        # noqa: E402


# ─────────────────────────── 기본 경로(클러스터 기준으로 수정) ───────────────
SEG_W = "/shared/home/mai/JeongGeon/Private/Model/SegSTN/weights/finetuned_9601.pt"
STN_W = "/shared/home/mai/JeongGeon/Private/Model/SegSTN/weights/stn_weights.pth"
SRC   = "/shared/home/mai/JeongGeon/Private/CXR/Merged/images_original"
DST   = "/shared/home/mai/JeongGeon/Private/CXR/Merged/images_normalize"
MASK  = "/shared/home/mai/JeongGeon/Private/CXR/Merged/masks"   # train.py MASK_DIR 와 동일

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ═══════════════════════════════════════════════════════════════════
# PreprocessPipeline : seg + STN align  (원 dorga/preprocessing/pipeline.py)
# ═══════════════════════════════════════════════════════════════════
class PreprocessPipeline(nn.Module):
    def __init__(self, seg_weights: str, stn_weights: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.segmenter = LungSegmenter(seg_weights, device=device)
        self.stn = STN(in_shape=(1, 512, 512))
        state = torch.load(stn_weights, map_location=self.device, weights_only=False)
        self.stn.load_state_dict(state)
        self.stn.to(self.device).eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """images: [B,1,H,W] float[0,1] → (aligned_img[B,1,512,512], aligned_mask[B,1,512,512]) cpu.

        이미지와 마스크에 '동일한' affine grid 를 적용해 둘을 같은 좌표계로 정렬한다.
        (과거 버그: 이미지에만 theta 를 적용하고 정렬 전 원본 좌표계 마스크를 반환 →
         images_normalize(정렬 좌표계)와 mask(원본 좌표계)가 어긋나 4-ROI 오버레이가
         폐 위치와 불일치했다. 반환 마스크는 이제 이미지와 같은 정렬 좌표계의 binary.)
        """
        images = images.float()
        images_1024 = F.interpolate(images, size=(1024, 1024), mode="bilinear", align_corners=False)
        images_512 = F.interpolate(images, size=(512, 512), mode="bilinear",
                                   align_corners=False).to(self.device)

        masks_1024 = self.segmenter(images_1024)                    # [B,1,1024,1024] cpu {0,1} · 원본 좌표계
        mask_512 = F.interpolate(masks_1024.float(), size=(512, 512),
                                 mode="nearest").to(self.device)

        theta = self.stn(mask_512)                                  # 마스크로 affine 추정
        grid = F.affine_grid(theta, images_512.size(), align_corners=False)   # 이미지·마스크 공용 grid

        aligned_img = F.grid_sample(images_512, grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
        aligned_mask = F.grid_sample(mask_512, grid, mode="nearest",
                                     padding_mode="zeros", align_corners=False) > 0.5
        return aligned_img.cpu(), aligned_mask.cpu()


# ═══════════════════════════════════════════════════════════════════
# 폴더 변환 실행기
# ═══════════════════════════════════════════════════════════════════
def list_images(src):
    out = []
    for root, _, names in os.walk(src):
        for n in names:
            if n.lower().endswith(EXTS):
                out.append(os.path.join(root, n))
    return sorted(out)


def load_gray_tensor(path):
    arr = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, None]                        # [1,1,H,W]


def main():
    ap = argparse.ArgumentParser(description="images_original → images_normalize (GUNet seg + STN align)")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--seg", default=SEG_W)
    ap.add_argument("--stn", default=STN_W)
    ap.add_argument("--mask-dst", default=MASK, help="마스크 저장 폴더(기본: CXR/Merged/masks = train.py MASK_DIR)")
    ap.add_argument("--save-mask", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="기존 출력도 덮어씀(기본: skip)")
    ap.add_argument("--execute", action="store_true", help="미지정 시 DRY-RUN")
    args = ap.parse_args()

    mask_dst = args.mask_dst

    def dst_of(src_path, base):
        rel = os.path.relpath(src_path, args.src)
        return os.path.join(base, os.path.splitext(rel)[0] + ".png")

    # [1] 준비검증
    for w in (args.seg, args.stn):
        if not os.path.exists(w):
            sys.exit(f"[중단] 가중치 없음: {w}")
    if not os.path.isdir(args.src):
        sys.exit(f"[중단] 입력 폴더 없음: {args.src}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    files = list_images(args.src)

    def needs_processing(f):
        if args.overwrite:
            return True
        if not os.path.exists(dst_of(f, args.dst)):            # 정규화 이미지 없음
            return True
        if args.save_mask and not os.path.exists(dst_of(f, mask_dst)):  # 마스크만 없음
            return True
        return False

    todo = [f for f in files if needs_processing(f)]
    print(f"=== normalize ({'EXECUTE' if args.execute else 'DRY-RUN'}) · device={dev} ===")
    print(f"  src={args.src}\n  dst={args.dst}")
    print(f"  입력 {len(files)}장 · 변환대상 {len(todo)}장 · skip {len(files)-len(todo)}"
          f"{' · +mask' if args.save_mask else ''}")

    if not args.execute:
        for f in files[:5]:
            print("   ", os.path.relpath(f, args.src), "->", os.path.relpath(dst_of(f, args.dst), args.dst))
        print("(DRY-RUN) 실제 변환 없음. --execute 로 실행.")
        return

    # [4] 파이프라인 로드(1회) → 이미지별 변환(원본 해상도 그대로 → 파이프라인이 내부 interp, build_2026_cache와 동일)
    pipe = PreprocessPipeline(seg_weights=args.seg, stn_weights=args.stn, device=dev)
    failed = []
    for f in tqdm(todo, desc="seg+align"):
        try:
            aligned, aligned_mask = pipe(load_gray_tensor(f))
            out = dst_of(f, args.dst)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            im = (aligned[0, 0].numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(im).save(out)
            if args.save_mask:
                mout = dst_of(f, mask_dst)
                os.makedirs(os.path.dirname(mout), exist_ok=True)
                mk = aligned_mask[0, 0].numpy()                # 이미지와 동일 좌표계(정렬됨), resize 금지
                Image.fromarray((mk * 255).astype(np.uint8)).save(mout)
        except Exception as e:                       # seg 실패 등은 기록하고 계속
            failed.append(f"{f}\t{type(e).__name__}: {e}")

    if failed:
        log = os.path.join(args.dst, "failed.log")
        os.makedirs(args.dst, exist_ok=True)
        with open(log, "w", encoding="utf-8") as fp:
            fp.write("\n".join(failed))
        print(f"[경고] 실패 {len(failed)}건 → {log}")

    # [5] 검증
    n_out = len(list_images(args.dst))
    print(f"[검증] 출력 {n_out}장 · 입력 {len(files)}장 · 실패 {len(failed)}건")


if __name__ == "__main__":
    main()
