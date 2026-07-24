# -*- coding: utf-8 -*-
r"""
L2 확장 — ROI 픽셀 분포에서 특징 배터리 추출 후 인접경계 AUC 전부 계산.

특징(9): 평균 median max min skew(비대칭도) kurtosis(편평도)
         uniformity(균일도=Σp²) entropy(엔트로피) opacity(불투명비율=폐평균보다 밝은 비율)
ROI 추출: split_lungs_to_four(폐마스크→4분면 box) ∩ 폐마스크 (roi_intensity_full.csv 와 동일 규약)
  - 2024: image_normalize + mask_normalize (디스크)
  - 2026: raw → seg+align (dorga 파이프라인 재사용)

산출: roi_features_full.csv (이미지×ROI 특징) + 콘솔에 경계별 AUC 표
실행: python roi_features.py
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import skew, kurtosis
from skimage.measure import label, regionprops

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Experiment (core)
import core as B                            # 학습 코어(자립) — 舊 dorga_train 대체

MANIFEST = B.CSV_PATH
IMG2024 = Path(r"D:\MICCAI2026\inhauh\image_normalize")
MSK2024 = Path(r"D:\MICCAI2026\inhauh\mask_normalize")
ROI = ["RT", "LT", "RB", "LB"]             # 영역 순서 통일(core 기준)
OUT = B.OUT_ROOT.parent.parent / "Result" / "L" / "roi_features.csv"   # Result/L/


def to512(a, nearest=False):
    im = Image.fromarray(a)
    if im.size != (512, 512):
        im = im.resize((512, 512), Image.NEAREST if nearest else Image.BILINEAR)
    return np.array(im)


def load_2024(img_path):
    img = np.array(Image.open(img_path).convert("L")).astype(np.float32) / 255.0
    stem = Path(img_path).name
    msk = np.array(Image.open(MSK2024 / stem).convert("L"))
    return to512(img), (to512(msk, nearest=True) > 0).astype(np.uint8)


def roi_pixel_masks(mask_bin):
    coords = B.split_lungs_to_four(mask_bin)
    if coords is None:
        coords = [(0, 0, .5, .5), (.5, 0, 1, .5), (0, .5, .5, 1), (.5, .5, 1, 1)]
    H, W = mask_bin.shape
    out = np.zeros((4, H, W), np.uint8)
    for i, (y0, x0, y1, x1) in enumerate(coords):
        yy0, yy1, xx0, xx1 = int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W)
        out[i, yy0:yy1, xx0:xx1] = mask_bin[yy0:yy1, xx0:xx1]
    return out


def feats(v, lung_mean):
    """ROI 픽셀 배열 v(∈[0,1]) → 9 특징 dict."""
    if len(v) < 20:
        return {k: np.nan for k in FEATS}
    hist, _ = np.histogram(v, bins=32, range=(0, 1), density=False)
    p = hist / max(hist.sum(), 1)
    nz = p[p > 0]
    return dict(mean=v.mean(), median=np.median(v), max=v.max(), min=v.min(),
                skew=float(skew(v)), kurt=float(kurtosis(v)),
                uniformity=float((p ** 2).sum()), entropy=float(-(nz * np.log2(nz)).sum()),
                opacity=float((v > lung_mean).mean()))

FEATS = ["mean", "median", "max", "min", "skew", "kurt", "uniformity", "entropy", "opacity"]


def build():
    df = pd.read_csv(MANIFEST)
    # 2026 캐시 (seg+align)
    raw26 = sorted({B.raw_path_from_npz(p) for p in df.loc[df.year == 2026, "image_path"]})
    cache = B.build_2026_cache(raw26)
    recs = []
    for _, r in df.iterrows():
        if int(r.year) == 2024:
            img, msk = load_2024(r.image_path)
        else:
            iu, mu = cache[B.raw_path_from_npz(r.image_path)]
            img, msk = iu.astype(np.float32) / 255.0, (mu > 127).astype(np.uint8)
        lung = img[msk > 0]
        lm = float(lung.mean()) if len(lung) else 0.5
        rmasks = roi_pixel_masks(msk)
        rec = dict(year=int(r.year), patient=r.patient,
                   RT=int(r.RT), LT=int(r.LT), RB=int(r.RB), LB=int(r.LB))
        for i, roi in enumerate(ROI):
            v = img[rmasks[i] > 0]
            for k, val in feats(v, lm).items():
                rec[f"{roi}_{k}"] = val
        recs.append(rec)
    out = pd.DataFrame(recs)
    out.to_csv(OUT, index=False)
    print("saved:", OUT, out.shape)
    return out


def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    allv = np.concatenate([pos, neg]); rk = pd.Series(allv).rank().values
    U = rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    a = U / (len(pos) * len(neg))
    return max(a, 1 - a)          # 방향 무관 분리도 (feature 가 어느 쪽으로든 가르면 됨)


def report(fe):
    print("\n" + "=" * 100)
    print("경계별 특징 분리도 AUC (방향무관, max(a,1-a)) — 굵게 볼 값: 강도(mean)로 안 갈리는데 다른 특징이 살리나?")
    for roi in ROI:
        for y in [2024, 2026]:
            sub = fe[fe.year == y]
            print(f"\n[{roi} {y}]  {'boundary':>9} " + "".join(f"{f:>7}" for f in FEATS))
            for g in range(4):
                pos = sub[sub[roi] == g + 1]; neg = sub[sub[roi] == g]
                if len(pos) < 5 or len(neg) < 5:
                    continue
                cells = "".join(f"{auc(pos[f'{roi}_{k}'].values, neg[f'{roi}_{k}'].values):>7.3f}"
                                if not np.isnan(auc(pos[f'{roi}_{k}'].values, neg[f'{roi}_{k}'].values)) else f"{'—':>7}"
                                for k in FEATS)
                print(f"           {g}→{g+1:<6} {cells}")


if __name__ == "__main__":
    if os.path.exists(OUT):
        fe = pd.read_csv(OUT); print("기존 feature CSV 로드:", fe.shape)
    else:
        fe = build()
    report(fe)
