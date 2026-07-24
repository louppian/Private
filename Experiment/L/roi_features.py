# -*- coding: utf-8 -*-
r"""
L2 — 라벨-영상 정합 (draft §4.3) → Result/L/{roi_features.csv, l2_auc.csv}

ROI 픽셀 분포에서 특징을 뽑아 인접 등급을 영상만으로 가르는 능력을 방향무관 AUC 로 측정
(모델 배제 = 학습 불필요). 사전정렬 images_normalize + masks 를 uid 로 읽는다(core 로더).

특징(9, 1차): mean median max min skew kurt uniformity entropy opacity
ROI 추출: core.split_lungs_to_four(폐마스크→4분면) ∩ 폐마스크. 영역 [RT,LT,RB,LB].

산출:
  roi_features.csv : 이미지×ROI 특징 전체
  l2_auc.csv       : 3→4 경계 최고 AUC per (year, roi)  ← check_value_l 대조

⚠ md §4.3 표2 는 16-특징(1차4 + GLRLM6 + GLSZM6, 텍스처)을 썼다. 여기는 1차 9특징 subset
   이라 AUC 가 md 와 다를 수 있다(텍스처 배터리는 후속 확장). 절차·경계·방향무관 AUC 는 동일.

실행: python Experiment/L/roi_features.py
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Experiment (core)
import core as B                            # noqa: E402  로더·split·경로

REPO = Path(B._REPO)
ROI = ["RT", "LT", "RB", "LB"]
RESULT_L = REPO / "Result" / "L"
FEATS = ["mean", "median", "max", "min", "skew", "kurt", "uniformity", "entropy", "opacity"]
REF_L2 = {2024: {"RT": 0.657, "RB": 0.631, "LT": 0.926, "LB": 0.831},   # md 표2 (3→4)
          2026: {"RT": 0.876, "RB": 0.837, "LT": 0.930, "LB": 0.843}}


def _year(pid):
    return {"24": 2024, "26": 2026}.get(str(pid)[:2])


def feats(v, lung_mean):
    if len(v) < 20:
        return {k: np.nan for k in FEATS}
    hist, _ = np.histogram(v, bins=32, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    nz = p[p > 0]
    return dict(mean=v.mean(), median=float(np.median(v)), max=float(v.max()), min=float(v.min()),
                skew=float(skew(v)), kurt=float(kurtosis(v)),
                uniformity=float((p ** 2).sum()), entropy=float(-(nz * np.log2(nz)).sum()),
                opacity=float((v > lung_mean).mean()))


def roi_pixel_masks(mask_bin):
    coords = B.split_lungs_to_four(mask_bin) or [(0, 0, .5, .5), (0, .5, .5, 1), (.5, 0, 1, .5), (.5, .5, 1, 1)]
    H, W = mask_bin.shape
    out = np.zeros((4, H, W), np.uint8)
    for i, (y0, x0, y1, x1) in enumerate(coords):
        yy0, yy1, xx0, xx1 = int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W)
        out[i, yy0:yy1, xx0:xx1] = mask_bin[yy0:yy1, xx0:xx1]
    return out


def build():
    df = pd.read_csv(B.CSV_PATH)
    has_ip = "image_path" in df.columns
    recs = []
    for _, r in df.iterrows():
        uid = str(r[B.UID_COL]); yr = _year(r[B.PATIENT_COL])
        img = np.asarray(B._load_img(uid, r["image_path"] if has_ip else None),
                         dtype=np.float32) / 255.0          # 512 gray (사전정렬)
        msk = B._load_mask_np(uid).astype(np.uint8)          # 512 binary
        lung = img[msk > 0]
        lm = float(lung.mean()) if len(lung) else 0.5
        rmasks = roi_pixel_masks(msk)
        rec = dict(uid=uid, year=yr, patient=r[B.PATIENT_COL],
                   RT=int(r.RT), LT=int(r.LT), RB=int(r.RB), LB=int(r.LB))
        for i, roi in enumerate(ROI):
            for k, val in feats(img[rmasks[i] > 0], lm).items():
                rec[f"{roi}_{k}"] = val
        recs.append(rec)
    out = pd.DataFrame(recs)
    RESULT_L.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULT_L / "roi_features.csv", index=False)
    print(f"[save] {RESULT_L / 'roi_features.csv'}  {out.shape}")
    return out


def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    allv = np.concatenate([pos, neg]); rk = pd.Series(allv).rank().values
    U = rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    a = U / (len(pos) * len(neg))
    return max(a, 1 - a)                     # 방향 무관 분리도


def export_l2(fe):
    """3→4 경계 최고 AUC per (year, roi) → l2_auc.csv (check_value_l 대조)."""
    rows = []
    for yr in (2024, 2026):
        sub = fe[fe.year == yr]
        for roi in ROI:
            pos, neg = sub[sub[roi] == 4], sub[sub[roi] == 3]
            best_a, best_f = np.nan, ""
            for k in FEATS:
                a = auc(pos[f"{roi}_{k}"].values, neg[f"{roi}_{k}"].values)
                if not np.isnan(a) and (np.isnan(best_a) or a > best_a):
                    best_a, best_f = a, k
            rows.append(dict(year=yr, roi=roi, boundary="3to4",
                             auc=round(best_a, 4) if not np.isnan(best_a) else "",
                             feature=best_f, ref=REF_L2[yr][roi]))
    with open(RESULT_L / "l2_auc.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["year", "roi", "boundary", "auc", "feature", "ref"])
        w.writeheader(); w.writerows(rows)
    print(f"[save] {RESULT_L / 'l2_auc.csv'}")
    for r in rows:
        print(f"  {r['year']} {r['roi']:<3} 3→4 AUC {r['auc']} ({r['feature']})  ref {r['ref']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="features 재계산(기본: 있으면 재사용)")
    args = ap.parse_args()
    fpath = RESULT_L / "roi_features.csv"
    fe = pd.read_csv(fpath) if (fpath.exists() and not args.rebuild) else build()
    export_l2(fe)


if __name__ == "__main__":
    main()
