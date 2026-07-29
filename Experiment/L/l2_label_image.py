# -*- coding: utf-8 -*-
r"""
L2 — 라벨-영상 정합 (draft §4.3 / 부록 A) → Result/L/{l2_features.csv, l2_auc.csv}

각 ROI 픽셀 분포에서 **16-특징 배터리**를 뽑아 인접 등급을 영상만으로 가르는 능력을
방향무관 AUC(Mann-Whitney max(a,1−a))로 측정한다(모델 배제 = 학습 불필요).
사전정렬 images_normalize + masks 를 uid 로 읽는다(core 로더).

16-특징 (draft §4.3):
  1차(4)   : 평균 mean · 중앙값 median · 균일도 uniformity(Σp²) · 엔트로피 entropy
  GLRLM(6) : SRE LRE GLN RLN HGLRE LRHGLE       (Gray Level Run Length Matrix)
  GLSZM(6) : SAE LZE GLN SZN ZP HGLZE           (Gray Level Size Zone Matrix)
텍스처는 pyradiomics 없이 직접 구현(IBSI 정의). 그레이 Ng=16, ROI [min,max] 양자화,
GLRLM 4방향 합산, GLSZM 8-연결. 영역 [RT,LT,RB,LB].

산출:
  l2_features.csv : 이미지×ROI×16특징
  l2_auc.csv      : 3→4 경계 최고 AUC per (year, roi)  ← check_value_l 대조 (md 표2)

⚠ AUC 절대값은 양자화(Ng·binning) 규약에 따라 md 와 소폭 다를 수 있으나, 특징 정의·경계·
   방향무관 AUC 절차는 draft 와 동일하다.

실행: python Experiment/L/l2_label_image.py [--rebuild] [--ng 16]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Experiment (core)
import core as B                            # noqa: E402  로더·split·경로

REPO = Path(B._REPO)
ROI = ["RT", "LT", "RB", "LB"]
RESULT_L = REPO / "Result" / "L"
NG = 16                                     # 텍스처 그레이 레벨 수

GLRLM_F = ["glrlm_SRE", "glrlm_LRE", "glrlm_GLN", "glrlm_RLN", "glrlm_HGLRE", "glrlm_LRHGLE"]
GLSZM_F = ["glszm_SAE", "glszm_LZE", "glszm_GLN", "glszm_SZN", "glszm_ZP", "glszm_HGLZE"]
FEATS = ["mean", "median", "uniformity", "entropy"] + GLRLM_F + GLSZM_F   # 16

REF_L2 = {2024: {"RT": 0.657, "RB": 0.631, "LT": 0.926, "LB": 0.831},   # md 표2 (3→4 최고)
          2026: {"RT": 0.876, "RB": 0.837, "LT": 0.930, "LB": 0.843}}


def _year(pid):
    return {"24": 2024, "26": 2026}.get(str(pid)[:2])


# ─────────────────────────── 1차 특징(4) ───────────────────────────
def first_order(v):
    if len(v) < 20:
        return {k: np.nan for k in ["mean", "median", "uniformity", "entropy"]}
    hist, _ = np.histogram(v, bins=32, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    nz = p[p > 0]
    return dict(mean=float(v.mean()), median=float(np.median(v)),
                uniformity=float((p ** 2).sum()), entropy=float(-(nz * np.log2(nz)).sum()))


# ─────────── 양자화 (ROI [min,max] → 0..Ng-1, 밖은 -1) ───────────
def quantize(img, roi_mask, ng):
    q = np.full(img.shape, -1, dtype=np.int32)
    v = img[roi_mask]
    if v.size == 0:
        return q
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        q[roi_mask] = 0
        return q
    lvl = np.clip(((img - lo) / (hi - lo) * ng).astype(np.int32), 0, ng - 1)
    q[roi_mask] = lvl[roi_mask]
    return q


def _crop(q):
    ys, xs = np.where(q >= 0)
    if ys.size == 0:
        return None
    return q[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ─────────────────────────── GLRLM (4방향 합산) ───────────────────────────
def _rle(line, P):
    n = len(line); i = 0
    while i < n:
        if line[i] < 0:
            i += 1; continue
        j = i
        while j + 1 < n and line[j + 1] == line[i]:
            j += 1
        key = (int(line[i]), j - i + 1)
        P[key] = P.get(key, 0) + 1
        i = j + 1


def glrlm_mat(q):
    P = {}
    H, W = q.shape
    for r in range(H):
        _rle(q[r, :], P)
    for c in range(W):
        _rle(q[:, c], P)
    for off in range(-H + 1, W):
        _rle(np.diagonal(q, offset=off), P)
    qf = q[:, ::-1]
    for off in range(-H + 1, W):
        _rle(np.diagonal(qf, offset=off), P)
    return P


def glrlm_feats(P):
    Nr = sum(P.values())
    if Nr == 0:
        return {k: np.nan for k in GLRLM_F}
    gi, ri = {}, {}
    SRE = LRE = HGLRE = LRHGLE = 0.0
    for (i, j), c in P.items():
        SRE += c / j ** 2; LRE += c * j ** 2
        HGLRE += c * (i + 1) ** 2; LRHGLE += c * (i + 1) ** 2 * j ** 2
        gi[i] = gi.get(i, 0) + c; ri[j] = ri.get(j, 0) + c
    GLN = sum(x ** 2 for x in gi.values()); RLN = sum(x ** 2 for x in ri.values())
    return {"glrlm_SRE": SRE / Nr, "glrlm_LRE": LRE / Nr, "glrlm_GLN": GLN / Nr,
            "glrlm_RLN": RLN / Nr, "glrlm_HGLRE": HGLRE / Nr, "glrlm_LRHGLE": LRHGLE / Nr}


# ─────────────────────────── GLSZM (8-연결 존) ───────────────────────────
_S8 = np.ones((3, 3), dtype=int)


def glszm_mat(q, ng):
    P = {}
    for g in range(ng):
        lab, n = cc_label(q == g, structure=_S8)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]          # zone 별 픽셀 수
        for s in sizes:
            key = (g, int(s))
            P[key] = P.get(key, 0) + 1
    return P


def glszm_feats(P, n_pix):
    Nz = sum(P.values())
    if Nz == 0:
        return {k: np.nan for k in GLSZM_F}
    gi, si = {}, {}
    SAE = LZE = HGLZE = 0.0
    for (i, s), c in P.items():
        SAE += c / s ** 2; LZE += c * s ** 2; HGLZE += c * (i + 1) ** 2
        gi[i] = gi.get(i, 0) + c; si[s] = si.get(s, 0) + c
    GLN = sum(x ** 2 for x in gi.values()); SZN = sum(x ** 2 for x in si.values())
    return {"glszm_SAE": SAE / Nz, "glszm_LZE": LZE / Nz, "glszm_GLN": GLN / Nz,
            "glszm_SZN": SZN / Nz, "glszm_ZP": Nz / max(n_pix, 1), "glszm_HGLZE": HGLZE / Nz}


def roi_feats(img, roi_mask):
    v = img[roi_mask]
    out = first_order(v)
    if len(v) < 20:
        out.update({k: np.nan for k in GLRLM_F + GLSZM_F}); return out
    qc = _crop(quantize(img, roi_mask, NG))
    if qc is None:
        out.update({k: np.nan for k in GLRLM_F + GLSZM_F}); return out
    out.update(glrlm_feats(glrlm_mat(qc)))
    out.update(glszm_feats(glszm_mat(qc, NG), n_pix=int(roi_mask.sum())))
    return out


def roi_pixel_masks(mask_bin):
    coords = B.split_lungs_to_four(mask_bin) or [(0, 0, .5, .5), (0, .5, .5, 1), (.5, 0, 1, .5), (.5, .5, 1, 1)]
    H, W = mask_bin.shape
    out = np.zeros((4, H, W), bool)
    for i, (y0, x0, y1, x1) in enumerate(coords):
        yy0, yy1, xx0, xx1 = int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W)
        out[i, yy0:yy1, xx0:xx1] = mask_bin[yy0:yy1, xx0:xx1] > 0
    return out


def build():
    df = pd.read_csv(B.CSV_PATH)
    has_ip = "image_path" in df.columns
    recs = []
    for n, (_, r) in enumerate(df.iterrows(), 1):
        uid = str(r[B.UID_COL]); yr = _year(r[B.PATIENT_COL])
        img = np.asarray(B._load_img(uid, r["image_path"] if has_ip else None),
                         dtype=np.float32) / 255.0
        msk = B._load_mask_np(uid) > 0
        rmasks = roi_pixel_masks(msk)
        rec = dict(uid=uid, year=yr, patient=r[B.PATIENT_COL],
                   RT=int(r.RT), LT=int(r.LT), RB=int(r.RB), LB=int(r.LB))
        for i, roi in enumerate(ROI):
            for k, val in roi_feats(img, rmasks[i]).items():
                rec[f"{roi}_{k}"] = val
        recs.append(rec)
        if n % 100 == 0:
            print(f"  {n} imgs", end="\r")
    out = pd.DataFrame(recs)
    RESULT_L.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULT_L / "l2_features.csv", index=False)
    print(f"\n[save] {RESULT_L / 'l2_features.csv'}  {out.shape}")
    return out


def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    allv = np.concatenate([pos, neg]); rk = pd.Series(allv).rank().values
    U = rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    a = U / (len(pos) * len(neg))
    return max(a, 1 - a)


def _auc_dir(pos, neg):
    """방향무관 AUC (tie-correct rankdata, NaN 제거). 순열 루프용."""
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    U = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    a = U / (len(pos) * len(neg))
    return max(a, 1 - a)


def perm_maxauc(sub, roi, feats, n_perm=1500, seed=0):
    """3→4 경계 16-특징 max AUC 의 귀무 max 순열 p값 (§4.3 표2).
    라벨(3/4)을 n_perm 회 섞어 매번 16-특징 max AUC 를 재계산 → 귀무 max 분포.
    최댓값 선택 편향(16개 중 최고 선택)을 이 분포로 보정한다.
    반환: (관측 max AUC, 구동 특징, perm p = P(귀무 max ≥ 관측 max))."""
    cols = [f"{roi}_{k}" for k in feats if f"{roi}_{k}" in sub.columns]
    P = sub.loc[sub[roi] == 4, cols].to_numpy(float)
    N = sub.loc[sub[roi] == 3, cols].to_numpy(float)
    if len(P) < 5 or len(N) < 5:
        return np.nan, "", np.nan
    obs = np.array([_auc_dir(P[:, j], N[:, j]) for j in range(len(cols))])
    if np.all(np.isnan(obs)):
        return np.nan, "", np.nan
    obs_max = float(np.nanmax(obs)); best = cols[int(np.nanargmax(obs))].split(f"{roi}_")[-1]
    X = np.vstack([P, N]); n4 = len(P); rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(X))
        pp, nn = X[idx[:n4]], X[idx[n4:]]
        nm = np.nanmax([_auc_dir(pp[:, j], nn[:, j]) for j in range(X.shape[1])])
        if nm >= obs_max:
            ge += 1
    return obs_max, best, (ge + 1) / (n_perm + 1)


def export_l2(fe, n_perm=1500):
    """3→4 경계 16-특징 max AUC + 귀무 max 순열 perm p per (year, roi) → l2_auc.csv.
    Bonferroni(4-ROI) 임계 α=0.0125."""
    rows = []
    for yr in (2024, 2026):
        sub = fe[fe.year == yr]
        for roi in ROI:
            a, feat, pp = perm_maxauc(sub, roi, FEATS, n_perm=n_perm)
            rows.append(dict(year=yr, roi=roi, boundary="3to4",
                             auc=round(a, 4) if a == a else "",
                             feature=feat, perm_p=round(pp, 4) if pp == pp else "",
                             bonferroni_0p0125=(int(pp < 0.0125) if pp == pp else ""),
                             ref=REF_L2[yr][roi]))
    with open(RESULT_L / "l2_auc.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["year", "roi", "boundary", "auc", "feature",
                                          "perm_p", "bonferroni_0p0125", "ref"])
        w.writeheader(); w.writerows(rows)
    print(f"[save] {RESULT_L / 'l2_auc.csv'}")
    for r in rows:
        bf = "통과" if r["bonferroni_0p0125"] == 1 else ("미통과" if r["bonferroni_0p0125"] == 0 else "-")
        print(f"  {r['year']} {r['roi']:<3} 3→4 AUC {r['auc']} ({r['feature']})  "
              f"perm p {r['perm_p']} [{bf}]  ref {r['ref']}")


def main():
    global NG
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="features 재계산(기본: 있으면 재사용)")
    ap.add_argument("--ng", type=int, default=NG, help="텍스처 그레이 레벨 수")
    ap.add_argument("--nperm", type=int, default=1500, help="귀무 max 순열 반복 수")
    args = ap.parse_args()
    NG = args.ng
    fpath = RESULT_L / "l2_features.csv"
    fe = pd.read_csv(fpath) if (fpath.exists() and not args.rebuild) else build()
    export_l2(fe, n_perm=args.nperm)


if __name__ == "__main__":
    main()
