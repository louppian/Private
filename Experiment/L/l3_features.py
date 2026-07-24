# -*- coding: utf-8 -*-
r"""
L3 프록시 독립성 — 강도 아닌 ROI 특징으로 층화 잔차 재산출

목적: L2·L3 가 공유하던 '평균 강도' 프록시를 깨기 위해, 같은 ROI(mask∩box)에서
      강도와 (준)직교인 특징을 계산해 L3 층화(S7 cell 7 공식)를 다시 돌린다.
특징:
  - mu   : 평균 강도 (검증용, roi_intensity_full.csv 와 대조)
  - grad : Sobel gradient 크기 평균 (경계 선명도, 강도와 최대 직교)
  - var  : 5x5 국소 분산 평균 (텍스처/불균질성)
  - p90  : 상위 백분위 강도 (초점성 음영; 강도와 부분 상관)
ROI 정의: split4(mask) = 폐 좌/우 성분 bbox 상·하 2등분 → [RT,RB,LT,LB] (S7 동일)
데이터: 2024=image_normalize/mask_normalize(디스크), 2026=seg+STN 정렬본(a1_common 캐시)

L3 지표(ROI별): explained% = 1 - |adj|/|marg|, adj=Σ_g Δ_g·n_g/Σn_g, marg=mean26-mean24
판정: RB explained% 가 강도(8%)처럼 낮게(잔차 지배) 유지되면, L3 신호가 강도 프록시
      아티팩트가 아니라 특징 독립적임을 뜻한다 → L3·L4 수렴이 서로 다른 정보원 위에 섬.

실행: python L3_features.py   (GPU 필요: 2026 seg+STN)
산출: D:\npjDM2026\runs\roi_features_full.csv , L3_feature_summary.txt
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "E"))
import e_common as A

OUT = A.A1_OUT
IMG24 = r"D:\MICCAI2026\inhauh\image_normalize"
MSK24 = r"D:\MICCAI2026\inhauh\mask_normalize"
R4 = ["RT", "RB", "LT", "LB"]          # split4 순서
CSV = pd.read_csv(A.MANIFEST)


def split4(mask_bin, min_area=1000):
    lab, n = ndi.label(mask_bin > 0)
    comps = [(i, int((lab == i).sum())) for i in range(1, n + 1)]
    comps = [c for c in comps if c[1] >= min_area]
    if len(comps) < 2:
        return None
    cen = {i: ndi.center_of_mass(lab == i)[1] for i, _ in comps}
    order = sorted(cen, key=cen.get)
    L, Rr = order[0], order[-1]
    H, W = mask_bin.shape; out = []
    for i in (L, Rr):
        ys, xs = np.where(lab == i)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        hv = np.linspace(y0, y1, 3)
        for k in range(2):
            out.append((int(hv[k]) / H, x0 / W, int(hv[k + 1]) / H, x1 / W))
    return out                          # [RT,RB,LT,LB] normalized (y0,x0,y1,x1)


def features(img, mask):
    """img float[0,1] HxW, mask bool HxW → ROI별 {mu,grad,var,p90}. 실패 시 None."""
    boxes = split4(mask)
    fb = False
    if boxes is None:                   # fallback: 고정 사분면
        boxes = [(0, 0, .5, .5), (.5, 0, 1, .5), (0, .5, .5, 1), (.5, .5, 1, 1)]; fb = True
    H, W = img.shape
    gx = ndi.sobel(img, axis=1); gy = ndi.sobel(img, axis=0)
    gmag = np.hypot(gx, gy)
    lm = ndi.uniform_filter(img, 5); lv = ndi.uniform_filter(img * img, 5) - lm * lm
    lv = np.clip(lv, 0, None)
    out = {"fallback": int(fb)}
    for name, (y0, x0, y1, x1) in zip(R4, boxes):
        ys, ye, xs, xe = int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W)
        m = mask[ys:ye, xs:xe]
        if m.sum() < 20:                # 너무 작으면 box 전체
            m = np.ones_like(m, bool)
        reg = img[ys:ye, xs:xe][m]
        out[f"mu_{name}"]   = float(reg.mean())
        out[f"grad_{name}"] = float(gmag[ys:ye, xs:xe][m].mean())
        out[f"var_{name}"]  = float(lv[ys:ye, xs:xe][m].mean())
        out[f"p90_{name}"]  = float(np.percentile(reg, 90))
    return out


def build():
    rows = []
    # 2026 캐시 (seg+STN)
    cache = A.build_full_2026_cache()
    for _, r in CSV.iterrows():
        y = int(r["year"])
        if y == 2024:
            stem = Path(r["image_path"]).name
            img = np.asarray(Image.open(Path(IMG24) / stem).convert("L"), np.float32) / 255.0
            mask = np.asarray(Image.open(Path(MSK24) / stem).convert("L")) > 0
        else:
            key = A.B.raw_path_from_npz(r["image_path"])
            if key not in cache:
                continue
            img_u8, msk_u8 = cache[key]
            img = img_u8.astype(np.float32) / 255.0
            mask = msk_u8 > 0
        f = features(img, mask)
        f.update(dict(year=y, patient=r["patient"],
                      RT=int(r["RT"]), LT=int(r["LT"]), RB=int(r["RB"]), LB=int(r["LB"])))
        rows.append(f)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "roi_features_full.csv", index=False, encoding="utf-8")
    return df


def l3_explain(df, feat):
    """S7 cell 7 공식으로 특징 feat 의 ROI별 explained% 반환."""
    out = {}
    for r in R4:
        col = f"{feat}_{r}"
        num, den = 0.0, 0
        for g in sorted(df[r].unique()):
            a = df.loc[(df.year == 2024) & (df[r] == g), col].dropna()
            b = df.loc[(df.year == 2026) & (df[r] == g), col].dropna()
            if len(a) < 8 or len(b) < 8:
                continue
            num += (b.mean() - a.mean()) * (len(a) + len(b)); den += len(a) + len(b)
        adj = num / den if den else float("nan")
        marg = df[df.year == 2026][col].mean() - df[df.year == 2024][col].mean()
        expl = 1 - abs(adj) / abs(marg) if marg else float("nan")
        out[r] = dict(crude=marg, adj=adj, explained=expl)
    return out


def main():
    df = build()
    # mu 검증: roi_intensity_full.csv 와 2024 상관
    ref = pd.read_csv(os.path.join(Path(A.MANIFEST).parent, "roi_intensity_full.csv"))
    L = ["검증: 재계산 mu vs roi_intensity_full.csv (2024, ROI별 상관)"]
    for r in R4:
        m = df[(df.year == 2024)].reset_index(drop=True)[f"mu_{r}"]
        rr = ref[ref.year == 2024].reset_index(drop=True)[f"mu_{r}"]
        n = min(len(m), len(rr))
        L.append(f"  mu_{r}: corr={np.corrcoef(m[:n], rr[:n])[0,1]:.3f}")
    # L3 재산출
    L.append("\nL3 explained% (등급이 연도차를 설명하는 비율; 낮을수록 잔차 지배=드리프트 신호)")
    L.append(f"{'특징':>6}" + "".join(f"{r:>9}" for r in R4))
    for feat in ["mu", "grad", "var", "p90"]:
        e = l3_explain(df, feat)
        L.append(f"{feat:>6}" + "".join(f"{e[r]['explained']*100:>8.0f}%" for r in R4))
    L.append("\n상세 (crude → adj):")
    for feat in ["mu", "grad", "var", "p90"]:
        e = l3_explain(df, feat)
        L.append(f"  [{feat}] " + "  ".join(f"{r} {e[r]['crude']:+.4f}→{e[r]['adj']:+.4f}({e[r]['explained']*100:.0f}%)" for r in R4))
    txt = "\n".join(L)
    (OUT / "L3_feature_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("\nsaved:", OUT / "roi_features_full.csv", "/", OUT / "L3_feature_summary.txt")


if __name__ == "__main__":
    main()
