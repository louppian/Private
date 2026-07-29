# -*- coding: utf-8 -*-
r"""
check_value_l.py — L 증거 사다리 값이 draft md(§4) 기준과 일치하는지 검증.

  L1 구조·분포 (§4.2 표1)   : labels.csv 만으로 즉시 (학습·이미지 불필요, 최속)
  L2 라벨-영상 정합 (§4.3 표2): ROI 3→4 경계 AUC   → l2_label_image 실행 후
  L3 방향 반전 분해 (§4.4 표3/4): E1 npz 의 δ/γ
  L4 재현성 (§4.5 표5)       : 자기일치·κ           → l4_reproducibility 구현 후

실행: python check_value_l.py [--csv labels.csv] [--tol 0.03]
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

REPO = Path(__file__).resolve().parent
ROI = ["RT", "LT", "RB", "LB"]
RUNS = REPO / "checkpoint" / "E1" / "dorga"    # E1 cross 산출
RESULT_L = REPO / "Result" / "L"
CSV_DEFAULT = "/shared/home/mai/JeongGeon/Private/CXR/Merged/labels.csv"

# ── draft §4.2 표1 (L1) ──
REF_L1 = {
    2024: {"n_pat": 68, "n_img": 718, "seq": 10.56, "mean": 1.640, "g0": 0.165, "g4": 0.093},
    2026: {"n_pat": 46, "n_img": 587, "seq": 12.76, "mean": 1.427, "g0": 0.236, "g4": 0.070},
}
# ── draft §4.3 표2 (L2) 3→4 경계 최고 AUC (6-특징 배터리) ──
REF_L2 = {2024: {"RT": 0.649, "RB": 0.604, "LT": 0.833, "LB": 0.791},
          2026: {"RT": 0.890, "RB": 0.818, "LT": 0.897, "LB": 0.826}}
# ── draft §4.4 표3/4 (L3) ──
REF_BIAS = {"fwd": -0.224, "rev": +0.120}                       # §4.4 표3 (새 백본)
REF_L3 = {"overall": (+0.172, -0.052), "RT": (+0.124, -0.026),
          "RB": (+0.240, -0.067), "LT": (+0.216, -0.021), "LB": (+0.108, -0.092)}   # §4.4 표4
REVERSAL = ["RB", "LT"]                                          # (실제로는 4 ROI 전부 반전)


def pooled_bias(mode, roi):
    files = sorted(glob.glob(str(RUNS / f"{mode}_split*" / "test_preds.npz")))
    if not files:
        return None
    Ps, Ys, PT = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        Ps.append(d["preds"]); Ys.append(d["labels"]); PT.append(np.asarray(d["patients"]).astype(str))
    P, Y, pt = np.vstack(Ps), np.vstack(Ys), np.concatenate(PT)
    e = (P - Y).astype(float).mean(1) if roi is None else (P[:, ROI.index(roi)] - Y[:, ROI.index(roi)]).astype(float)
    u = np.unique(pt)
    return float(np.array([e[pt == p].mean() for p in u]).mean())


def check_l1(csv, rec, tol):
    print("\n" + "=" * 76); print(f"[L1] 구조·분포 (§4.2 표1)   csv: {csv}"); print("=" * 76)
    if not Path(csv).exists():
        print("  [SKIP] labels.csv 없음"); return
    df = pd.read_csv(csv)
    df["year"] = df["patient_id"].astype(str).str[:2].map({"24": 2024, "26": 2026})
    print(f"  {'지표':<12}{'2024 ref/got':>22}{'2026 ref/got':>22}")
    for key, lab, cnt, tl in [("n_pat", "환자수", 1, 0), ("n_img", "영상수", 1, 0),
                              ("seq", "시퀀스길이", 0, 0.05), ("mean", "평균등급", 0, tol),
                              ("g0", "등급0비율", 0, tol), ("g4", "등급4비율", 0, tol)]:
        cells = []
        for yr in (2024, 2026):
            s = df[df.year == yr]; g = s[ROI].to_numpy(); np_ = s["patient_id"].nunique()
            got = {"n_pat": np_, "n_img": len(s), "seq": len(s) / max(np_, 1),
                   "mean": float(g.mean()), "g0": float((g == 0).mean()), "g4": float((g == 4).mean())}[key]
            ref = REF_L1[yr][key]; ok = (got == ref) if cnt else (abs(got - ref) <= tl)
            rec(ok, f"L1 {lab} {yr}")
            cells.append((f"{ref}/{got}" if cnt else f"{ref:.3f}/{got:.3f}") + ("  OK" if ok else "  X"))
        print(f"  {lab:<12}{cells[0]:>22}{cells[1]:>22}")


def check_l2(rec, tol):
    print("\n" + "=" * 76); print("[L2] 라벨-영상 정합 3→4 AUC (§4.3 표2)"); print("=" * 76)
    src = RESULT_L / "l2_auc.csv"
    if not src.exists():
        print(f"  [SKIP] {src} 없음 — l2_label_image.py 실행 후 대조"); return
    got = pd.read_csv(src)  # 기대 컬럼: year, roi, auc
    for yr in (2024, 2026):
        for roi in ROI:
            row = got[(got.year == yr) & (got.roi == roi)]
            if row.empty:
                rec(False, f"L2 {yr} {roi} 없음"); continue
            a = float(row.auc.iloc[0]); ref = REF_L2[yr][roi]
            rec(abs(a - ref) <= tol, f"L2 {yr} {roi} AUC≈{ref:.3f}")


def check_l3(rec, tol):
    print("\n" + "=" * 76); print(f"[L3] 방향 반전 분해 (§4.4)   runs: {RUNS}"); print("=" * 76)
    fwd_all, rev_all = pooled_bias("24to26", None), pooled_bias("26to24", None)
    if fwd_all is None or rev_all is None:
        print("  [SKIP] test_preds.npz 없음 — python Experiment/E/e1_cross.py"); return
    print("\n  [표3] 전체 방향편향")
    for tag, got in (("fwd", fwd_all), ("rev", rev_all)):
        ref = REF_BIAS[tag]; ok = abs(got - ref) <= tol
        print(f"    {tag}  ref{ref:+.3f}  got{got:+.3f}  Δ{got-ref:+.3f}  {'OK' if ok else 'X'}")
        rec(ok, f"L3 표3 {tag}≈{ref:+.3f}")
    print("\n  [표4] ROI δ(라벨)/γ(모델)")
    for roi in ["overall"] + ROI:
        fwd = fwd_all if roi == "overall" else pooled_bias("24to26", roi)
        rev = rev_all if roi == "overall" else pooled_bias("26to24", roi)
        dg, gg = (rev - fwd) / 2, (rev + fwd) / 2; dr, gr = REF_L3[roi]
        okd, okg = abs(dg - dr) <= tol, abs(gg - gr) <= tol
        print(f"    {roi:8} δ ref{dr:+.3f} got{dg:+.3f} {'OK' if okd else 'X'}   γ ref{gr:+.3f} got{gg:+.3f} {'OK' if okg else 'X'}")
        rec(okd, f"L3 {roi} δ≈{dr:+.3f}"); rec(okg, f"L3 {roi} γ≈{gr:+.3f}")
    print("\n  [부호반전] RB·LT")
    for roi in REVERSAL:
        fwd, rev = pooled_bias("24to26", roi), pooled_bias("26to24", roi)
        ok = fwd < 0 < rev
        print(f"    {roi}: fwd{fwd:+.3f} rev{rev:+.3f} → {'반전 OK' if ok else '반전아님 X'}"); rec(ok, f"L3 {roi} 부호반전")


REF_L4 = {"RT": (0.798, 0.887), "LT": (0.882, 0.848), "RB": (0.689, 0.899)}   # md 표5 (ACC, wκ)


def check_l4(rec, tol):
    print("\n" + "=" * 76); print("[L4] 재현성 κ (§4.5 표5)"); print("=" * 76)
    src = RESULT_L / "l4_reproducibility.csv"
    if not src.exists():
        print(f"  [SKIP] {src} 없음 — 재측정 데이터로 l4_reproducibility.py 실행 후"); return
    got = pd.read_csv(src)
    for roi, (ra, rk) in REF_L4.items():
        row = got[got.roi == roi]
        if row.empty:
            rec(False, f"L4 {roi} 없음"); continue
        a, k = float(row.self_acc.iloc[0]), float(row.weighted_kappa.iloc[0])
        oka, okk = abs(a - ra) <= tol, abs(k - rk) <= tol
        print(f"  {roi}  ACC {a:.3f}(ref{ra}) {'OK' if oka else 'X'}   wκ {k:.3f}(ref{rk}) {'OK' if okk else 'X'}")
        rec(oka, f"L4 {roi} ACC≈{ra}"); rec(okk, f"L4 {roi} wκ≈{rk}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--csv", default=CSV_DEFAULT)
    a = ap.parse_args()
    PASS, FAIL = [], []
    rec = lambda ok, n: (PASS if ok else FAIL).append(n)
    print("=" * 76); print(f"check_value_L — draft §4 L 사다리 대조 (TOL ±{a.tol})"); print("=" * 76)
    # L2 는 6-특징 배터리 + 귀무 max 순열보정 구현(l2_label_image.py) → 표2 대조 재활성화.
    check_l1(a.csv, rec, a.tol); check_l2(rec, a.tol); check_l3(rec, a.tol); check_l4(rec, a.tol)
    print("\n" + "=" * 76); print(f"결과(L): PASS {len(PASS)} · FAIL {len(FAIL)}")
    for n in FAIL: print(f"  - {n}")
    print("=" * 76); sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
