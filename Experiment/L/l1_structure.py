# -*- coding: utf-8 -*-
r"""
L0–L1 — 구조·분포 (draft §4.2 표1) → Result/L/l1_structure.csv

labels.csv 만으로 계산(학습·이미지·GPU 불필요, 최속). 연도 = patient_id 접두어 24_/26_.

지표: 환자수·영상수·환자당 시퀀스길이·전체 평균등급·등급0~4 비율.
(마르코프 등급전이 방향비는 시퀀스 순서 컬럼이 필요 — labels.csv 에 순서/시간이 없어 생략.
 build_dataset 에서 순서 컬럼을 넣으면 여기에 추가한다.)

실행: python Experiment/L/l1_structure.py [--csv labels.csv]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[2]
ROI = ["RT", "LT", "RB", "LB"]
CSV_DEFAULT = "/shared/home/mai/JeongGeon/Private/CXR/Merged/labels.csv"
OUT = REPO / "Result" / "L" / "l1_structure.csv"

REF = {"n_pat": (68, 46), "n_img": (718, 587), "seq": (10.56, 12.76),
       "mean": (1.640, 1.427), "g0": (0.165, 0.236), "g4": (0.093, 0.070),
       "mk_worsen": (0.19, 0.19), "mk_same": (0.64, 0.64), "mk_improve": (0.17, 0.17)}   # md 표1


def markov_ratios(s):
    """환자별 seq 순 연속 프레임의 등급 전이 방향비 (악화↑/유지/호전↓), 전 ROI pool.
    등급↑ = 악화(중증도 증가). seq 컬럼(프레임 순서)이 있어야 계산."""
    if "seq" not in s.columns:
        return (float("nan"),) * 3
    w = sa = b = 0
    for _, gp in s.groupby("patient_id"):
        arr = gp.sort_values("seq")[ROI].to_numpy()          # [n_frame, 4]
        if len(arr) < 2:
            continue
        d = np.diff(arr, axis=0)                             # 연속 프레임 등급차
        w += int((d > 0).sum()); sa += int((d == 0).sum()); b += int((d < 0).sum())
    tot = w + sa + b
    return (w / tot, sa / tot, b / tot) if tot else (float("nan"),) * 3


def cohort_stats(df, yr):
    s = df[df.year == yr]
    g = s[ROI].to_numpy()
    npat = int(s["patient_id"].nunique())
    mkw, mks, mkb = markov_ratios(s)
    return {"n_pat": npat, "n_img": int(len(s)), "seq": len(s) / max(npat, 1),
            "mean": float(g.mean()),
            **{f"g{k}": float((g == k).mean()) for k in range(5)},
            "mk_worsen": mkw, "mk_same": mks, "mk_improve": mkb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT)
    args = ap.parse_args()
    if not Path(args.csv).exists():
        sys.exit(f"[중단] labels.csv 없음: {args.csv}")

    df = pd.read_csv(args.csv)
    df["year"] = df["patient_id"].astype(str).str[:2].map({"24": 2024, "26": 2026})
    v24, v26 = cohort_stats(df, 2024), cohort_stats(df, 2026)

    keys = ["n_pat", "n_img", "seq", "mean", "g0", "g1", "g2", "g3", "g4",
            "mk_worsen", "mk_same", "mk_improve"]
    rows = []
    for k in keys:
        r24, r26 = REF.get(k, ("", ""))
        rows.append(dict(metric=k,
                         y2024=round(v24[k], 4), y2026=round(v26[k], 4),
                         ref2024=r24, ref2026=r26))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "y2024", "y2026", "ref2024", "ref2026"])
        w.writeheader(); w.writerows(rows)
    print(f"[save] {OUT}")
    for r in rows:
        print(f"  {r['metric']:<7} 2024 {r['y2024']:>8} (ref {r['ref2024']})   "
              f"2026 {r['y2026']:>8} (ref {r['ref2026']})")


if __name__ == "__main__":
    main()
