# -*- coding: utf-8 -*-
r"""
L4 — 판독자 재현성 (draft §4.5 표5) → Result/L/l4_reproducibility.csv

판독자 재측정(2세션) 라벨로 ROI별 자기일치 ACC·우연보정 weighted κ·체계적 드리프트
검정(Wilcoxon)을 낸다. κ 역설(ACC 최저 RB 가 κ 최고) 확인용. 학습·이미지 불필요.

입력: 재측정 CSV (--rr). ROI별 2세션 등급 컬럼 기대:
      RT_1,LT_1,RB_1,LB_1, RT_2,LT_2,RB_2,LB_2  (등급 0~4)
  ※ 이 데이터는 현재 repo/서버에 없음 — build_dataset 에서 재측정 라벨을 뽑으면 경로 지정.

실행: python Experiment/L/l4_reproducibility.py --rr <reread.csv>
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
OUT = REPO / "Result" / "L" / "l4_reproducibility.csv"
RR_DEFAULT = "/shared/home/mai/JeongGeon/Private/CXR/Merged/reread.csv"
REF = {"RT": (0.798, 0.887), "LT": (0.882, 0.848), "RB": (0.689, 0.899)}   # md 표5 (ACC, weighted κ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", default=RR_DEFAULT, help="판독자 재측정 CSV (RT_1,..,RT_2,.. 등급 0~4)")
    args = ap.parse_args()

    if not Path(args.rr).exists():
        print(f"[SKIP] 판독자 재측정 데이터 없음: {args.rr}")
        print("  필요 스키마: 컬럼 RT_1,LT_1,RB_1,LB_1,RT_2,LT_2,RB_2,LB_2 (등급 0~4, 두 세션)")
        print("  → build_dataset 에서 재측정 라벨 산출 후 --rr 로 지정")
        return

    from scipy.stats import wilcoxon
    from sklearn.metrics import cohen_kappa_score

    df = pd.read_csv(args.rr)
    rows = []
    for roi in ROI:
        s1 = df[f"{roi}_1"].to_numpy(); s2 = df[f"{roi}_2"].to_numpy()
        acc = float((s1 == s2).mean())
        kap = float(cohen_kappa_score(s1, s2, weights="quadratic"))
        try:
            _, p = wilcoxon(s1, s2); p = float(p)
        except Exception:
            p = float("nan")
        ra, rk = REF.get(roi, ("", ""))
        rows.append(dict(roi=roi, self_acc=round(acc, 4), weighted_kappa=round(kap, 4),
                         wilcoxon_p=round(p, 4), ref_acc=ra, ref_kappa=rk))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["roi", "self_acc", "weighted_kappa",
                                          "wilcoxon_p", "ref_acc", "ref_kappa"])
        w.writeheader(); w.writerows(rows)
    print(f"[save] {OUT}")
    for r in rows:
        print(f"  {r['roi']:<3} ACC {r['self_acc']} (ref {r['ref_acc']})  "
              f"wκ {r['weighted_kappa']} (ref {r['ref_kappa']})  Wilcoxon p {r['wilcoxon_p']}")


if __name__ == "__main__":
    main()
