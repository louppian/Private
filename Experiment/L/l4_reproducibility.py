# -*- coding: utf-8 -*-
r"""
L4 — 판독자 재현성 (draft §4.5 표5) → Result/L/l4_reproducibility.csv

재측정 8케이스(2~3세션 반복 판독)에서 세션 pair 를 만들어 판독자 자기일치도를 낸다
(발표자료 S5_reproducibility 재현).
  · 라벨 = user_edit_{RT,LT,RB,LB}  — posthoc 은 모델 inference 와 95% 동일(재판독 아님), inference 는 모델.
  · pair = 케이스별 base(1세션) vs 각 후속 세션, min 길이로 truncate (3세션은 2쌍) → 총 119쌍.
  · 지표(ROI별+전체): 자기일치 ACC · quadratic weighted κ · MAE · drift Wilcoxon p.
  · κ 역설: ACC 최저 RB(68.9%)가 weighted κ 최고(0.899).

데이터: <root>/<case>/result{1,2,3}.txt  (컬럼 seq, inference_*, user_edit_*, posthoc_*)
  서버: /shared/home/mai/JeongGeon/Private/CXR/2026 CXR/4_재측정

⚠ 표5의 '모델-판독자 일치' 열은 여기서 산출하지 않는다 — 그것은 DORGA 모델 예측 vs 판독자
   라벨의 정확도(preds==labels)로, 재측정 자기일치도와는 별개 출처다(별도 계산 필요).

실행: python Experiment/L/l4_reproducibility.py [--root <재측정폴더>]
"""
import argparse
import csv
import glob
import os
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
QUAD = ["RT", "LT", "RB", "LB"]                      # user_edit 컬럼 순서
COLS = [f"user_edit_{r}" for r in QUAD]
OUT = REPO / "Result" / "L" / "l4_reproducibility.csv"
RR_DEFAULT = "/shared/home/mai/JeongGeon/Private/CXR/2026 CXR/4_재측정"
# md 표5 (self-ACC, weighted κ)
REF = {"RT": (0.798, 0.887), "LT": (0.882, 0.848), "RB": (0.689, 0.899),
       "LB": (0.824, 0.903), "전체": (0.798, 0.905)}


def build_pairs(root):
    """케이스별 result*.txt(정렬) → base(1세션) vs 각 후속 세션 pair. (PA, PB)[N,4] + 케이스 목록."""
    cases = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    PA, PB, per_case = [], [], []
    for case in cases:
        files = sorted(glob.glob(os.path.join(root, case, "result*.txt")))
        if len(files) < 2:
            continue
        labs = [pd.read_csv(f)[COLS].to_numpy() for f in files]   # 각 세션 [n,4]
        base = labs[0]
        for l in labs[1:]:
            n = min(len(base), len(l))                            # 공유 프레임(앞 n)
            PA.append(base[:n]); PB.append(l[:n])
        per_case.append((case, len(files)))
    return np.vstack(PA), np.vstack(PB), per_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=RR_DEFAULT, help="재측정 폴더 (<case>/result*.txt)")
    args = ap.parse_args()

    if not Path(args.root).exists():
        print(f"[SKIP] 재측정 폴더 없음: {args.root}")
        print("  구조: <root>/<case>/result{1,2,3}.txt (seq, inference_*, user_edit_*, posthoc_*)")
        return

    from scipy.stats import wilcoxon
    from sklearn.metrics import cohen_kappa_score

    PA, PB, per_case = build_pairs(args.root)
    print(f"[load] {len(per_case)}케이스 · {len(PA)}쌍  ({args.root})")

    rows = []
    for k, q in enumerate(QUAD + ["전체"]):
        a, b = (PA.ravel(), PB.ravel()) if q == "전체" else (PA[:, k], PB[:, k])
        acc = float((a == b).mean())
        kap = float(cohen_kappa_score(a, b, weights="quadratic"))
        mae = float(np.abs(a - b).mean())
        d = (b - a).astype(float); nz = d[d != 0]
        p = float(wilcoxon(nz).pvalue) if len(nz) else float("nan")
        ra, rk = REF.get(q, ("", ""))
        rows.append(dict(roi=q, self_acc=round(acc, 4), weighted_kappa=round(kap, 4),
                         mae=round(mae, 4), drift_wilcoxon_p=round(p, 4),
                         ref_acc=ra, ref_kappa=rk))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["roi", "self_acc", "weighted_kappa", "mae",
                                          "drift_wilcoxon_p", "ref_acc", "ref_kappa"])
        w.writeheader(); w.writerows(rows)
    print(f"[save] {OUT}")
    for r in rows:
        print(f"  {r['roi']:<4} ACC {r['self_acc']*100:>5.1f}% (ref {r['ref_acc']})  "
              f"wκ {r['weighted_kappa']:.3f} (ref {r['ref_kappa']})  "
              f"MAE {r['mae']:.3f}  drift p {r['drift_wilcoxon_p']}")


if __name__ == "__main__":
    main()
