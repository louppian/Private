# -*- coding: utf-8 -*-
r"""
L4 — 판독자 재현성 (draft §4.5 표5) → Result/L/l4_reproducibility.csv

표5는 두 가지 서로 다른 집합에서 나온다 (발표자료 S5 재현):

1) 자기일치 ACC · weighted κ · drift Wilcoxon  ← **재측정 8케이스**(4_재측정)
   · 라벨 = user_edit_{RT,LT,RB,LB} (posthoc 은 모델 inference 와 95% 동일 = 재판독 아님)
   · pair = 케이스별 base(1세션) vs 각 후속 세션, min 길이 truncate (3세션은 2쌍) → 119쌍
   · κ 역설: ACC 최저 RB(68.9%)가 weighted κ 최고(0.899)

2) 모델-판독자 일치  ← **분석셋**(1_독립 + 2_완전동일 = 46명·587장)
   · inference_{RT..LB}(DORGA 모델) vs user_edit_{RT..LB}(판독자) 일치율

데이터: <cxr_root>/{4_재측정,1_독립,2_완전동일}/<case>/result{1,2,3}.txt
  컬럼 seq, inference_*, user_edit_*, posthoc_* (모두 해부학 순서 [RT,LT,RB,LB])
  서버: /shared/home/mai/JeongGeon/Private/CXR/2026 CXR

실행: python Experiment/L/l4_reproducibility.py [--cxr_root <2026 CXR 폴더>]
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
QUAD = ["RT", "LT", "RB", "LB"]
EDT = [f"user_edit_{r}" for r in QUAD]
INF = [f"inference_{r}" for r in QUAD]
OUT = REPO / "Result" / "L" / "l4_reproducibility.csv"
CXR_DEFAULT = "/shared/home/mai/JeongGeon/Private/CXR/2026 CXR"
REMEASURE = "4_재측정"                         # 재측정(자기일치)
ANALYSIS = ["1_독립", "2_완전동일"]              # 분석셋(모델-판독자, 46명·587장)
# md 표5 (self-ACC, weighted κ, 모델-판독자)
REF = {"RT": (0.798, 0.887, 0.775), "LT": (0.882, 0.848, 0.579),
       "RB": (0.689, 0.899, 0.506), "LB": (0.824, 0.903, 0.676),
       "전체": (0.798, 0.905, 0.634)}


def build_pairs(remeasure_dir):
    """재측정: 케이스별 result*.txt(정렬) → base vs 각 후속 세션 pair. (PA, PB)[N,4] + 케이스수."""
    cases = sorted(d for d in os.listdir(remeasure_dir)
                   if os.path.isdir(os.path.join(remeasure_dir, d)))
    PA, PB, n_case = [], [], 0
    for case in cases:
        files = sorted(glob.glob(os.path.join(remeasure_dir, case, "result*.txt")))
        if len(files) < 2:
            continue
        labs = [pd.read_csv(f)[EDT].to_numpy() for f in files]
        base = labs[0]
        for l in labs[1:]:
            n = min(len(base), len(l))
            PA.append(base[:n]); PB.append(l[:n])
        n_case += 1
    return np.vstack(PA), np.vstack(PB), n_case


def model_reader(cxr_root):
    """분석셋(1_독립+2_완전동일) result*.txt: inference(모델) vs user_edit(판독자) 일치율(ROI+전체)."""
    I, E = [], []
    for cat in ANALYSIS:
        for f in sorted(glob.glob(os.path.join(cxr_root, cat, "*", "result*.txt"))):
            d = pd.read_csv(f); I.append(d[INF].to_numpy()); E.append(d[EDT].to_numpy())
    if not I:
        return {}
    I, E = np.vstack(I), np.vstack(E)
    out = {q: float((I[:, k] == E[:, k]).mean()) for k, q in enumerate(QUAD)}
    out["전체"] = float((I == E).mean())
    out["_n"] = len(I)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxr_root", default=CXR_DEFAULT, help="2026 CXR 폴더 (하위 4_재측정·1_독립·2_완전동일)")
    args = ap.parse_args()

    rm_dir = os.path.join(args.cxr_root, REMEASURE)
    if not os.path.isdir(rm_dir):
        print(f"[SKIP] 재측정 폴더 없음: {rm_dir}")
        return

    from scipy.stats import wilcoxon
    from sklearn.metrics import cohen_kappa_score

    PA, PB, n_case = build_pairs(rm_dir)
    mr = model_reader(args.cxr_root)
    print(f"[load] 재측정 {n_case}케이스·{len(PA)}쌍  |  모델-판독자 분석셋 {mr.get('_n', 0)}장")

    rows = []
    for k, q in enumerate(QUAD + ["전체"]):
        a, b = (PA.ravel(), PB.ravel()) if q == "전체" else (PA[:, k], PB[:, k])
        acc = float((a == b).mean())
        kap = float(cohen_kappa_score(a, b, weights="quadratic"))
        mae = float(np.abs(a - b).mean())
        d = (b - a).astype(float); nz = d[d != 0]
        p = float(wilcoxon(nz).pvalue) if len(nz) else float("nan")
        ra, rk, rm_ref = REF.get(q, ("", "", ""))
        rows.append(dict(roi=q, self_acc=round(acc, 4), weighted_kappa=round(kap, 4),
                         model_reader=round(mr.get(q, float("nan")), 4), mae=round(mae, 4),
                         drift_wilcoxon_p=round(p, 4),
                         ref_acc=ra, ref_kappa=rk, ref_model_reader=rm_ref))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["roi", "self_acc", "weighted_kappa", "model_reader",
                                          "mae", "drift_wilcoxon_p",
                                          "ref_acc", "ref_kappa", "ref_model_reader"])
        w.writeheader(); w.writerows(rows)
    print(f"[save] {OUT}")
    for r in rows:
        print(f"  {r['roi']:<4} ACC {r['self_acc']*100:>5.1f}%(ref {r['ref_acc']})  "
              f"wκ {r['weighted_kappa']:.3f}(ref {r['ref_kappa']})  "
              f"모델-판독자 {r['model_reader']*100:>5.1f}%(ref {r['ref_model_reader']})  drift p {r['drift_wilcoxon_p']}")


if __name__ == "__main__":
    main()
