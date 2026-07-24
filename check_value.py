# -*- coding: utf-8 -*-
r"""
check_value.py — 산출 값이 draft md 기준값과 일치하는지 검증.

draft §4.4 (L3 방향 반전 분해, 표3·표4)를 기준값으로 두고,
checkpoint/E/runs/dorga/<mode>_s<seed>/test_preds.npz 에서 재계산해 대조한다.

  fwd = 정방향(2024→2026) 환자단위 bias 평균
  rev = 역방향(2026→2024) 환자단위 bias 평균
  δ(라벨) = (rev − fwd)/2 ,  γ(모델) = (rev + fwd)/2
  (CI 는 환자 단위 부트스트랩 — 여기선 점추정만 대조)

기준값 출처: draft §4.4 최종모델(50ep). 현 파이프라인은 50ep/early-stop 이라
값이 소폭 다를 수 있어 Δ 를 함께 출력한다(|Δ|≤TOL 이면 일치).

실행: python check_value.py            (E0 cross 가 먼저 돌아 npz 가 있어야 함)
      python check_value.py --tol 0.05
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):          # cp949 콘솔 크래시 방지
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parent
ROI = ["RT", "LT", "RB", "LB"]
RUNS = REPO / "checkpoint" / "E" / "runs" / "dorga"

# ── draft §4.4 기준값 ──
REF_BIAS = {"fwd": -0.231, "rev": +0.004}                      # 표3 전체 방향별 편향
REF = {                                                        # 표4 ROI δ(라벨)/γ(모델)
    "overall": (+0.118, -0.113),
    "RT": (+0.043, -0.096), "RB": (+0.198, -0.151),
    "LT": (+0.138, -0.081), "LB": (+0.091, -0.126),
}
REVERSAL = ["RB", "LT"]                                        # 부호반전(fwd<0<rev) 기대 ROI


def pooled_bias(mode, roi):
    """seed 전부 pool → 환자단위 bias 평균. roi=None 이면 4-ROI 평균. npz 없으면 None."""
    files = sorted(glob.glob(str(RUNS / f"{mode}_s*" / "test_preds.npz")))
    if not files:
        return None
    Ps, Ys, PT = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        Ps.append(d["preds"]); Ys.append(d["labels"])
        PT.append(np.asarray(d["patients"]).astype(str))
    P, Y, pt = np.vstack(Ps), np.vstack(Ys), np.concatenate(PT)
    if roi is None:
        e = (P - Y).astype(float).mean(1)
    else:
        j = ROI.index(roi); e = (P[:, j] - Y[:, j]).astype(float)
    u = np.unique(pt)
    return float(np.array([e[pt == p].mean() for p in u]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.03, help="일치 허용오차 |Δ|")
    args = ap.parse_args()
    TOL = args.tol

    PASS, FAIL = [], []
    def rec(ok, name):
        (PASS if ok else FAIL).append(name)

    print("=" * 76)
    print(f"check_value — draft §4.4 L3 기준값 대조 (TOL ±{TOL})")
    print(f"  runs: {RUNS}")
    print("=" * 76)

    fwd_all = pooled_bias("2024to2026", None)
    rev_all = pooled_bias("2026to2024", None)
    if fwd_all is None or rev_all is None:
        print("  [SKIP] test_preds.npz 없음 — 먼저 E0 실행:")
        print("         python Experiment/E/e0_cross.py --seeds 42 1 2")
        print("=" * 76)
        sys.exit(0)

    # [표3] 전체 방향별 편향
    print("\n[표3] 전체 방향별 편향")
    print(f"  {'':6}{'ref':>9}{'got':>9}{'Δ':>9}")
    for tag, got in (("fwd", fwd_all), ("rev", rev_all)):
        ref = REF_BIAS[tag]; dlt = got - ref
        ok = abs(dlt) <= TOL
        print(f"  {tag:6}{ref:>+9.3f}{got:>+9.3f}{dlt:>+9.3f}  {'OK' if ok else 'X'}")
        rec(ok, f"표3 {tag} bias≈{ref:+.3f}")

    # [표4] ROI δ/γ
    print("\n[표4] ROI δ(라벨)/γ(모델)")
    print(f"  {'ROI':7}{'δref':>8}{'δgot':>8}{'Δδ':>8}    {'γref':>8}{'γgot':>8}{'Δγ':>8}")
    for roi in ["overall"] + ROI:
        fwd = fwd_all if roi == "overall" else pooled_bias("2024to2026", roi)
        rev = rev_all if roi == "overall" else pooled_bias("2026to2024", roi)
        dg, gg = (rev - fwd) / 2, (rev + fwd) / 2
        dr, gr = REF[roi]
        dd, dgm = dg - dr, gg - gr
        ok_d, ok_g = abs(dd) <= TOL, abs(dgm) <= TOL
        print(f"  {roi:7}{dr:>+8.3f}{dg:>+8.3f}{dd:>+8.3f} {'OK' if ok_d else 'X ':>3}  "
              f"{gr:>+8.3f}{gg:>+8.3f}{dgm:>+8.3f} {'OK' if ok_g else 'X'}")
        rec(ok_d, f"표4 {roi} δ≈{dr:+.3f}")
        rec(ok_g, f"표4 {roi} γ≈{gr:+.3f}")

    # [부호반전] RB·LT
    print("\n[부호반전] fwd<0<rev 기대: RB, LT")
    for roi in REVERSAL:
        fwd = pooled_bias("2024to2026", roi); rev = pooled_bias("2026to2024", roi)
        ok = fwd < 0 < rev
        print(f"  {roi}: fwd {fwd:+.3f}  rev {rev:+.3f}  → {'반전 OK' if ok else '반전아님 X'}")
        rec(ok, f"{roi} 부호반전")

    print("\n" + "=" * 76)
    print(f"결과: PASS {len(PASS)} · FAIL {len(FAIL)}  (TOL ±{TOL})")
    if FAIL:
        print("불일치(값이 md 와 다름):")
        for n in FAIL:
            print(f"  - {n}")
    print("=" * 76)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
