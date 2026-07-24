# -*- coding: utf-8 -*-
r"""
check_value_a.py — A1 최종 판정값이 draft md(§5.3)와 일치하는지 검증.

  δ_obs   (§4.4 표4)  : RB +0.198, LT +0.138           ← E0 npz 또는 verdict
  Δγ 정합 (§5.2)      : RB −0.102                        ← A1_verdict.json
  δ_corr  (§5.3 표)   : δ_obs + Δγ/2. RB +0.147[+0.029,+0.263](CI 0 배제),
                         LT +0.064(CI 0 포함). CI 0 배제 ROI = RB 뿐.
  최종 판정: A1 보정 후 유의 드리프트는 RB 상위경계 하나로 국소화.

산출물: checkpoint/A1_verdict.json (run_all.py). 실행: python check_value_a.py
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

REPO = Path(__file__).resolve().parent
CKPT = REPO / "checkpoint"
ROI = ["RT", "LT", "RB", "LB"]
RUNS = CKPT / "E" / "runs" / "dorga"

REF_DOBS = {"RB": +0.198, "LT": +0.138}                        # §4.4 표4
REF_DCORR = {"RB": +0.147, "LT": +0.064}                       # §5.3 표
REF_EXCLUDE0 = {"RB"}                                          # δ_corr CI 0 배제 ROI


def pooled_bias(mode, roi):
    files = sorted(glob.glob(str(RUNS / f"{mode}_s*" / "test_preds.npz")))
    if not files:
        return None
    Ps, Ys, PT = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        Ps.append(d["preds"]); Ys.append(d["labels"]); PT.append(np.asarray(d["patients"]).astype(str))
    P, Y, pt = np.vstack(Ps), np.vstack(Ys), np.concatenate(PT)
    j = ROI.index(roi); e = (P[:, j] - Y[:, j]).astype(float); u = np.unique(pt)
    return float(np.array([e[pt == p].mean() for p in u]).mean())


def check_dobs(rec, tol):
    print("\n" + "=" * 76); print("[δ_obs] 방향반전 관측 라벨성분 (§4.4 표4)"); print("=" * 76)
    fok = all(pooled_bias(m, "RB") is not None for m in ("2024to2026", "2026to2024"))
    if not fok:
        print("  [SKIP] E0 npz 없음 — python Experiment/E/e0_cross.py --seeds 42 1 2"); return
    for roi, ref in REF_DOBS.items():
        fwd, rev = pooled_bias("2024to2026", roi), pooled_bias("2026to2024", roi)
        dobs = (rev - fwd) / 2; ok = abs(dobs - ref) <= tol
        print(f"  {roi}  δ_obs ref{ref:+.3f} got{dobs:+.3f}  {'OK' if ok else 'X'}"); rec(ok, f"δ_obs {roi}≈{ref:+.3f}")


def check_verdict(rec, tol):
    print("\n" + "=" * 76); print("[δ_corr] A1 보정 후 최종 판정 (§5.3)"); print("=" * 76)
    p = CKPT / "A1_verdict.json"
    if not p.exists():
        print(f"  [SKIP] {p} 없음 — python Experiment/A/run_all.py (또는 --verdict_only)"); return
    V = json.loads(p.read_text(encoding="utf-8"))
    dc = V.get("delta_corrected", {})
    if not dc:
        print("  [SKIP] delta_corrected 키 없음"); return
    for roi, ref in REF_DCORR.items():
        row = dc.get(roi, {})
        val = row.get("delta_corr", row.get("delta", row.get("value")))
        ci = row.get("ci")
        if val is None:
            rec(False, f"δ_corr {roi} 값 없음"); print(f"  {roi}: 값 없음  X"); continue
        okv = abs(val - ref) <= tol
        excl = (ci[0] * ci[1] > 0) if (ci and len(ci) == 2) else None      # CI 0 배제?
        exp_excl = roi in REF_EXCLUDE0
        line = f"  {roi}  δ_corr ref{ref:+.3f} got{val:+.3f} {'OK' if okv else 'X'}"
        rec(okv, f"δ_corr {roi}≈{ref:+.3f}")
        if excl is not None:
            oke = (excl == exp_excl); line += f"  | CI0배제 got={excl} exp={exp_excl} {'OK' if oke else 'X'}"
            rec(oke, f"δ_corr {roi} CI0배제={exp_excl}")
        print(line)
    print(f"  → 기대: CI 0 배제 ROI = {sorted(REF_EXCLUDE0)} (RB 상위경계 국소화)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()
    PASS, FAIL = [], []
    rec = lambda ok, n: (PASS if ok else FAIL).append(n)
    print("=" * 76); print(f"check_value_A — draft §5.3 A1 판정값 대조 (TOL ±{a.tol})"); print("=" * 76)
    check_dobs(rec, a.tol); check_verdict(rec, a.tol)
    print("\n" + "=" * 76); print(f"결과(A): PASS {len(PASS)} · FAIL {len(FAIL)}")
    for n in FAIL: print(f"  - {n}")
    print("=" * 76); sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
