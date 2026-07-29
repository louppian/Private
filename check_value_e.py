# -*- coding: utf-8 -*-
r"""
check_value_e.py — E 실험 산출값이 draft md(§5) 기준과 일치하는지 검증.

  E1 cross (§4.4)       : fwd −0.224 / rev +0.120         ← checkpoint/E1/dorga npz
  E2 정합 Δγ (§5.2)      : 전 ROI CI 0 포함(A1 미기각), RB Δγ≈−0.139  ← E2/e2_summary.json
  E3 복원곡선 (§5.1)     : slope≈0.940, intercept≈0.011    ← E3/e3_summary.json
  E4 누출 (§5.3)        : frac 1.0→+0.015, 0.5→+0.035, 0.25→−0.009  ← E4/e4_summary.json
  부록B raw Δg          : 기준표 없음, 정보출력만          ← AppendixB/appb_summary.json

실행: python check_value_e.py [--tol 0.05]
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

REPO = Path(__file__).resolve().parent
CKPT = REPO / "checkpoint"                      # npz(가중치 옆)
RESULT = REPO / "Result"                        # summary json (git 추적)
ROI = ["RT", "LT", "RB", "LB"]
RUNS = CKPT / "E1" / "dorga"                     # E1 npz

REF_BIAS = {"fwd": -0.224, "rev": +0.120}                       # §4.4 표3 (E1, 새 백본)
REF_E2_DG = {"RB": -0.139}                                      # §5.2 matched RB Δγ
REF_E3 = {"slope": 0.940, "intercept": 0.011}                  # §5.1 복원곡선 (E3 양성대조)
REF_E4 = {1.0: +0.015, 0.5: +0.035, 0.25: -0.009}              # §5.3 누출 (E4 음성대조)


def pooled_bias(mode):
    files = sorted(glob.glob(str(RUNS / f"{mode}_split*" / "test_preds.npz")))
    if not files:
        return None
    Ps, Ys, PT = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        Ps.append(d["preds"]); Ys.append(d["labels"]); PT.append(np.asarray(d["patients"]).astype(str))
    P, Y, pt = np.vstack(Ps), np.vstack(Ys), np.concatenate(PT)
    e = (P - Y).astype(float).mean(1); u = np.unique(pt)
    return float(np.array([e[pt == p].mean() for p in u]).mean())


def _load(name):
    p = RESULT.joinpath(*name.split("/"))       # summary json 은 Result 에
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def check_e1(rec, tol):
    print("\n" + "=" * 76); print(f"[E1] cross 방향편향 (§4.4)   runs: {RUNS}"); print("=" * 76)
    fwd, rev = pooled_bias("24to26"), pooled_bias("26to24")
    if fwd is None or rev is None:
        print("  [SKIP] npz 없음 — python Experiment/E/e1_cross.py"); return
    for tag, got in (("fwd", fwd), ("rev", rev)):
        ref = REF_BIAS[tag]; ok = abs(got - ref) <= tol
        print(f"  {tag}  ref{ref:+.3f} got{got:+.3f} Δ{got-ref:+.3f}  {'OK' if ok else 'X'}"); rec(ok, f"E1 {tag}≈{ref:+.3f}")


def check_e2(rec, tol):
    print("\n" + "=" * 76); print("[부록B] raw in-domain Δg_raw (정합 전)"); print("=" * 76)
    e2 = _load("AppendixB/appb_summary.json")
    if not e2:
        print("  [SKIP] Result/AppendixB/appb_summary.json 없음 — appendixB_indomain_raw.py 실행 후"); return
    r = e2.get("A1_test", {})
    if not r:
        print("  [SKIP] A1_test 키 없음"); return
    for key, v in r.items():                                  # 정보출력(기준표 없음, 부록 B)
        print(f"  {key:8} Δg_raw {v.get('delta_g'):+.3f}  reject_A1={v.get('reject_A1')}")


def check_e3(rec, tol):
    print("\n" + "=" * 76); print("[E2] 정합 in-domain Δγ_matched (§5.2)"); print("=" * 76)
    e3 = _load("E2/e2_summary.json")
    if not e3:
        print("  [SKIP] Result/E2/e2_summary.json 없음 — e2_indomain_matched.py 실행 후"); return
    m = e3.get("A1_test_matched", {})
    if not m:
        print("  [SKIP] A1_test_matched 키 없음"); return
    n_reject = 0
    for key, v in m.items():
        dg = v.get("delta_g"); rej = v.get("reject_A1")
        if rej: n_reject += 1
        line = f"  {key:8} Δγ {dg:+.3f}" + (f"  reject_A1={rej}" if rej is not None else "")
        if key in REF_E2_DG:
            ok = abs(dg - REF_E2_DG[key]) <= tol; line += f"  (ref{REF_E2_DG[key]:+.3f} {'OK' if ok else 'X'})"
            rec(ok, f"E3 {key} Δγ≈{REF_E2_DG[key]:+.3f}")
        print(line)
    rec(n_reject == 0, "E3 전 ROI A1 미기각(CI 0 포함)")
    print(f"  → A1 기각 ROI {n_reject}개 (0 기대)")


def check_e4(rec, tol):
    print("\n" + "=" * 76); print("[E3] 양성대조 복원곡선 (§5.1)"); print("=" * 76)
    e4 = _load("E3/e3_summary.json")
    if not e4:
        print("  [SKIP] Result/E3/e3_summary.json 없음 — e3_positive_control.py 실행 후"); return
    slope = e4.get("recovery_slope"); inter = e4.get("recovery_intercept", e4.get("intercept"))
    if slope is None:
        print("  [SKIP] recovery_slope 키 없음"); return
    ok = abs(slope - REF_E3["slope"]) <= tol
    print(f"  slope ref{REF_E3['slope']:.3f} got{slope:.3f}  {'OK' if ok else 'X'}"); rec(ok, f"E4 slope≈{REF_E3['slope']}")
    if inter is not None:
        oki = abs(inter - REF_E3["intercept"]) <= tol
        print(f"  intercept ref{REF_E3['intercept']:.3f} got{inter:.3f}  {'OK' if oki else 'X'}"); rec(oki, "E4 intercept≈0.04")


def check_e5(rec, tol):
    print("\n" + "=" * 76); print("[E4] 음성대조 누출 (§5.3)"); print("=" * 76)
    e5 = _load("E4/e4_summary.json")
    if not e5:
        print("  [SKIP] Result/E4/e4_summary.json 없음 — e4_negative_control.py 실행 후"); return
    got = {round(c["train_frac"], 2): c["delta_spurious"] for c in e5.get("curve", [])}
    for frac, ref in REF_E4.items():
        if frac not in got:
            rec(False, f"E5 frac{frac} 없음"); print(f"  frac {frac}: 없음  X"); continue
        ok = abs(got[frac] - ref) <= tol
        print(f"  frac {frac}: ref{ref:+.3f} got{got[frac]:+.3f}  {'OK' if ok else 'X'}"); rec(ok, f"E5 frac{frac} δ≈{ref:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()
    PASS, FAIL = [], []
    rec = lambda ok, n: (PASS if ok else FAIL).append(n)
    print("=" * 76); print(f"check_value_E — draft §5 실험값 대조 (TOL ±{a.tol})"); print("=" * 76)
    check_e1(rec, a.tol); check_e2(rec, a.tol); check_e3(rec, a.tol)
    check_e4(rec, a.tol); check_e5(rec, a.tol)
    print("\n" + "=" * 76); print(f"결과(E): PASS {len(PASS)} · FAIL {len(FAIL)}")
    for n in FAIL: print(f"  - {n}")
    print("=" * 76); sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
