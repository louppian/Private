# -*- coding: utf-8 -*-
r"""
temp.py — ROI δ_corr 대비 (RB 가 다른 사분면보다 큰가). 콘솔 출력만, 파일 안 만든다.

A1_verdict.json 에는 ROI별 주변 CI 만 있어서 대비를 못 낸다. ROI 간 공분산이 필요하다.
그래서 E1 의 환자 단위 예측(npz)과 E2 matched 의 환자 벡터에서 **환자 행을 통째로**
재표집해 4 ROI 를 같은 resample 에서 산출하고, 그 구름에서 대비를 뽑는다.

  δ_obs = (rev − fwd)/2 ,  δ_corr = δ_obs + Δγ/2
  대비 : RB − LT · RB − mean(RT,LT,LB) · RB − mean(RT,LB)

E1 npz 가 없으면 점추정만 찍고 끝낸다(부트스트랩은 서버에서만 가능).

실행: python temp.py
"""

import json
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

REPO = Path(__file__).resolve().parent
VERDICT = REPO / "Result" / "A1_verdict.json"
E2_SUMMARY = REPO / "Result" / "E2" / "e2_summary.json"
CKPT_E1 = REPO / "checkpoint" / "E1" / "dorga"

# npz 의 preds/labels 컬럼 순서. 해부학 순서로 고정이며 바꾸면 안 된다
# (Experiment/core.py · PLAN.md · check_value_l.py 모두 이 순서).
DATA_ROI = ["RT", "LT", "RB", "LB"]
# 출력·행렬 열 순서. DATA_ROI 와 다르므로 npz 인덱싱에 쓰면 안 된다.
ROI = ["RT", "RB", "LT", "LB"]

NBOOT = 5000
SEED = 20260801

CONTRASTS = {
    "RB − LT": ["LT"],
    "RB − mean(RT,LT,LB)": ["RT", "LT", "LB"],
    # plan_phase2A_v2 §4.3 의 C_ROI 와 같은 구성. LT 는 Phase 1 에서 후보 가능성이
    # 있어 음성대조에서 제외한다.
    "RB − mean(RT,LB)": ["RT", "LB"],
}


def pooled_patient_bias(mode, roi):
    """E1 npz → 환자별 평균 bias 벡터. 없으면 None."""
    files = sorted(CKPT_E1.glob(f"{mode}_split*/test_preds.npz"))
    if not files:
        return None
    P, Y, PT = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        P.append(d["preds"]); Y.append(d["labels"])
        PT.append(np.asarray(d["patients"]).astype(str))
    p, y = np.vstack(P), np.vstack(Y)
    pt = np.concatenate(PT)
    # npz 컬럼은 DATA_ROI 순서다. ROI(출력 순서)로 인덱싱하면 RB 와 LT 가 뒤바뀐다.
    j = DATA_ROI.index(roi)
    err = (p[:, j] - y[:, j]).astype(float)
    return np.array([err[pt == u].mean() for u in np.unique(pt)], dtype=float)


def boot_mean(mat, rng):
    """행(환자)을 통째로 재표집. 열(ROI) 간 공분산이 보존된다."""
    n = mat.shape[0]
    idx = rng.integers(0, n, size=(NBOOT, n))
    return mat[idx].mean(axis=1)


def main():
    W = 74
    delta = {r: float(json.loads(VERDICT.read_text(encoding="utf-8"))
                      ["delta_corrected"][r]["delta_corr"]) for r in ROI}

    print("=" * W)
    print("temp — ROI δ_corr 대비 (환자 단위 joint bootstrap)")
    print("=" * W)
    print("  δ_corr 점추정  " + " · ".join(f"{r} {delta[r]:+.3f}" for r in ROI))

    print("\n  [점추정 대비]")
    for name, others in CONTRASTS.items():
        v = delta["RB"] - sum(delta[o] for o in others) / len(others)
        print(f"    {name:<22}{v:+.4f}")

    fwd, rev = [], []
    for roi in ROI:
        f, r = pooled_patient_bias("24to26", roi), pooled_patient_bias("26to24", roi)
        if f is None or r is None:
            print(f"\n  [SKIP] E1 npz 없음 — {CKPT_E1}/{{24to26,26to24}}_split*/test_preds.npz")
            print("         주변 CI 만으로는 대비를 낼 수 없다. 서버에서 실행한다.")
            print("=" * W)
            return
        fwd.append(f); rev.append(r)

    rng = np.random.default_rng(SEED)
    fwd_mat, rev_mat = np.column_stack(fwd), np.column_stack(rev)
    # fwd 와 rev 는 test 환자 집합이 서로 다르므로 독립 재표집이 맞다.
    d_obs = (boot_mean(rev_mat, rng) - boot_mean(fwd_mat, rng)) / 2.0

    e2 = json.loads(E2_SUMMARY.read_text(encoding="utf-8"))["matched"]
    a = np.column_stack([np.asarray(e2["2024"][r]["vec"], float) for r in ROI])
    b = np.column_stack([np.asarray(e2["2026"][r]["vec"], float) for r in ROI])
    dg = boot_mean(a, rng) - boot_mean(b, rng)

    # 중심 보정: Experiment/E/summary.py 의 CI 관례와 맞춘다.
    raw = d_obs + dg / 2.0
    dc = raw - raw.mean(axis=0) + np.array([delta[r] for r in ROI])
    col = {r: i for i, r in enumerate(ROI)}

    print(f"\n  환자 수  E1 fwd test {fwd_mat.shape[0]} · rev test {rev_mat.shape[0]} · "
          f"E2 matched {a.shape[0]}/{b.shape[0]}   (n_boot {NBOOT})")

    print(f"\n  [ROI 상관행렬]  δ_corr bootstrap 구름")
    c = np.corrcoef(dc, rowvar=False)
    print("    " + " " * 6 + "".join(f"{r:>8}" for r in ROI))
    for i, r in enumerate(ROI):
        print(f"    {r:<6}" + "".join(f"{c[i, k]:>8.3f}" for k in range(len(ROI))))

    print(f"\n  [대비]  단측 p = bootstrap 구름이 0 이하인 비율")
    print(f"    {'대비':<22}{'추정':>9}{'95% CI':>22}{'단측 p':>10}")
    for name, others in CONTRASTS.items():
        v = dc[:, col["RB"]] - sum(dc[:, col[o]] for o in others) / float(len(others))
        lo, hi = np.percentile(v, [2.5, 97.5])
        p = float(np.mean(v <= 0.0))
        ci = f"[{lo:+.3f}, {hi:+.3f}]"
        print(f"    {name:<22}{np.mean(v):>+9.4f}{ci:>22}{p:>10.4f}")

    print("\n  세 대비는 서로 겹치는 탐색적 비교다. 다중비교 보정을 하지 않았으므로")
    print("  확증 결론이 아니라 RB 국소화의 보조 근거로만 쓴다.")
    print("=" * W)


if __name__ == "__main__":
    main()
