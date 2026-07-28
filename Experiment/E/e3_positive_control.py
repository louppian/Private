# -*- coding: utf-8 -*-
r"""
E3 — 반합성 양성 대조 (β 오프셋 주입, 복원곡선)   [draft §5.1 = Experiment 3]

목적: A1 이 '참인 조건'에서 방향 반전 분해가 주입한 라벨 오프셋 β 를 정확히 복원하는가.
설계: 한 코호트(기본 2024)를 등급분포 정합된 두 반쪽 H1,H2 로 나눔(→ 구성상 g_f=g_r, A1 성립).
      H2 라벨에 알려진 오프셋 β 주입(b_H2 - b_H1 = β).
        fwd = train H1 → test H2,   rev = train H2 → test H1.
      δ=(rev-fwd)/2 는 β 를, s=(rev+fwd)/2 는 g 를 복원해야 한다.
판정: 복원곡선(주입 β vs 추정 δ) 기울기 1·절편 0. 이탈분 = 추정기 편향.
      (clip(0,C-1) 때문에 큰 β 에서 기울기 감쇠 — 곡선이 그 지점을 드러낸다.)

실행:  python e4_positive_control.py --year 2024 --betas 0 0.25 0.5 1.0 --reps 42 1 2 --epochs 50
산출:  checkpoint/E3/ (가중치·npz) + Result/E3/ (per-run json) + Result/E3/e3_summary.json (복원곡선)
"""
import argparse, os as _os, sys as _sys
import numpy as np
import pandas as pd
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E"))
import e_common as A


def two_halves(df, year, seed):
    """year 환자를 평균등급 기준 정합된 두 반쪽으로 (교대 배정 → 분포 균등)."""
    sub = df[df.year == year]
    g = sub.groupby("patient")[A.ROI].apply(lambda x: x.to_numpy().mean()).sort_values()
    pats = list(g.index)
    rng = np.random.default_rng(seed)
    # 인접쌍을 랜덤하게 두 반쪽에 (분포 보존 + 무작위성)
    H1, H2 = set(), set()
    for i in range(0, len(pats), 2):
        pair = pats[i:i + 2]
        rng.shuffle(pair)
        H1.add(pair[0]); (H2.add(pair[1]) if len(pair) > 1 else None)
    return H1, H2


def half_cross_splitter(year, train_half, test_half, offset_half, beta, val_seed):
    """train_half → test_half in-cohort. offset_half(=B) 소속 행 전체에 β 주입."""
    def _fn(df, _seed):
        rng = np.random.default_rng(val_seed)
        tr_pool = np.array(list(train_half), dtype=object); rng.shuffle(tr_pool)
        n_val = max(2, int(round(len(tr_pool) * A.B.VAL_FRAC)))
        val_pat, train_pat = tr_pool[:n_val], tr_pool[n_val:]
        sub = A._mark(df, train_pat, val_pat, np.array(list(test_half), dtype=object), year)
        mask = sub.patient.isin(offset_half)          # B(=offset_half)=test_half(fwd) 또는 train_half(rev)
        sub = A.inject_offset(sub, mask.to_numpy(), beta, val_seed)
        return sub, year, year
    return _fn


def run_beta(year, H1, H2, beta, reps, epochs, root, skip_existing=True):
    """A=H1(오프셋0), B=H2(오프셋β). fwd: train H1→test H2, rev: train H2→test H1."""
    deltas, ss = [], []
    for isd in reps:
        fwd = A.run_arm(f"E3_{year}_beta{beta}_fwd",
                        half_cross_splitter(year, H1, H2, H2, beta, isd), isd, epochs, root,
                        skip_existing=skip_existing)
        rev = A.run_arm(f"E3_{year}_beta{beta}_rev",
                        half_cross_splitter(year, H2, H1, H2, beta, isd), isd, epochs, root,
                        skip_existing=skip_existing)
        dec = A.decompose(fwd["npz"], rev["npz"], seed=isd)
        deltas.append(dec["delta"]); ss.append(dec["s"])
    return dict(beta=float(beta), delta_mean=float(np.mean(deltas)), delta_sd=float(np.std(deltas)),
                s_mean=float(np.mean(ss)), s_sd=float(np.std(ss)), deltas=deltas, ss=ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--reps", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--half_seed", type=int, default=0)
    ap.add_argument("--overwrite", dest="skip_existing", action="store_false",
                    help="기본은 test_preds.npz 있으면 학습 생략(skip-existing 기본 ON). 이 옵션이면 강제 재학습")
    ap.set_defaults(skip_existing=True)
    args = ap.parse_args()

    root = A.A1_OUT / "E3"
    df = A._prep(pd.read_csv(A.MANIFEST))         # year·patient 파생(two_halves 가 df.year 사용)
    H1, H2 = two_halves(df, args.year, args.half_seed)

    curve = [run_beta(args.year, H1, H2, b, args.reps, A.B.EPOCHS, root, skip_existing=args.skip_existing)
             for b in args.betas]

    # 복원곡선 선형회귀 (β → δ): 기울기·절편
    bs = np.array([c["beta"] for c in curve]); ds = np.array([c["delta_mean"] for c in curve])
    slope, intercept = np.polyfit(bs, ds, 1) if len(bs) > 1 else (float("nan"), float("nan"))
    out = dict(year=int(args.year), n_H1=len(H1), n_H2=len(H2), curve=curve,
               recovery_slope=float(slope), recovery_intercept=float(intercept),
               s_grand=float(np.mean([c["s_mean"] for c in curve])),
               verdict="기울기≈1·절편≈0 이면 추정기 무편향 (A1-참 조건 검증됨)")

    A.save_json(out, A.RESULT_OUT / "E3" / "e3_summary.json")
    print("\n" + "=" * 70)
    print(f"{'β 주입':>8}{'δ 복원':>12}{'s(모델)':>12}")
    for c in curve:
        print(f"{c['beta']:>8.2f}{c['delta_mean']:>+12.4f}{c['s_mean']:>+12.4f}")
    print(f"\n복원곡선: δ ≈ {slope:.3f}·β + {intercept:+.3f}  (이상적 1·0)")
    print("saved:", A.RESULT_OUT / "E3" / "e3_summary.json")


if __name__ == "__main__":
    main()
