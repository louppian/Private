# -*- coding: utf-8 -*-
r"""
E4 — 반합성 음성 대조 (학습량 비대칭, 누출곡선)   [draft §5.3 = Experiment 4]

목적: A1 이 '위반된 조건'에서 방향 비대칭이 δ 로 얼마나 새는지 측정.
설계: 라벨 오프셋 β=0 (참 δ=0). 대신 한 방향의 학습만 약화시켜 g_f≠g_r 유도.
      기본 노브 = '약화 half 의 학습 데이터 비율 ρ'(split-only, 영상 수정 없음).
        fwd = train H1(ρ=1.0) → test H2,   rev = train H2(ρ) → test H1.
      참 δ=0 인데 δ 가 0 에서 벗어난 양 = 오염 이득 ∂δ/∂(g_r-g_f).
활용: E1 관측 δ·E2/E3 Δg 를 이 곡선에 대입 → 실데이터 δ 의 오염 추정.
      (선택: 영상 블러/노이즈 변형은 InhaUHMaskDataset 훅이 필요 — 본 스크립트는 데이터량 비대칭 사용.)

실행:  python e5_negative_control.py --year 2024 --fracs 1.0 0.5 0.25 --reps 42 1 2 --epochs 50
산출:  checkpoint/E4/ (가중치·npz) + Result/E4/ (per-run json) + Result/E4/e4_summary.json (누출곡선)
"""
import argparse, os as _os, sys as _sys
import numpy as np
import pandas as pd
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E"))
import e_common as A
from e3_positive_control import two_halves         # 동일 정합 half 분할 재사용


def degraded_splitter(year, train_half, test_half, train_frac, val_seed):
    """train_half 를 train_frac 비율로 축소해 학습(β=0). g 를 방향별로 다르게 만든다."""
    def _fn(df, _seed):
        rng = np.random.default_rng(val_seed)
        tr_pool = np.array(list(train_half), dtype=object); rng.shuffle(tr_pool)
        n_val = max(2, int(round(len(tr_pool) * A.B.VAL_FRAC)))
        val_pat = tr_pool[:n_val]
        rest = tr_pool[n_val:]
        n_keep = max(2, int(round(len(rest) * train_frac)))
        train_pat = rest[:n_keep]                        # ★ 학습 환자 축소 (약화)
        sub = A._mark(df, train_pat, val_pat, np.array(list(test_half), dtype=object), year)
        return sub, year, year                           # β=0: 라벨 주입 없음
    return _fn


def run_frac(year, H1, H2, frac, reps, epochs, root, skip_existing=True):
    """fwd: train H1(1.0)→test H2,  rev: train H2(frac)→test H1. 약화는 rev(=B=H2)에만."""
    deltas, ss = [], []
    for isd in reps:
        fwd = A.run_arm(f"E4_{year}_frac{frac}_fwd",
                        degraded_splitter(year, H1, H2, 1.0, isd), isd, epochs, root,
                        skip_existing=skip_existing)
        rev = A.run_arm(f"E4_{year}_frac{frac}_rev",
                        degraded_splitter(year, H2, H1, frac, isd), isd, epochs, root,
                        skip_existing=skip_existing)
        dec = A.decompose(fwd["npz"], rev["npz"], seed=isd)
        deltas.append(dec["delta"]); ss.append(dec["s"])
    return dict(train_frac=float(frac), delta_spurious=float(np.mean(deltas)),
                delta_sd=float(np.std(deltas)), s_mean=float(np.mean(ss)), deltas=deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    ap.add_argument("--reps", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--half_seed", type=int, default=0)
    ap.add_argument("--overwrite", dest="skip_existing", action="store_false",
                    help="기본은 test_preds.npz 있으면 학습 생략(skip-existing 기본 ON). 이 옵션이면 강제 재학습")
    ap.set_defaults(skip_existing=True)
    args = ap.parse_args()

    root = A.A1_OUT / "E4"
    df = A._prep(pd.read_csv(A.MANIFEST))         # year·patient 파생(two_halves 가 df.year 사용)
    H1, H2 = two_halves(df, args.year, args.half_seed)

    curve = [run_frac(args.year, H1, H2, f, args.reps, A.B.EPOCHS, root, skip_existing=args.skip_existing)
             for f in args.fracs]
    out = dict(year=int(args.year), n_H1=len(H1), n_H2=len(H2), curve=curve,
               note="참 δ=0. frac↓ 일수록 δ_spurious 가 0 에서 벗어나면 A1 위반이 δ 로 누출됨을 뜻함.")

    A.save_json(out, A.RESULT_OUT / "E4" / "e4_summary.json")
    print("\n" + "=" * 70)
    print(f"{'train_frac':>12}{'δ_spurious':>14}{'s(모델)':>12}   (참 δ=0)")
    for c in curve:
        print(f"{c['train_frac']:>12.2f}{c['delta_spurious']:>+14.4f}{c['s_mean']:>+12.4f}")
    print("saved:", A.RESULT_OUT / "E4" / "e4_summary.json")


if __name__ == "__main__":
    main()
