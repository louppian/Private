# -*- coding: utf-8 -*-
r"""
E0 — 실데이터 cross arm 재실행 (δ_obs 를 early-stop 규약으로 재산출)

기존 δ_obs(dorga_bias_direction_runs)는 50ep no-earlystop·final model 규약이다.
E1~E4 는 early-stop@50 규약이므로, 보정식 δ_corr=δ_obs+Δg/2 의 두 항을 같은
규약으로 맞추기 위해 cross arm(2024→2026, 2026→2024)을 동일 규약으로 다시 돌린다.

커스텀 splitter 를 등록하지 않으므로 make_split 디스패처가 원래 split(교차)로 fallback 한다.

실행:  python e0_cross.py --seeds 42 1 2 --epochs 50
산출:  runs/E0/{2024to2026_s*, 2026to2024_s*}/test_preds.npz
"""
import argparse, os as _os, sys as _sys
from pathlib import Path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E"))
import e_common as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    args = ap.parse_args()

    root = A.A1_OUT / "E" / "runs" / "dorga"    # checkpoint/E/runs/dorga (summary.py 입력 규약)
    for mode in ["2024to2026", "2026to2024"]:   # 미등록 mode → core 원래 교차 split 로 fallback
        for s in args.seeds:
            A.B.train_arm("dorga", mode, s, A.B.EPOCHS, root)
    print("E0 완료:", root)


if __name__ == "__main__":
    main()
