# -*- coding: utf-8 -*-
r"""
E1 — 실데이터 cross arm (δ_obs 산출)   [draft §4.4 / 방향 반전 분해의 관측값]

cross(2024→2026, 2026→2024)를 양방향으로 학습·평가해 방향별 편향을 재고,
δ_obs=(rev−fwd)/2(라벨 성분)·γ=(rev+fwd)/2(모델 성분)를 만든다.
이 δ_obs 가 E2 의 Δg 로 A1 보정되어 최종 δ_corr(§5.3)이 된다.

전 실험 공통 규약: 50ep / early-stop 10 (core 고정). 커스텀 splitter 미등록 →
make_split 디스패처가 core 원래 교차 split 로 fallback.

실행:  python e1_cross.py --seeds 42 1 2
산출:  checkpoint/E1/runs/dorga/{2024to2026_s*, 2026to2024_s*}/test_preds.npz
"""
import argparse, os as _os, sys as _sys
from pathlib import Path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E"))
import e_common as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    args = ap.parse_args()

    root = A.A1_OUT / "E1" / "runs" / "dorga"   # checkpoint/E1/runs/dorga (summary.py·check_value 입력 규약)
    for mode in ["2024to2026", "2026to2024"]:   # 미등록 mode → core 원래 교차 split 로 fallback
        for s in args.seeds:
            A.B.train_arm("dorga", mode, s, A.B.EPOCHS, root)
    print("E1 완료:", root)


if __name__ == "__main__":
    main()
