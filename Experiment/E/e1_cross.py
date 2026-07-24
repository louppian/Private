# -*- coding: utf-8 -*-
r"""
E1 — 실데이터 cross arm (δ_obs 산출)   [draft §4.4]

cross(2024→2026, 2026→2024)를 양방향 학습·평가해 방향별 편향 → δ_obs=(rev−fwd)/2·γ.

설계: **val 5-fold(8:2) 비중복.** best_val 모델 선택의 val-운을 제거하려 train연도 환자를
      5-fold 로 나눠 fold k 를 val(20%), 나머지 train(80%). test 는 반대연도 전체(고정).
      → 방향당 split 1~5, 모든 train환자가 val 1회. δ_obs 는 split 평균.
      (test 회전은 없음 — cross test 는 반대연도 통째. E2 와 다른 점.)

전 실험 공통: 50ep / early-stop 10 / best_val reload (core 고정).

실행:  python e1_cross.py                       # fwd·rev × split 1~5 (10 arm)
       python e1_cross.py --mode fwd --split 1  # 특정 방향·split 만
산출:  checkpoint/E1/dorga/{24to26,26to24}_split{k}/{test_preds.npz, results.json}
"""
import argparse, os as _os, sys as _sys
import numpy as np
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E"))
import e_common as A

FOLDS = 5
DIRS = {"fwd": (2024, 2026, "24to26"), "rev": (2026, 2024, "26to24")}   # 이름: train연도→test연도


def cross_val_fold(train_year, test_year, k, folds=FOLDS):
    """train_year 를 folds 로 나눠 fold k=val(20%)·나머지=train(80%). test=test_year 전체."""
    def _fn(df, _seed):
        df = df.copy(); df["split"] = None
        tp = np.array(sorted(A._patients_of(df, train_year)), dtype=object)
        np.random.default_rng(0).shuffle(tp)             # 고정 분할(재현·비중복)
        val = set(tp[k::folds])                          # 인터리브 fold k
        m = df["year"] == train_year
        df.loc[m & df["patient"].isin(val), "split"] = "val"
        df.loc[m & ~df["patient"].isin(val), "split"] = "train"
        df.loc[df["year"] == test_year, "split"] = "test"
        return df[df.split.notna()].copy(), train_year, test_year
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="+", default=["fwd", "rev"], choices=["fwd", "rev"])
    ap.add_argument("--split", type=int, nargs="+", default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    root = A.A1_OUT / "E1" / "dorga"                     # checkpoint/E1/dorga
    for m in args.mode:
        ty, ey, tag = DIRS[m]
        for k in args.split:
            arm = f"{tag}_split{k}"                       # 예: 24to26_split1
            A.register(arm, cross_val_fold(ty, ey, k - 1))
            print(f"\n### E1 {m} {arm} (train {ty} 80% / val 20%(fold {k}) / test {ey} 전체)")
            A.B.run_one_dorga(arm, k, A.B.EPOCHS, root, arm=arm)
    print("\nE1 완료:", root)


if __name__ == "__main__":
    main()
