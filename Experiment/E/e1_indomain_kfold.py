# -*- coding: utf-8 -*-
r"""
E1 — 검정력 있는 in-domain 추정 (실데이터)   [A1_검증실험계획 §3 E1]

목적: g_2024, g_2026 을 '전 환자를 test 로' 추정하고 H0: g_2024 = g_2026 검정.
      in-domain(c→c)은 라벨 오프셋 b_c 가 상쇄되므로 bias(c→c)=g_c (순수 모델오차).

설계: 코호트별 환자 단위 K-fold(모든 환자가 1회 test) × init seed 반복.
      각 fold: test=fold 환자, val=학습풀에서 VAL_FRAC, train=나머지.

실행:  python e1_indomain_kfold.py --folds 5 --fold_seed 0 --init_seeds 42 1 2 --epochs 50
산출:  runs/E1/<year>_f<k>_is<seed>_s<seed>/...  +  runs/E1/E1_summary.json
"""
import argparse, json, os as _os, sys as _sys
from pathlib import Path
import numpy as np
import pandas as pd
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E"))
import e_common as A


def indomain_fold_splitter(year, test_pat, seed):
    """year in-domain: test=test_pat, 나머지에서 VAL_FRAC 를 val 로."""
    def _fn(df, _seed):
        pool = np.array([p for p in A._patients_of(df, year) if p not in set(test_pat)], dtype=object)
        rng = np.random.default_rng(seed)
        rng.shuffle(pool)
        n_val = max(2, int(round(len(pool) * A.B.VAL_FRAC)))
        val_pat, train_pat = pool[:n_val], pool[n_val:]
        return A._mark(df, train_pat, val_pat, test_pat, year), year, year
    return _fn


def run_cohort(year, folds, fold_seed, init_seeds, epochs, cache, root):
    """한 코호트 K-fold × init_seeds → overall·ROI별 환자 bias 벡터(reps 평균)."""
    df = pd.read_csv(A.MANIFEST)
    fold_list = A.kfold_patient_folds(df, year, folds, fold_seed)
    keys = ["overall"] + list(A.ROI)
    acc = {k: {} for k in keys}                       # key -> {patient: [bias per rep]}
    for k, test_pat in fold_list:
        for isd in init_seeds:
            mode = f"E1_{year}_f{k}_is{isd}"
            res = A.run_arm(mode, indomain_fold_splitter(year, test_pat, fold_seed),
                            isd, epochs, cache, root)
            d = np.load(res["npz"], allow_pickle=True)
            P, Y, pats = d["preds"], d["labels"], np.asarray(d["patients"])
            for key in keys:
                e = (P - Y).mean(axis=1) if key == "overall" \
                    else (P[:, A.ROI.index(key)] - Y[:, A.ROI.index(key)]).astype(float)
                for u in np.unique(pats):
                    acc[key].setdefault(u, []).append(float(e[pats == u].mean()))
    out = {"year": int(year), "n_pat": len(acc["overall"])}
    for key in keys:
        pat_ids = sorted(acc[key])
        vec = np.array([np.mean(acc[key][u]) for u in pat_ids])
        g, ci = A.mean_ci(vec, seed=fold_seed)
        out[key] = dict(g=g, ci=list(ci), vec=vec.tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=0)
    ap.add_argument("--init_seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2026])
    args = ap.parse_args()

    root = A.A1_OUT / "E1"
    cache = A.build_full_2026_cache() if 2026 in args.years else A.EMPTY_CACHE
    keys = ["overall"] + list(A.ROI)

    out = {}
    for y in args.years:
        out[str(y)] = run_cohort(y, args.folds, args.fold_seed, args.init_seeds,
                                  A.B.EPOCHS, cache, root)

    # A1 검정: Δg = g_2024 - g_2026 (overall + ROI별)
    if "2024" in out and "2026" in out:
        a1 = {}
        for key in keys:
            a = np.array(out["2024"][key]["vec"]); b = np.array(out["2026"][key]["vec"])
            dg, dci = A.diff_ci(a, b, seed=0)
            a1[key] = dict(delta_g=dg, ci=list(dci), reject_A1=bool(A.sig(dci)))
        out["A1_test"] = a1

    A.save_json(out, root / "E1_summary.json")
    print("\n" + "=" * 78)
    print(f"{'ROI':<9}{'g_2024':>18}{'g_2026':>18}{'Δg (A1)':>14}{'A1':>10}")
    for key in keys:
        o24, o26 = out["2024"][key], out["2026"][key]
        t = out.get("A1_test", {}).get(key, {})
        flag = ("위반" if t.get("reject_A1") else "기각못함") if t else ""
        print(f"{key:<9}{o24['g']:>+9.3f}[{o24['ci'][0]:+.2f},{o24['ci'][1]:+.2f}]"
              f"{o26['g']:>+9.3f}[{o26['ci'][0]:+.2f},{o26['ci'][1]:+.2f}]"
              f"{t.get('delta_g', float('nan')):>+14.3f}{flag:>10}")
    print(f"(n: 2024={out['2024']['n_pat']}명 · 2026={out['2026']['n_pat']}명)")
    print("saved:", root / "E1_summary.json")


if __name__ == "__main__":
    main()
