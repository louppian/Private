# -*- coding: utf-8 -*-
r"""
E2 — 분포 정합 in-domain 대조   [A1_검증실험계획 §3 E2]

목적: E1 의 Δg 가 '진짜 방향 비대칭'인지 '등급분포 차이에서 온 수축'인지 분리.
      두 코호트를 공통 등급분포·공통 환자수로 서브샘플한 뒤 in-domain k-fold 재실행.
판정: 정합 후 Δg → 0  ⇒ 비대칭은 분포(수축) 기원 (L3 가 이미 방어).
      정합 후 Δg 잔존   ⇒ 진짜 방향 비대칭 → A1 실질 위반.

실행:  python e2_matched_indomain.py --folds 5 --match_seed 0 --init_seeds 42 1 2 --epochs 50
산출:  runs/E2/...  +  runs/E2/E2_summary.json  (E1_summary.json 있으면 Δg 비교 포함)
"""
import argparse, json, os as _os, sys as _sys
from pathlib import Path
import numpy as np
import pandas as pd
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E"))
import e_common as A


def matched_fold_splitter(year, test_pat, keep_pat, seed):
    """year in-domain, 단 keep_pat(정합 부분집합) 안에서만 train/val/test."""
    def _fn(df, _seed):
        keep = set(keep_pat)
        pool = np.array([p for p in A._patients_of(df, year)
                         if p in keep and p not in set(test_pat)], dtype=object)
        rng = np.random.default_rng(seed)
        rng.shuffle(pool)
        n_val = max(2, int(round(len(pool) * A.B.VAL_FRAC)))
        val_pat, train_pat = pool[:n_val], pool[n_val:]
        return A._mark(df, train_pat, val_pat, test_pat, year), year, year
    return _fn


def run_cohort_matched(year, keep_pat, folds, match_seed, init_seeds, epochs, cache, root):
    df = pd.read_csv(A.MANIFEST)
    # 정합 부분집합 환자만으로 k-fold
    pats = np.array([p for p in A._patients_of(df, year) if p in set(keep_pat)], dtype=object)
    rng = np.random.default_rng(match_seed); rng.shuffle(pats)
    fold_list = [(k, pats[k::folds]) for k in range(folds)]

    keys = ["overall"] + list(A.ROI)
    acc = {k: {} for k in keys}
    for k, test_pat in fold_list:
        for isd in init_seeds:
            mode = f"E2_{year}_f{k}_is{isd}"
            res = A.run_arm(mode, matched_fold_splitter(year, test_pat, keep_pat, match_seed),
                            isd, epochs, cache, root)
            d = np.load(res["npz"], allow_pickle=True)
            P, Y, pats_te = d["preds"], d["labels"], np.asarray(d["patients"])
            for key in keys:
                e = (P - Y).mean(axis=1) if key == "overall" \
                    else (P[:, A.ROI.index(key)] - Y[:, A.ROI.index(key)]).astype(float)
                for u in np.unique(pats_te):
                    acc[key].setdefault(u, []).append(float(e[pats_te == u].mean()))
    out = {"year": int(year), "n_pat": len(acc["overall"])}
    for key in keys:
        pat_ids = sorted(acc[key])
        vec = np.array([np.mean(acc[key][u]) for u in pat_ids])
        g, ci = A.mean_ci(vec, seed=match_seed)
        out[key] = dict(g=g, ci=list(ci), vec=vec.tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--match_seed", type=int, default=0)
    ap.add_argument("--init_seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    root = A.A1_OUT / "E2"
    df = pd.read_csv(A.MANIFEST)
    keep24, keep26 = A.match_two_cohorts(df, args.match_seed)
    cache = A.build_full_2026_cache()          # 2026 정합셋 학습에 필요

    keys = ["overall"] + list(A.ROI)
    out = {"matched_n": {"2024": len(keep24), "2026": len(keep26)}}
    out["2024"] = run_cohort_matched(2024, keep24, args.folds, args.match_seed, args.init_seeds, args.epochs, cache, root)
    out["2026"] = run_cohort_matched(2026, keep26, args.folds, args.match_seed, args.init_seeds, args.epochs, cache, root)

    a1 = {}
    for key in keys:
        a, b = np.array(out["2024"][key]["vec"]), np.array(out["2026"][key]["vec"])
        dg, dci = A.diff_ci(a, b, seed=0)
        a1[key] = dict(delta_g=dg, ci=list(dci), reject_A1=bool(A.sig(dci)))
    out["A1_test_matched"] = a1

    # E1 과 비교 (수축분이 정합으로 사라졌는가) — overall + ROI별
    e1p = A.A1_OUT / "E1" / "E1_summary.json"
    if e1p.exists():
        e1 = json.loads(e1p.read_text(encoding="utf-8"))
        if "A1_test" in e1:
            out["compare_vs_E1"] = {key: dict(
                delta_g_raw=e1["A1_test"][key]["delta_g"], delta_g_matched=a1[key]["delta_g"],
                shrinkage_removed=float(e1["A1_test"][key]["delta_g"] - a1[key]["delta_g"]))
                for key in keys if key in e1["A1_test"]}

    A.save_json(out, root / "E2_summary.json")
    print("\n" + "=" * 78)
    print(f"matched n: 2024={len(keep24)}명  2026={len(keep26)}명")
    print(f"{'ROI':<9}{'Δg(matched)':>16}{'A1':>12}")
    for key in keys:
        t = a1[key]
        print(f"{key:<9}{t['delta_g']:>+16.3f}{'위반 잔존' if t['reject_A1'] else '정합후 소멸':>12}")
    if "compare_vs_E1" in out:
        print("\nΔg raw → matched (수축분 제거):")
        for key, c in out["compare_vs_E1"].items():
            print(f"  {key:<9}{c['delta_g_raw']:>+8.3f} → {c['delta_g_matched']:>+8.3f}  ({c['shrinkage_removed']:+.3f})")
    print("saved:", root / "E2_summary.json")


if __name__ == "__main__":
    main()
