# -*- coding: utf-8 -*-
r"""
E2 — in-domain Δg 추정 + 분포 정합 (raw·matched 통합)   [draft §5.2 / 舊 E1+E2]

목적: g_2024·g_2026 을 in-domain(c→c)으로 추정해 A1 검정(Δg).
      in-domain 은 라벨 오프셋 b_c 가 상쇄되어 bias(c→c)=g_c(순수 모델오차).
  raw     : 전 환자 K-fold → Δg_raw            (舊 E1, draft 부록 B)
  matched : 두 코호트 등급분포 정합 서브샘플 → Δg_matched  (舊 E2, draft §5.2 본문)
  비교    : Δg_raw → Δg_matched (수축분 제거량). A1 위반 판정 = matched CI 가 0 배제.

설계: 코호트별 환자 단위 K-fold(모든 환자 1회 test) × init seed 반복. 영역 [RT,LT,RB,LB].

실행:  python e2_indomain.py --folds 5 --seed 0 --init_seeds 42 1 2   # raw+matched
       python e2_indomain.py --raw_only                               # raw 만
산출:  checkpoint/E2/... + checkpoint/E2/E2_summary.json
       (A1_test=Δg_raw, A1_test_matched=Δg_matched, compare=수축분)
"""
import argparse, os as _os, sys as _sys
from pathlib import Path
import numpy as np
import pandas as pd
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E"))
import e_common as A


def fold_splitter(year, test_pat, seed, keep_pat=None):
    """year in-domain splitter. keep_pat 주면 그 정합 부분집합 안에서만 train/val/test."""
    def _fn(df, _seed):
        keep = set(keep_pat) if keep_pat is not None else None
        te = set(test_pat)
        pool = np.array([p for p in A._patients_of(df, year)
                         if p not in te and (keep is None or p in keep)], dtype=object)
        rng = np.random.default_rng(seed); rng.shuffle(pool)
        n_val = max(2, int(round(len(pool) * 0.20)))     # 8:2 (val 20%, E1 과 동일)
        return A._mark(df, pool[n_val:], pool[:n_val], test_pat, year), year, year
    return _fn


def run_cohort(year, folds, seed, init_seeds, root, keep_pat=None, tag="E2"):
    """한 코호트 K-fold × init_seeds → overall·ROI별 환자 bias 벡터(reps 평균)."""
    df = A._prep(pd.read_csv(A.MANIFEST))         # year·patient 파생(labels.csv → 구 splitter 호환)
    if keep_pat is None:
        fold_list = A.kfold_patient_folds(df, year, folds, seed)
    else:
        pats = np.array([p for p in A._patients_of(df, year) if p in set(keep_pat)], dtype=object)
        rng = np.random.default_rng(seed); rng.shuffle(pats)
        fold_list = [(k, pats[k::folds]) for k in range(folds)]
    keys = ["overall"] + list(A.ROI)
    bias_pat = {k: {} for k in keys}
    Pall, Yall = [], []                                # ACC/MAE 용 (전 arm pool)
    for k, test_pat in fold_list:
        for isd in init_seeds:
            mode = f"{tag}_{year}_fold{k}"             # 읽기 쉬운 이름. seed 는 core 가 _s{isd} 로 붙임
            res = A.run_arm(mode, fold_splitter(year, test_pat, seed, keep_pat), isd, A.B.EPOCHS, None, root)
            d = np.load(res["npz"], allow_pickle=True)
            P, Y, pats_te = d["preds"], d["labels"], np.asarray(d["patients"])
            Pall.append(P); Yall.append(Y)
            for key in keys:
                e = (P - Y).mean(axis=1) if key == "overall" \
                    else (P[:, A.ROI.index(key)] - Y[:, A.ROI.index(key)]).astype(float)
                for u in np.unique(pats_te):
                    bias_pat[key].setdefault(u, []).append(float(e[pats_te == u].mean()))
    P, Y = np.vstack(Pall), np.vstack(Yall)
    out = {"year": int(year), "n_pat": len(bias_pat["overall"]),
           "acc": float((P == Y).mean()), "mae": float(np.abs(P - Y).mean())}
    for key in keys:
        vec = np.array([np.mean(bias_pat[key][u]) for u in sorted(bias_pat[key])])
        g, ci = A.mean_ci(vec, seed=seed)
        out[key] = dict(g=g, ci=list(ci), vec=vec.tolist())
    return out


def a1_test(out24, out26, keys):
    a1 = {}
    for key in keys:
        a, b = np.array(out24[key]["vec"]), np.array(out26[key]["vec"])
        dg, dci = A.diff_ci(a, b, seed=0)
        a1[key] = dict(delta_g=dg, ci=list(dci), reject_A1=bool(A.sig(dci)))
    return a1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="fold/정합 공통 seed")
    ap.add_argument("--init_seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--raw_only", action="store_true", help="정합 없이 raw in-domain 만")
    args = ap.parse_args()

    root = A.A1_OUT / "E2" / "dorga"              # checkpoint/E2/dorga (E1 과 동일 규약)
    keys = ["overall"] + list(A.ROI)
    df = A._prep(pd.read_csv(A.MANIFEST))         # match_two_cohorts 가 year·patient 사용

    # raw in-domain (전 환자) → Δg_raw
    raw24 = run_cohort(2024, args.folds, args.seed, args.init_seeds, root, tag="raw")
    raw26 = run_cohort(2026, args.folds, args.seed, args.init_seeds, root, tag="raw")
    out = {"raw": {"2024": raw24, "2026": raw26}, "A1_test": a1_test(raw24, raw26, keys)}

    # 분포 정합 → Δg_matched + 비교
    if not args.raw_only:
        keep24, keep26 = A.match_two_cohorts(df, args.seed)
        m24 = run_cohort(2024, args.folds, args.seed, args.init_seeds, root, keep_pat=keep24, tag="matched")
        m26 = run_cohort(2026, args.folds, args.seed, args.init_seeds, root, keep_pat=keep26, tag="matched")
        out["matched"] = {"2024": m24, "2026": m26, "n": {"2024": len(keep24), "2026": len(keep26)}}
        out["A1_test_matched"] = a1_test(m24, m26, keys)
        out["compare"] = {key: dict(
            delta_g_raw=out["A1_test"][key]["delta_g"],
            delta_g_matched=out["A1_test_matched"][key]["delta_g"],
            shrinkage_removed=float(out["A1_test"][key]["delta_g"] - out["A1_test_matched"][key]["delta_g"]))
            for key in keys}

    A.save_json(out, root / "E2_summary.json")

    print("\n" + "=" * 78)
    print(f"{'ROI':<9}{'Δg_raw':>12}{'Δg_matched':>14}{'A1(matched)':>14}")
    for key in keys:
        r = out["A1_test"][key]["delta_g"]
        if "A1_test_matched" in out:
            t = out["A1_test_matched"][key]
            flag = "위반 잔존" if t["reject_A1"] else "정합후 소멸"
            print(f"{key:<9}{r:>+12.3f}{t['delta_g']:>+14.3f}{flag:>14}")
        else:
            print(f"{key:<9}{r:>+12.3f}{'—':>14}{'(raw only)':>14}")
    if "matched" in out:
        print(f"(정합 n: 2024={out['matched']['n']['2024']}명 · 2026={out['matched']['n']['2026']}명)")

    print("\n[in-domain 성능 (cross 대조용)]")
    for variant in [x for x in ("raw", "matched") if x in out]:
        for yr in ("2024", "2026"):
            c = out[variant][yr]
            print(f"  {variant:<8} {yr}: ACC {c['acc']:.4f}  MAE {c['mae']:.4f}  (n_pat {c['n_pat']})")
    print("saved:", root / "E2_summary.json")


if __name__ == "__main__":
    main()
