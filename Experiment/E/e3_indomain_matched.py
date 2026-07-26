# -*- coding: utf-8 -*-
r"""
E3 — in-domain matched Δg 추정 (등급분포 정합)   [draft §5.2 본문 / 舊 E2]

목적: 두 코호트를 등급분포·환자수 맞춘 부분집합으로 제한해 Δg_matched 추정.
      raw(E2) 대비 수축분(Δg_raw−Δg_matched)이 분포차 기여. A1 위반 판정 = matched CI 가 0 배제.
설계: match_two_cohorts 로 정합 → 코호트별 환자 K-fold × init seed. 영역 [RT,LT,RB,LB].

실행:  python e3_indomain_matched.py --folds 5 --init_seeds 42 1 2
산출:  checkpoint/E3/dorga/ (가중치·npz) + Result/E3/dorga/ (per-run json) + Result/E3/e3_summary.json
       (A1_test_matched=Δg_matched)
"""
import argparse, os as _os, sys as _sys
import pandas as pd
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E"))
import e_common as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--init_seeds", type=int, nargs="+", default=[42, 1, 2])
    args = ap.parse_args()

    root = A.A1_OUT / "E3" / "dorga"              # checkpoint/E3/dorga
    keys = ["overall"] + list(A.ROI)
    df = A._prep(pd.read_csv(A.MANIFEST))         # match_two_cohorts 가 year·patient 사용

    keep24, keep26 = A.match_two_cohorts(df, 0)                                   # 정합 seed 고정 0
    m24 = A.run_cohort(2024, args.folds, 0, args.init_seeds, root, keep_pat=keep24, tag="matched")  # fold 분할 seed 0
    m26 = A.run_cohort(2026, args.folds, 0, args.init_seeds, root, keep_pat=keep26, tag="matched")
    out = {"matched": {"2024": m24, "2026": m26, "n": {"2024": len(keep24), "2026": len(keep26)}},
           "A1_test_matched": A.a1_test(m24, m26, keys)}
    A.save_json(out, A.RESULT_OUT / "E3" / "e3_summary.json")

    print("\n" + "=" * 66)
    print(f"E3 matched in-domain Δg_matched = g_2024 − g_2026 (정합 후)")
    print(f"{'ROI':<9}{'Δg_matched':>14}{'CI':>26}{'A1(matched)':>14}")
    for key in keys:
        t = out["A1_test_matched"][key]
        flag = "위반 잔존" if t["reject_A1"] else "정합후 소멸"
        print(f"{key:<9}{t['delta_g']:>+14.3f}   [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]{flag:>14}")
    print(f"(정합 n: 2024={out['matched']['n']['2024']}명 · 2026={out['matched']['n']['2026']}명)")
    print("\n[in-domain 성능 (cross 대조용)]")
    for yr in ("2024", "2026"):
        c = out["matched"][yr]
        print(f"  matched {yr}: ACC {c['acc']:.4f}  MAE {c['mae']:.4f}  (n_pat {c['n_pat']})")
    print("saved:", A.RESULT_OUT / "E3" / "e3_summary.json")


if __name__ == "__main__":
    main()
