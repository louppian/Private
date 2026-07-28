# -*- coding: utf-8 -*-
r"""
E2 — in-domain matched Δg 추정 (등급분포 정합)   [draft §5.2 = Experiment 2]

목적: 두 코호트를 등급분포·환자수 맞춘 부분집합으로 제한해 Δg_matched 추정.
      raw(E2) 대비 수축분(Δg_raw−Δg_matched)이 분포차 기여. A1 위반 판정 = matched CI 가 0 배제.
설계: match_two_cohorts 로 정합 → 코호트별 환자 K-fold × init seed. 영역 [RT,LT,RB,LB].

실행:  python e3_indomain_matched.py --folds 5 --init_seeds 42 1 2
산출:  checkpoint/E2/dorga/ (가중치·npz) + Result/E2/dorga/ (per-run json) + Result/E2/e2_summary.json
       (A1_test_matched=Δg_matched)
"""
import argparse, json, os as _os, sys as _sys
import pandas as pd
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E"))
import e_common as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5, help="fold 분할 수(파티션). 기본 5")
    ap.add_argument("--only-folds", dest="only_folds", type=int, nargs="+", default=None,
                    help="학습할 fold 인덱스(0~folds-1). 미지정=전체. 특정 fold 만 돌릴 때")
    ap.add_argument("--init_seeds", type=int, nargs="+", default=[42, 1, 2],
                    help="init seed 목록. 하나만 주면 그 seed 만 (예: --init_seeds 42)")
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2026], choices=[2024, 2026],
                    help="학습할 코호트. 한 연도만 주면 그 코호트만 학습·캐시(Δγ 는 두 연도 모두 있어야 산출)")
    ap.add_argument("--overwrite", dest="skip_existing", action="store_false",
                    help="기본은 test_preds.npz 있으면 학습 생략(skip-existing 기본 ON). 이 옵션이면 강제 재학습")
    ap.set_defaults(skip_existing=True)
    args = ap.parse_args()

    root = A.A1_OUT / "E2" / "dorga"              # checkpoint/E2/dorga
    out_dir = A.RESULT_OUT / "E2"
    keys = ["overall"] + list(A.ROI)
    df = A._prep(pd.read_csv(A.MANIFEST))         # match_two_cohorts 가 year·patient 사용
    keep = dict(zip((2024, 2026), A.match_two_cohorts(df, 0)))                    # 정합 seed 고정 0

    # 지정 연도만 학습 → 코호트별 부분결과를 캐시(e2_cohort_{yr}.json)
    cohort = {}
    for yr in args.years:
        m = A.run_cohort(yr, args.folds, 0, args.init_seeds, root, keep_pat=keep[yr],
                         tag="matched", skip_existing=args.skip_existing,
                         only_folds=args.only_folds)                              # fold 분할 seed 0
        A.save_json(m, out_dir / f"e2_cohort_{yr}.json")
        cohort[yr] = m
        print(f"  matched {yr}: ACC {m['acc']:.4f}  MAE {m['mae']:.4f}  (n_pat {m['n_pat']})")

    # 이번에 안 돌린 연도는 이전 캐시에서 로드
    for yr in (2024, 2026):
        if yr not in cohort and (out_dir / f"e2_cohort_{yr}.json").exists():
            cohort[yr] = json.loads((out_dir / f"e2_cohort_{yr}.json").read_text(encoding="utf-8"))

    if 2024 not in cohort or 2026 not in cohort:
        need = [y for y in (2024, 2026) if y not in cohort]
        print(f"\n[부분 완료] 보유 {sorted(cohort)} — {need} 도 실행해야 Δγ·e2_summary.json 산출.")
        return

    # 두 코호트 모두 확보 → Δγ = g_2024 − g_2026 산출
    out = {"matched": {"2024": cohort[2024], "2026": cohort[2026],
                       "n": {"2024": len(keep[2024]), "2026": len(keep[2026])}},
           "A1_test_matched": A.a1_test(cohort[2024], cohort[2026], keys)}
    A.save_json(out, out_dir / "e2_summary.json")

    print("\n" + "=" * 66)
    print("E2 matched in-domain Δg_matched = g_2024 − g_2026 (정합 후)")
    print(f"{'ROI':<9}{'Δg_matched':>14}{'CI':>26}{'A1(matched)':>14}")
    for key in keys:
        t = out["A1_test_matched"][key]
        flag = "위반 잔존" if t["reject_A1"] else "정합후 소멸"
        print(f"{key:<9}{t['delta_g']:>+14.3f}   [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]{flag:>14}")
    print(f"(정합 n: 2024={out['matched']['n']['2024']}명 · 2026={out['matched']['n']['2026']}명)")
    print("saved:", out_dir / "e2_summary.json")


if __name__ == "__main__":
    main()
