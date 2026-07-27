# -*- coding: utf-8 -*-
r"""
E2 — in-domain raw Δg 추정 (전 환자)   [draft §5.2 raw / 舊 E1 부록 B]

목적: g_2024·g_2026 을 in-domain(c→c)으로 추정해 A1 검정(Δg_raw).
      in-domain 은 라벨 오프셋 b_c 가 상쇄되어 bias(c→c)=g_c(순수 모델오차).
설계: 코호트별 환자 단위 K-fold(모든 환자 1회 test) × init seed 반복. 영역 [RT,LT,RB,LB].
      matched(등급분포 정합)는 E3 에서 별도 산출 → 수축분(Δg_raw−Δg_matched) 은 summary.py 판정.

실행:  python e2_indomain_raw.py --folds 5 --init_seeds 42 1 2
산출:  checkpoint/E2/dorga/ (가중치·npz) + Result/E2/dorga/ (per-run json) + Result/E2/e2_summary.json
       (A1_test=Δg_raw)
"""
import argparse, json, os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E"))
import e_common as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--init_seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2026], choices=[2024, 2026],
                    help="학습할 코호트. 한 연도만 주면 그 코호트만 학습·캐시(Δg 는 두 연도 모두 있어야 산출)")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                    help="arm 의 test_preds.npz 가 이미 있으면 학습 생략(중단 후 재개용)")
    args = ap.parse_args()

    root = A.A1_OUT / "E2" / "dorga"              # checkpoint/E2/dorga
    out_dir = A.RESULT_OUT / "E2"
    keys = ["overall"] + list(A.ROI)

    cohort = {}
    for yr in args.years:
        m = A.run_cohort(yr, args.folds, 0, args.init_seeds, root, tag="raw",
                         skip_existing=args.skip_existing)                        # fold 분할 seed 고정 0
        A.save_json(m, out_dir / f"e2_cohort_{yr}.json")
        cohort[yr] = m
        print(f"  raw {yr}: ACC {m['acc']:.4f}  MAE {m['mae']:.4f}  (n_pat {m['n_pat']})")

    for yr in (2024, 2026):
        if yr not in cohort and (out_dir / f"e2_cohort_{yr}.json").exists():
            cohort[yr] = json.loads((out_dir / f"e2_cohort_{yr}.json").read_text(encoding="utf-8"))

    if 2024 not in cohort or 2026 not in cohort:
        need = [y for y in (2024, 2026) if y not in cohort]
        print(f"\n[부분 완료] 보유 {sorted(cohort)} — {need} 도 실행해야 Δg·e2_summary.json 산출.")
        return

    out = {"raw": {"2024": cohort[2024], "2026": cohort[2026]},
           "A1_test": A.a1_test(cohort[2024], cohort[2026], keys)}
    A.save_json(out, out_dir / "e2_summary.json")

    print("\n" + "=" * 66)
    print("E2 raw in-domain Δg_raw = g_2024 − g_2026")
    print(f"{'ROI':<9}{'Δg_raw':>12}{'CI':>26}{'A1(raw)':>12}")
    for key in keys:
        t = out["A1_test"][key]
        flag = "위반 신호" if t["reject_A1"] else "미기각"
        print(f"{key:<9}{t['delta_g']:>+12.3f}   [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]{flag:>12}")
    print("saved:", out_dir / "e2_summary.json")


if __name__ == "__main__":
    main()
