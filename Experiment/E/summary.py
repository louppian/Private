# -*- coding: utf-8 -*-
r"""
w6 결과 집계 — runs/<model>/<mode>_s<seed>/results.json 18개를 읽어
방향(fwd/rev) × 모델 × 영역별 test bias 를 seed 평균±sd 로 요약하고 방향반전을 판정.

핵심 질문(방향 반전): fwd(train2024→test2026) 에서 음수이던 bias 가 rev(train2026→
test2024) 에서 양수로 뒤집히면 = 코호트 간 라벨 calibration 차이(H_data), 그대로면 모델효과.

산출:
  콘솔 표 + <OUT_ROOT>/summary.csv + <OUT_ROOT>/verdict.txt

실행:
  python w6_summary.py                      # 기본 경로(CONFIG)
  python w6_summary.py --runs /path/to/w6_out/runs
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent          # Experiment/E
_REPO = _HERE.parents[1]                          # Private repo 루트
RUNS_DIR   = _REPO / "checkpoint" / "E1" / "runs" # 입력: E1 cross 산출 <model>/<mode>_s<seed>/results.json
RESULT_DIR = _REPO / "Result" / "E"               # 출력: 집계 CSV (git 추적)
ROI = ["RT", "LT", "RB", "LB"]
DIR_LABEL = {"2024to2026": "fwd", "2026to2024": "rev"}
MODELS = ["dorga", "bsnet", "pafe"]


def load_all(runs_dir: Path):
    recs = []
    for p in sorted(runs_dir.glob("*/*/results.json")):
        try:
            recs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[skip] {p}: {e}")
    return recs


def ms(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def fmt(m, s):
    return f"{m:+.3f}±{s:.3f}" if not np.isnan(m) else "     -"


def export_e2():
    """checkpoint/E2/E2_summary.json → Result/E/e2_delta_g.csv (Δg_raw·matched·CI, git 추적)."""
    e2p = _REPO / "checkpoint" / "E2" / "E2_summary.json"
    if not e2p.exists():
        return
    e2 = json.loads(e2p.read_text(encoding="utf-8"))
    raw, mat = e2.get("A1_test", {}), e2.get("A1_test_matched", {})
    rows = []
    for roi in ["overall"] + ROI:
        r, m = raw.get(roi, {}), mat.get(roi, {})
        ci = m.get("ci", [None, None]) or [None, None]
        rows.append(dict(roi=roi,
                         delta_g_raw=round(r["delta_g"], 4) if "delta_g" in r else "",
                         delta_g_matched=round(m["delta_g"], 4) if "delta_g" in m else "",
                         ci_lo=round(ci[0], 4) if ci[0] is not None else "",
                         ci_hi=round(ci[1], 4) if ci[1] is not None else "",
                         reject_A1_matched=int(m["reject_A1"]) if "reject_A1" in m else ""))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULT_DIR / "e2_delta_g.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["roi", "delta_g_raw", "delta_g_matched",
                                          "ci_lo", "ci_hi", "reject_A1_matched"])
        w.writeheader(); w.writerows(rows)
    print(f"[save] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS_DIR))
    args = ap.parse_args()
    runs_dir = Path(args.runs)

    export_e2()                         # E2 Δg → Result/E/e2_delta_g.csv (cross 유무와 무관)

    recs = load_all(runs_dir)
    if not recs:
        print(f"[중단] cross results.json 없음: {runs_dir} (E2 CSV 는 위에서 저장됨)")
        return
    print(f"[load] results.json {len(recs)}개  ({runs_dir})")

    # (model, dir) -> [records]
    G = defaultdict(list)
    for r in recs:
        G[(r["model"], DIR_LABEL.get(r["mode"], r["mode"]))].append(r)

    csv_rows = []
    verdict_lines = []
    for model in MODELS:
        fwd, rev = G.get((model, "fwd"), []), G.get((model, "rev"), [])
        if not (fwd or rev):
            continue
        head = f"[{model}]  fwd(train2024→test2026) vs rev(train2026→test2024)  " \
               f"| seeds fwd={len(fwd)} rev={len(rev)}"
        print("\n" + "=" * 78); print(head); print("=" * 78)
        verdict_lines.append("\n" + head)

        # 전체 test 지표
        for tag, rr in (("fwd", fwd), ("rev", rev)):
            if not rr:
                continue
            mae = ms([r["mae"] for r in rr]); acc = ms([r["acc"] for r in rr])
            bia = ms([r["bias"] for r in rr])
            line = f"  [{tag}] test  MAE {fmt(*mae).replace('+','')}  " \
                   f"ACC {fmt(*acc).replace('+','')}  overall bias {fmt(*bia)}"
            print(line); verdict_lines.append(line)

        # 영역별 bias 표 + 방향반전
        print(f"\n  {'ROI':<6}{'fwd bias':>16}{'rev bias':>16}{'flip(H_data)':>16}")
        n_flip = 0
        for roi in ["overall"] + ROI:
            if roi == "overall":
                fb = [r["bias"] for r in fwd]; rb = [r["bias"] for r in rev]
            else:
                fb = [r["per_roi"][roi]["bias"] for r in fwd]
                rb = [r["per_roi"][roi]["bias"] for r in rev]
            fm, fs = ms(fb); rm, rs = ms(rb)
            # H_data 방향반전: fwd 음수 → rev 양수 (과소예측이 반대편에서 과대예측으로)
            flip = (not np.isnan(fm) and not np.isnan(rm) and fm < 0 < rm)
            if roi != "overall" and flip:
                n_flip += 1
            mark = "  YES" if flip else ("  no" if not np.isnan(fm) else "  -")
            row = f"  {roi:<6}{fmt(fm, fs):>16}{fmt(rm, rs):>16}{mark:>16}"
            print(row); verdict_lines.append(row)
            csv_rows.append(dict(model=model, roi=roi,
                                 fwd_bias_mean=round(fm, 4), fwd_bias_sd=round(fs, 4),
                                 rev_bias_mean=round(rm, 4), rev_bias_sd=round(rs, 4),
                                 flip_Hdata=int(flip)))

        v = f"  → {model}: 4개 영역 중 {n_flip}개에서 H_data 방향반전(fwd<0<rev)"
        print(v); verdict_lines.append(v)

    # 저장 → Result/E/ (git 추적)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model", "roi", "fwd_bias_mean", "fwd_bias_sd",
                                          "rev_bias_mean", "rev_bias_sd", "flip_Hdata"])
        w.writeheader(); w.writerows(csv_rows)
    (RESULT_DIR / "verdict.txt").write_text("\n".join(verdict_lines), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"[save] {csv_path}")
    print(f"[save] {RESULT_DIR / 'verdict.txt'}")
    print("해석: 특정 영역에서 3모델 모두 flip=YES 면, 아키텍처 무관한 라벨 드리프트(H_data) 근거.")


if __name__ == "__main__":
    main()
