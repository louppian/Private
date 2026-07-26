# -*- coding: utf-8 -*-
r"""
w6 결과 집계 — runs/<model>/<mode>_s<seed>/results.json 18개를 읽어
방향(fwd/rev) × 모델 × 영역별 test bias 를 seed 평균±sd 로 요약하고 방향반전을 판정.

핵심 질문(방향 반전): fwd(train2024→test2026) 에서 음수이던 bias 가 rev(train2026→
test2024) 에서 양수로 뒤집히면 = 코호트 간 라벨 calibration 차이(H_data), 그대로면 모델효과.

산출:
  콘솔 표 + Result/E1/e1_summary.csv (실험당 단일 요약 CSV)

실행:
  python Experiment/E/summary.py            # checkpoint/E1/dorga 집계 → Result/E1/e1_summary.csv
  python Experiment/E/summary.py --runs /path/to/runs
  (E2 raw Δg → Result/E2/e2_summary.csv, E3 matched Δg → Result/E3/e3_summary.csv 함께 export)
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent          # Experiment/E
_REPO = _HERE.parents[1]                          # Private repo 루트
RUNS_DIR   = _REPO / "Result" / "E1" / "dorga"      # 입력: {24to26,26to24}_split{k}/results.json (json 은 Result)
RESULT_E1  = _REPO / "Result" / "E1"                 # 출력: E1 cross 집계 (checkpoint/E1 대응)
RESULT_E2  = _REPO / "Result" / "E2"                 # 출력: E2 in-domain raw Δg
RESULT_E3  = _REPO / "Result" / "E3"                 # 출력: E3 in-domain matched Δg
ROI = ["RT", "LT", "RB", "LB"]


def _dir(mode):
    return "fwd" if str(mode).startswith("24to26") else "rev"


def load_all(runs_dir: Path):
    recs = []
    for p in sorted(runs_dir.glob("*/results.json")):     # {tag}_split{k}/results.json
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


def _export_indomain(summary_json, a1_key, dg_col, reject_col, out_dir, out_name):
    """checkpoint/<E>/dorga/<summary>.json 의 A1 결과 → Result/<E>/<name>.csv (roi·Δg·CI·reject)."""
    if not summary_json.exists():
        return
    d = json.loads(summary_json.read_text(encoding="utf-8"))
    a1 = d.get(a1_key, {})
    if not a1:
        return
    rows = []
    for roi in ["overall"] + ROI:
        t = a1.get(roi, {})
        ci = t.get("ci", [None, None]) or [None, None]
        rows.append({"roi": roi,
                     dg_col: round(t["delta_g"], 4) if "delta_g" in t else "",
                     "ci_lo": round(ci[0], 4) if ci[0] is not None else "",
                     "ci_hi": round(ci[1], 4) if ci[1] is not None else "",
                     reject_col: int(t["reject_A1"]) if "reject_A1" in t else ""})
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / out_name
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["roi", dg_col, "ci_lo", "ci_hi", reject_col])
        w.writeheader(); w.writerows(rows)
    print(f"[save] {p}")


def export_e2():
    """E2 raw: Result/E2/e2_summary.json → Result/E2/e2_summary.csv (Δg_raw·CI·reject)."""
    _export_indomain(RESULT_E2 / "e2_summary.json", "A1_test",
                     "delta_g_raw", "reject_A1_raw", RESULT_E2, "e2_summary.csv")


def export_e3():
    """E3 matched: Result/E3/e3_summary.json → Result/E3/e3_summary.csv (Δg_matched·CI·reject)."""
    _export_indomain(RESULT_E3 / "e3_summary.json", "A1_test_matched",
                     "delta_g_matched", "reject_A1_matched", RESULT_E3, "e3_summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS_DIR))
    args = ap.parse_args()
    runs_dir = Path(args.runs)

    export_e2()                         # E2 raw Δg     → Result/E2/e2_summary.csv
    export_e3()                         # E3 matched Δg → Result/E3/e3_summary.csv

    recs = load_all(runs_dir)
    if not recs:
        print(f"[중단] cross results.json 없음: {runs_dir} (E2 CSV 는 위에서 저장됨)")
        return
    print(f"[load] results.json {len(recs)}개  ({runs_dir})")

    G = {"fwd": [], "rev": []}
    for r in recs:
        G[_dir(r["mode"])].append(r)
    fwd, rev = G["fwd"], G["rev"]

    csv_rows = []
    head = f"[dorga]  fwd(2024→2026) vs rev(2026→2024)  | splits fwd={len(fwd)} rev={len(rev)}"
    print("\n" + "=" * 78); print(head); print("=" * 78)

    for tag, rr in (("fwd", fwd), ("rev", rev)):
        if not rr:
            continue
        mae = ms([r["mae"] for r in rr]); acc = ms([r["acc"] for r in rr]); bia = ms([r["bias"] for r in rr])
        line = f"  [{tag}] test  MAE {fmt(*mae).replace('+','')}  " \
               f"ACC {fmt(*acc).replace('+','')}  overall bias {fmt(*bia)}"
        print(line)

    print(f"\n  {'ROI':<6}{'fwd bias':>16}{'rev bias':>16}{'flip(H_data)':>16}")
    n_flip = 0
    for roi in ["overall"] + ROI:
        if roi == "overall":
            fb = [r["bias"] for r in fwd]; rb = [r["bias"] for r in rev]
        else:
            fb = [r["per_roi"][roi]["bias"] for r in fwd]
            rb = [r["per_roi"][roi]["bias"] for r in rev]
        fm, fs = ms(fb); rm, rs = ms(rb)
        flip = (not np.isnan(fm) and not np.isnan(rm) and fm < 0 < rm)
        if roi != "overall" and flip:
            n_flip += 1
        mark = "  YES" if flip else ("  no" if not np.isnan(fm) else "  -")
        row = f"  {roi:<6}{fmt(fm, fs):>16}{fmt(rm, rs):>16}{mark:>16}"
        print(row)
        csv_rows.append(dict(model="dorga", roi=roi,
                             fwd_bias_mean=round(fm, 4), fwd_bias_sd=round(fs, 4),
                             rev_bias_mean=round(rm, 4), rev_bias_sd=round(rs, 4),
                             flip_Hdata=int(flip)))

    v = f"  → dorga: 4개 영역 중 {n_flip}개에서 방향반전(fwd<0<rev)"
    print(v)

    # 저장 → Result/E1/e1_summary.csv (git 추적, 실험당 단일 요약 CSV)
    RESULT_E1.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_E1 / "e1_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model", "roi", "fwd_bias_mean", "fwd_bias_sd",
                                          "rev_bias_mean", "rev_bias_sd", "flip_Hdata"])
        w.writeheader(); w.writerows(csv_rows)

    print("\n" + "=" * 78)
    print(f"[save] {csv_path}")
    print("해석: 특정 영역에서 flip=YES(fwd<0<rev) 면 아키텍처 무관한 라벨 드리프트(H_data) 근거.")


if __name__ == "__main__":
    main()
