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

BASE = "/shared/home/mai/JeongGeon/Private"
OUT_ROOT = Path(f"{BASE}/w6_out")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(OUT_ROOT / "runs"))
    args = ap.parse_args()
    runs_dir = Path(args.runs)

    recs = load_all(runs_dir)
    if not recs:
        print(f"[중단] results.json 없음: {runs_dir}")
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

    # 저장
    out_root = runs_dir.parent
    csv_path = out_root / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model", "roi", "fwd_bias_mean", "fwd_bias_sd",
                                          "rev_bias_mean", "rev_bias_sd", "flip_Hdata"])
        w.writeheader(); w.writerows(csv_rows)
    (out_root / "verdict.txt").write_text("\n".join(verdict_lines), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"[save] {csv_path}")
    print(f"[save] {out_root / 'verdict.txt'}")
    print("해석: 특정 영역에서 3모델 모두 flip=YES 면, 아키텍처 무관한 라벨 드리프트(H_data) 근거.")


if __name__ == "__main__":
    main()
