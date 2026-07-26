# -*- coding: utf-8 -*-
r"""
L3 프록시 독립성 — 전처리 일치판 (2024 도 2026 과 동일한 seg+STN)

앞 L3_features.py 는 2024=디스크 정규화본 / 2026=seg+STN 이라 전처리 경로가 달랐고,
그 결과 grad/var 에서 전 ROI 공통 잔차(전처리 교란 시그니처)가 나왔다.
여기서는 **2024 raw(image_original)를 2026 과 똑같은 build_2026_cache(seg+STN)** 로 처리해
두 코호트 전처리를 일치시킨 뒤 grad/var 로 L3 를 다시 돌린다.

판정: 전처리가 같아진 상태에서 grad/var 의 RB explained% 가 낮게(잔차 지배) 유지되고
      RT/LT/LB 는 회복되면 → RB 드리프트가 강도 프록시 아티팩트가 아님이 특징 독립적으로 확증.
      여전히 전 ROI 잔차면 → 전처리가 아니라 다른 전역 요인.

실행: python L3_features_matched.py   (torch_ev + GPU)
산출: Result/L/roi_features_matched.csv , Result/L/L3_matched_summary.txt
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "E"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 같은 폴더 L3_features
import e_common as A
import l3_features as L               # features(), l3_explain(), R4 재사용

IMG_ORIG = r"D:\MICCAI2026\inhauh\image_original"


def build_matched():
    df = L.CSV
    # 2024 raw 를 2026 과 동일 seg+STN 으로
    raw24 = sorted({os.path.join(IMG_ORIG, Path(r["image_path"]).name)
                    for _, r in df[df.year == 2024].iterrows()})
    print(f"2024 raw {len(raw24)}장 → seg+STN ...")
    cache24 = A.B.build_2026_cache(raw24)
    print(f"2026 → seg+STN ...")
    cache26 = A.build_full_2026_cache()

    rows = []
    for _, r in df.iterrows():
        y = int(r["year"])
        if y == 2024:
            key = os.path.join(IMG_ORIG, Path(r["image_path"]).name)
            src = cache24
        else:
            key = A.B.raw_path_from_npz(r["image_path"]); src = cache26
        if key not in src:
            continue
        img_u8, msk_u8 = src[key]
        f = L.features(img_u8.astype(np.float32) / 255.0, msk_u8 > 0)
        f.update(dict(year=y, patient=r["patient"],
                      RT=int(r["RT"]), LT=int(r["LT"]), RB=int(r["RB"]), LB=int(r["LB"])))
        rows.append(f)
    return pd.DataFrame(rows)


def main():
    df = build_matched()
    df.to_csv(A.RESULT_OUT / "L" / "roi_features_matched.csv", index=False, encoding="utf-8")
    Lm = ["L3 explained% — 전처리 일치판 (2024·2026 모두 seg+STN)",
          "낮을수록 잔차 지배(드리프트 신호). RB 만 낮고 나머지 회복되면 프록시 독립 확증.",
          f"{'특징':>6}" + "".join(f"{r:>9}" for r in L.R4)]
    for feat in ["mu", "grad", "var", "p90"]:
        e = L.l3_explain(df, feat)
        Lm.append(f"{feat:>6}" + "".join(f"{e[r]['explained']*100:>8.0f}%" for r in L.R4))
    Lm.append("\n상세 (crude → adj):")
    for feat in ["mu", "grad", "var", "p90"]:
        e = L.l3_explain(df, feat)
        Lm.append(f"  [{feat}] " + "  ".join(
            f"{r} {e[r]['crude']:+.4f}→{e[r]['adj']:+.4f}({e[r]['explained']*100:.0f}%)" for r in L.R4))
    Lm.append(f"\nn: 2024 {int((df.year==2024).sum())}장 / 2026 {int((df.year==2026).sum())}장 · "
              f"fallback {int(df.fallback.sum())}건")
    txt = "\n".join(Lm)
    (A.RESULT_OUT / "L" / "L3_matched_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
