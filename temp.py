# -*- coding: utf-8 -*-
r"""
temp.py — rev 백본 스모크. 아래 MRM_PATH 에 백본 경로 붙여넣고  python temp.py.

우리 계획 세팅 고정(val 5-fold split1~5 비중복 · best_val · 50ep · early-stop 10 · FREEZE 6),
rev(2026→2024) 을 raw(전 환자)·matched(2024·2026 등급분포 정합) 두 조건으로 돌려 대조.
  md rev +0.004  ·  현재 Brixia raw rev +0.224
matched rev 이 raw보다 뚝 떨어져 ~0 이면 분포차(RTM)가 진범, 여전히 부풀면 백본/진짜 비대칭.
"""
# ═══════════ 백본 경로 붙여넣기 (빈 문자열이면 core 기본 = DORGA_Brixia.pth) ═══════════
MRM_PATH = ""
# ══════════════════════════════════════════════════════════════════════════════════════

import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parent
for _p in (str(REPO / "Experiment"), str(REPO / "Experiment" / "E"),
           str(REPO / "Model"), str(REPO / "Model" / "DORGA")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e_common as A          # noqa: E402
import e1_cross as E          # noqa: E402


SPLITS = 5      # val 5-fold 비중복 (원래계획). rev 을 split1~5 평균으로 본다.


def _run_variant(variant, ty, ey, tag, bk, root):
    """variant='raw' or 'matched' 로 rev split1~5 돌리고 평균 반환."""
    per_split = []
    for k in range(SPLITS):
        arm = f"{tag}_{variant}_split{k+1}"
        if variant == "matched":
            A.register(arm, E.cross_val_fold_matched(ty, ey, k, folds=SPLITS))
        else:
            A.register(arm, E.cross_val_fold(ty, ey, k, folds=SPLITS))
        A.B.run_one_dorga(arm, 1, 50, root, arm=arm)                 # seed 1, 우리 세팅
        r = json.loads((root / arm / "results.json").read_text(encoding="utf-8"))
        per_split.append(dict(split=k + 1, overall=r["bias"],
                              RB=r["per_roi"]["RB"]["bias"], LT=r["per_roi"]["LT"]["bias"]))
    _m = lambda key: sum(s[key] for s in per_split) / len(per_split)
    return per_split, {"overall": _m("overall"), "RB": _m("RB"), "LT": _m("LT")}


def run():
    if not Path(A.B.CSV_PATH).exists():
        print(f"[SKIP] 데이터 없음: {A.B.CSV_PATH} — 서버에서 실행")
        return
    if MRM_PATH:
        A.B.MRM_W = Path(MRM_PATH)                     # 백본만 교체
    bk = Path(A.B.MRM_W).stem
    ty, ey, tag = E.DIRS["rev"]                        # (2026, 2024, "26to24")
    root = A.A1_OUT / "E1_smoke" / bk
    print("#" * 78)
    print(f"# rev  백본={bk}  MRM={A.B.MRM_W}")
    print(f"#   {tag}: train 2026 80% / val 20% 비중복 / test 2024 · 50ep · es10 · best_val · seed 1")
    print(f"#   raw(전 환자) vs matched(2024·2026 등급분포 정합) — RTM(분포차) 분리")
    print("#" * 78)

    out = {}
    for variant in ("raw", "matched"):
        print("\n" + "#" * 30 + f"  {variant.upper()}  " + "#" * 30)
        per_split, mean = _run_variant(variant, ty, ey, tag, bk, root)
        out[variant] = (per_split, mean)

    print("\n" + "=" * 66)
    print(f"rev ({bk})   raw vs matched   [overall / RB / LT]  seed1 · 5-split 평균")
    print("-" * 66)
    for variant in ("raw", "matched"):
        per_split, mean = out[variant]
        for s in per_split:
            print(f"  {variant:7} split{s['split']}  overall {s['overall']:+.3f}  "
                  f"RB {s['RB']:+.3f}  LT {s['LT']:+.3f}")
        print(f"  {variant:7} 평균     overall {mean['overall']:+.3f}  "
              f"RB {mean['RB']:+.3f}  LT {mean['LT']:+.3f}")
        print("-" * 66)
    print(f"기준        md rev +0.004  ·  Brixia raw rev +0.224")
    print("→ matched rev 이 raw보다 뚝 떨어져 ~0 이면  '분포차(RTM)'가 진범.")
    print("→ matched 도 여전히 부풀면  백본/진짜 비대칭.")
    print("=" * 66)


if __name__ == "__main__":
    run()
