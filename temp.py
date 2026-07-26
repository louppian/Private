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
    print(f"# rev MATCHED  백본={bk}  MRM={A.B.MRM_W}")
    print(f"#   {tag} 정합: 2024·2026 등급분포 맞춘 환자만 · val 5-fold 비중복 · test 2024")
    print(f"#   50ep · es10 · best_val · seed 1   (raw 는 이미 값 있음 — 안 돌림)")
    print("#" * 78)

    per_split = []
    for k in range(SPLITS):
        arm = f"{tag}_matched_split{k+1}"
        A.register(arm, E.cross_val_fold_matched(ty, ey, k, folds=SPLITS))
        A.B.run_one_dorga(arm, 1, 50, root, arm=arm)                 # seed 1, 우리 세팅
        r = json.loads((root / arm / "results.json").read_text(encoding="utf-8"))
        per_split.append(dict(split=k + 1, overall=r["bias"],
                              RB=r["per_roi"]["RB"]["bias"], LT=r["per_roi"]["LT"]["bias"]))
    _m = lambda key: sum(s[key] for s in per_split) / len(per_split)

    print("\n" + "=" * 60)
    print(f"rev MATCHED ({bk})  split별 overall/RB/LT")
    for s in per_split:
        print(f"  split{s['split']}  overall {s['overall']:+.3f}  "
              f"RB {s['RB']:+.3f}  LT {s['LT']:+.3f}")
    print("-" * 60)
    print(f"matched 평균  overall {_m('overall'):+.3f}  RB {_m('RB'):+.3f}  LT {_m('LT'):+.3f}")
    print(f"기준          md rev +0.004  ·  Brixia raw rev +0.224  ·  MRM raw rev +0.138")
    print("→ matched 가 ~0 으로 떨어지면 분포차(RTM)가 진범, 부풀면 백본/진짜 비대칭.")
    print("=" * 60)


if __name__ == "__main__":
    run()
