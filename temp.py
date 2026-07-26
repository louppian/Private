# -*- coding: utf-8 -*-
r"""
temp.py — rev 백본 스모크. 아래 MRM_PATH 에 백본 경로 붙여넣고  python temp.py.

우리 계획 세팅 고정(val 5-fold split1 · best_val · 50ep · early-stop 10 · FREEZE 6),
백본만 교체해 rev(2026→2024) 을 돌린다. 비교 기준:
  md rev +0.004  ·  현재 Brixia 백본 E1 rev +0.224
rev 이 +0.004 로 붙으면 백본이 원인 확정 → core.MRM_W 를 그걸로 확정.
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


def run():
    if not Path(A.B.CSV_PATH).exists():
        print(f"[SKIP] 데이터 없음: {A.B.CSV_PATH} — 서버에서 실행")
        return
    if MRM_PATH:
        A.B.MRM_W = Path(MRM_PATH)                     # 백본만 교체
    bk = Path(A.B.MRM_W).stem
    ty, ey, tag = E.DIRS["rev"]                        # (2026, 2024, "26to24")
    arm = f"{tag}_split1"
    root = A.A1_OUT / "E1_smoke" / bk
    A.register(arm, E.cross_val_fold(ty, ey, 0))       # val fold0 (split1)
    print("#" * 78)
    print(f"# rev  백본={bk}  MRM={A.B.MRM_W}")
    print(f"#   {arm}: train 2026 80% / val 20% / test 2024 전체 · 50ep · early-stop 10 · best_val")
    print("#" * 78)
    A.B.run_one_dorga(arm, 1, 50, root, arm=arm)       # seed 1, 우리 세팅

    r = json.loads((root / arm / "results.json").read_text(encoding="utf-8"))
    print("\n" + "=" * 60)
    print(f"rev ({bk})  overall {r['bias']:+.3f}  "
          f"RB {r['per_roi']['RB']['bias']:+.3f}  LT {r['per_roi']['LT']['bias']:+.3f}")
    print(f"기준        md rev +0.004  ·  Brixia rev +0.224")
    print("→ +0.004 에 붙으면 백본이 원인. → core.MRM_W 를 그걸로 확정.")
    print("=" * 60)


if __name__ == "__main__":
    run()
