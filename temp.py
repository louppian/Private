# -*- coding: utf-8 -*-
r"""
temp.py — recipe 스모크: fwd·rev 를 따로 돌려 final / best_val / tail-5 로 rev·δ 비교.

목적: md(원 분석)는 EARLYSTOP=None(50 완주) + final/tail-5 → rev≈+0.004.
      우리 best_val 규약과 어느 규약이 md 에 붙는지 확인.

동작: --mode fwd|rev 로 한 방향씩 학습(50ep 고정, early-stop OFF=md).
      --mode summary 로 학습 없이 fwd·rev 를 읽어 final·best_val·tail-5 δ 표 + md 대조.
산출: checkpoint/E1_smoke/dorga/{24to26,26to24}_split1/  (실 E1 미간섭).

실행(서버, 데이터·GPU 필요):
  python temp.py --mode fwd
  python temp.py --mode rev
  python temp.py --mode summary   # δ 표 출력
"""
import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parent
EXP = REPO / "Experiment"
for _p in (str(EXP), str(EXP / "E"), str(REPO / "Model"), str(REPO / "Model" / "DORGA")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import e_common as A          # noqa: E402
import e1_cross as E          # noqa: E402  (cross_val_fold, DIRS)

EPOCHS = 50                    # 고정 (md 완주)
ROOT = A.A1_OUT / "E1_smoke" / "dorga"


def recipe_biases(run_dir):
    """history.json per-epoch test_bias → final / best_val / tail5 (overall bias)."""
    if not (run_dir / "history.json").exists():
        return None
    hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    res = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    tb = [(h["epoch"], h["test_bias"]) for h in hist if "test_bias" in h]
    if not tb:
        return None
    by_ep = dict(tb)
    best_ep = res.get("best_val_epoch")
    return {"final": tb[-1][1], "best_val": by_ep.get(best_ep, tb[-1][1]),
            "tail5": float(np.mean([b for _, b in tb[-5:]])), "best_epoch": best_ep}


def summarize():
    both = {m: recipe_biases(ROOT / f"{E.DIRS[m][2]}_split1") for m in ("fwd", "rev")}
    missing = [m for m in ("fwd", "rev") if not both[m]]
    if missing:
        print(f"[대기] 먼저 학습:  " + "  ".join(f"python temp.py --mode {m}" for m in missing))
        return
    print("\n" + "=" * 78)
    print("recipe 비교 (overall test bias, δ=(rev−fwd)/2)  · md: fwd −0.231 rev +0.004 δ +0.118")
    print("=" * 78)
    print(f"  {'recipe':<10}{'fwd':>10}{'rev':>10}{'δ_obs':>10}")
    for rc in ("final", "best_val", "tail5"):
        f, r = both["fwd"][rc], both["rev"][rc]
        print(f"  {rc:<10}{f:>+10.3f}{r:>+10.3f}{(r - f) / 2:>+10.3f}")
    print(f"\n  (best_val epoch: fwd {both['fwd']['best_epoch']} · rev {both['rev']['best_epoch']})")
    print("  → rev 이 +0.004 에 가장 가까운 recipe = md 규약. 그걸로 core 확정.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fwd", "rev", "summary"])
    args = ap.parse_args()

    if args.mode == "summary":                        # 학습 없이 δ 표만
        summarize()
        return

    if not Path(A.B.CSV_PATH).exists():
        print(f"[SKIP] 데이터 없음: {A.B.CSV_PATH} — 서버에서 실행")
        return

    A.B.EARLYSTOP_PATIENCE = None                     # md 규약: early-stop OFF
    ty, ey, tag = E.DIRS[args.mode]
    arm = f"{tag}_split1"
    A.register(arm, E.cross_val_fold(ty, ey, 0))
    print("\n" + "#" * 78)
    print(f"# {args.mode}  {arm}  (train {ty} 80% / val 20% / test {ey} 전체)  epochs={EPOCHS}, early-stop OFF")
    print("#" * 78)
    A.B.run_one_dorga(arm, 1, EPOCHS, ROOT, arm=arm)

    b = recipe_biases(ROOT / arm)
    print(f"\n[{args.mode}] final {b['final']:+.3f} · best_val {b['best_val']:+.3f}"
          f"(ep {b['best_epoch']}) · tail5 {b['tail5']:+.3f}")
    print("  → 둘 다 끝나면:  python temp.py --mode summary")


if __name__ == "__main__":
    main()
