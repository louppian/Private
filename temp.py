# -*- coding: utf-8 -*-
r"""
temp.py — recipe 스모크: 같은 학습을 final / best_val / tail-5 세 규약으로 읽어 rev·δ 비교.

목적: md(원 분석)는 EARLYSTOP=None(50 완주) + final/tail-5 로 보고 → rev≈+0.004.
      우리 best_val 규약과 어느 규약이 md 에 붙는지 한 arm 으로 빠르게 확인.

동작:
  - EARLYSTOP 끄고(=md) fwd split1, rev split1 을 --epochs 만큼 학습.
  - 각 arm 의 history.json 에서 per-epoch test_bias 궤적을 읽어
    final(마지막) / best_val(best epoch) / tail-5(마지막 5 평균) bias 를 뽑고
    δ_obs=(rev−fwd)/2 를 세 규약으로 계산해 md(δ +0.118) 와 대조.
  - 산출은 checkpoint/E1_smoke/dorga/ (실 E1 안 건드림).

실행(서버, 데이터·GPU 필요):
  python temp.py                 # epochs 25 (스모크)
  python temp.py --epochs 50     # md 와 동일 완주
"""
import argparse
import json
import os
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


def recipe_biases(run_dir):
    """history.json per-epoch test_bias → final / best_val / tail5 (overall bias)."""
    hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    res = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    tb = [(h["epoch"], h["test_bias"]) for h in hist if "test_bias" in h]
    if not tb:
        return None
    by_ep = dict(tb)
    final = tb[-1][1]
    best_ep = res.get("best_val_epoch")
    best = by_ep.get(best_ep, final)
    tail5 = float(np.mean([b for _, b in tb[-5:]]))
    return {"final": final, "best_val": best, "tail5": tail5, "best_epoch": best_ep, "n_test_ep": len(tb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25, help="스모크 에폭(md 완주는 50)")
    args = ap.parse_args()

    if not Path(A.B.CSV_PATH).exists():
        print(f"[SKIP] 데이터 없음: {A.B.CSV_PATH} — 서버에서 실행")
        return

    # md 규약: early-stop 끔 (50/args 완주)
    A.B.EARLYSTOP_PATIENCE = None
    root = A.A1_OUT / "E1_smoke" / "dorga"

    out = {}
    for m in ("fwd", "rev"):
        ty, ey, tag = E.DIRS[m]
        arm = f"{tag}_split1"
        A.register(arm, E.cross_val_fold(ty, ey, 0))
        print("\n" + "#" * 78)
        print(f"# {m}  {arm}  (train {ty} 80% / val 20% / test {ey} 전체)  epochs={args.epochs}, early-stop OFF")
        print("#" * 78)
        A.B.run_one_dorga(arm, 1, args.epochs, root, arm=arm)
        out[m] = recipe_biases(root / arm)

    print("\n" + "=" * 78)
    print("recipe 비교 (overall test bias, δ=(rev−fwd)/2)  · md: fwd −0.231 rev +0.004 δ +0.118")
    print("=" * 78)
    print(f"  {'recipe':<10}{'fwd':>10}{'rev':>10}{'δ_obs':>10}")
    for rc in ("final", "best_val", "tail5"):
        f, r = out["fwd"][rc], out["rev"][rc]
        print(f"  {rc:<10}{f:>+10.3f}{r:>+10.3f}{(r - f) / 2:>+10.3f}")
    print(f"\n  (best_val epoch: fwd {out['fwd']['best_epoch']} · rev {out['rev']['best_epoch']}"
          f" / test 평가 에폭수 {out['fwd']['n_test_ep']})")
    print("  → rev 이 +0.004 에 가장 가까운 recipe = md 규약. 그걸로 core 확정.")


if __name__ == "__main__":
    main()
