# -*- coding: utf-8 -*-
r"""
temp.py — 백본 가설 스모크 (우리 계획 실험 세팅 고정, 백본만 교체).

배경: recipe(final/best_val/tail5·early-stop)는 배제됨 — 어느 것도 rev 을 md(+0.004)로
      못 내림. fwd(−0.233)는 md 와 완벽 일치, in-domain γ_2026(+0.014) 정상인데
      **cross rev 만 +0.224**. 유일하게 남은 config 차이 = 백본:
        md  = MRM.pth        (MAE 프리트레인 ViT)
        우리 = DORGA_Brixia.pth (Brixia 학습 DORGA)  ← 등급 사전지식이 cross 비대칭 유발 의심.

세팅(고정, 우리 계획 그대로): E1 cross · val 5-fold(8:2, split1) · best_val reload ·
      50ep · early-stop 10 · FREEZE 6 · lr enc1e-5/head1e-4.  **백본만 --mrm 로 교체.**

동작:
  --mode fwd|rev  : 한 방향 학습(위 세팅 고정). --bk 라벨로 dorga_<bk>/ 에 저장.
  --mode summary  : dorga_* 백본별 fwd/rev·δ_obs 를 md 와 대조.

실행(서버):
  python temp.py --mode fwd --bk brixia                          # 현재 백본
  python temp.py --mode rev --bk brixia
  python temp.py --mode fwd --bk mrm --mrm /path/to/MRM.pth      # md 백본
  python temp.py --mode rev --bk mrm --mrm /path/to/MRM.pth
  python temp.py --mode summary                                  # 백본별 rev·δ 비교
산출: checkpoint/E1_smoke/dorga_<bk>/{24to26,26to24}_split1/  (실 E1 미간섭)
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

import e_common as A          # noqa: E402
import e1_cross as E          # noqa: E402  (cross_val_fold, DIRS)

EPOCHS = 50
SMOKE = A.A1_OUT / "E1_smoke"
ROI = ["RT", "LT", "RB", "LB"]
MD = {"fwd": -0.231, "rev": +0.004, "d_overall": +0.118,
      "d": {"RT": .043, "LT": .138, "RB": .198, "LB": .091}}


def arm_bias(run_dir):
    p = run_dir / "results.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    return {"overall": r["bias"], **{roi: r["per_roi"][roi]["bias"] for roi in ROI}}


def summarize():
    bks = sorted(d.name.replace("dorga_", "") for d in SMOKE.glob("dorga_*") if d.is_dir())
    if not bks:
        print("[대기] 먼저 학습: python temp.py --mode fwd --bk <name> [--mrm PATH] / --mode rev ...")
        return
    print("\n" + "=" * 78)
    print("백본별 방향편향·δ_obs  · md: fwd −0.231 rev +0.004 δ +0.118  (우리 세팅 고정)")
    print("=" * 78)
    print(f"  {'backbone':<10}{'fwd':>9}{'rev':>9}{'δ_obs':>9}   {'RB δ':>8}{'(md .198)':>10}")
    for bk in bks:
        root = SMOKE / f"dorga_{bk}"
        f = arm_bias(root / "24to26_split1")
        r = arm_bias(root / "26to24_split1")
        if not (f and r):
            print(f"  {bk:<10}  (fwd·rev 둘 다 필요)")
            continue
        do = (r["overall"] - f["overall"]) / 2
        drb = (r["RB"] - f["RB"]) / 2
        print(f"  {bk:<10}{f['overall']:>+9.3f}{r['overall']:>+9.3f}{do:>+9.3f}   {drb:>+8.3f}")
    print("\n  → rev 이 +0.004, δ_obs 가 +0.118, RB δ 가 +0.198 에 붙는 백본 = md. 그걸로 core.MRM_W 확정.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fwd", "rev", "summary"])
    ap.add_argument("--bk", default="brixia", help="백본 라벨(출력 폴더 dorga_<bk>)")
    ap.add_argument("--mrm", default=None, help="백본 체크포인트 경로(미지정 시 core.MRM_W)")
    args = ap.parse_args()

    if args.mode == "summary":
        summarize()
        return

    if not Path(A.B.CSV_PATH).exists():
        print(f"[SKIP] 데이터 없음: {A.B.CSV_PATH} — 서버에서 실행")
        return

    if args.mrm:                                       # 백본만 교체(나머지 세팅 고정)
        A.B.MRM_W = Path(args.mrm)
    ty, ey, tag = E.DIRS[args.mode]
    arm = f"{tag}_split1"
    root = SMOKE / f"dorga_{args.bk}"
    A.register(arm, E.cross_val_fold(ty, ey, 0))
    print("\n" + "#" * 78)
    print(f"# {args.mode}  bk={args.bk}  MRM={A.B.MRM_W}")
    print(f"#   {arm}: train {ty} 80% / val 20% / test {ey} 전체 · 50ep · early-stop 10 · best_val")
    print("#" * 78)
    A.B.run_one_dorga(arm, 1, EPOCHS, root, arm=arm)   # 우리 세팅(core 기본: early-stop 10, best_val)

    b = arm_bias(root / arm)
    print(f"\n[{args.mode}/{args.bk}] overall {b['overall']:+.3f}  "
          f"RB {b['RB']:+.3f}  LT {b['LT']:+.3f}  (md {args.mode} {MD[args.mode]:+.3f})")
    print("  → 반대 방향도 돌린 뒤:  python temp.py --mode summary")


if __name__ == "__main__":
    main()
