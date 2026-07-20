# -*- coding: utf-8 -*-
r"""
W6 — 방향별 3-seed 가중치 6개 저장 드라이버

2024→2026(forward) / 2026→2024(reverse) 를 seed 1·2·42 로 각각 학습하고,
best_val 체크포인트를 fwd1..fwd3 / rev1..rev3 으로 모은다.

  fwd1 = 2024→2026 seed 1     rev1 = 2026→2024 seed 1
  fwd2 = 2024→2026 seed 2     rev2 = 2026→2024 seed 2
  fwd3 = 2024→2026 seed 42    rev3 = 2026→2024 seed 42

학습 로직은 기존 run_one(dorga_train_2026to2024.py) 재사용 — E0 와 동일 규약
(epochs 50, early-stop@10, val MAE 로만 모델 선택). 커스텀 splitter 미등록이라
make_split 디스패처가 원래 교차 split 로 fallback 한다(= E0 arm 과 같은 분할).

★ 영역 순서 [RT, LT, RB, LB] (native [RT,RB,LT,LB] 아님)
  베이스 스크립트는 native 순서지만, w6 는 fresh 학습이므로 순서를 재배열해도
  안전하다(DORGA.py 주석 참조). 순서를 정하는 전역 3개를 monkeypatch 로 치환:
    ROI(라벨) · split_lungs_to_four(박스 기하) · REL_BOXES(폴백)
  → region index 가 일관되게 [RT,LT,RB,LB] 가 되어, 저장 가중치도 그 순서로 학습된다.
  베이스 스크립트·a1_common 은 수정하지 않으므로 E0~E4 의 native 규약은 그대로 유지.

★ DORGA 백본(MRM) 경로: --mrm 로 지정(기본 아래 MRM_DEFAULT). run_one 안에서만
  로드되므로 학습 시작 전에 patch 하면 된다.

산출:
  weights/{fwd,rev}{1,2,3}.pth   ← best_val 체크포인트 ({"state_dict",...}), [RT,LT,RB,LB]
  weights/manifest.json          ← 파일명 ↔ (mode, seed, epoch, val_mae, bias, 순서, MRM)
  runs/W6/<mode>_s<seed>/        ← 원시 산출(results/history/preds). 기본은 큰 .pth 정리.

실행:
  cd D:\npjDM2026\code
  python w6_weights.py                        # 6개 전부
  python w6_weights.py --skip-existing         # weights/ 에 이미 있으면 재학습 건너뜀
  python w6_weights.py --ckpt final            # final 체크포인트로 대신 저장
  python w6_weights.py --keep-staging          # runs/W6 의 .pth 도 남김
  python w6_weights.py --mrm /path/to/MRM.pth  # 백본 경로 지정
  python w6_weights.py --seeds 1 2 42 --epochs 50
"""
import argparse
import json
import shutil
from pathlib import Path

import a1_common as A

# 번호(1,2,3) → seed. 파일명 fwd1/rev1 등은 이 순서를 따른다.
DEFAULT_SEEDS = [1, 2, 42]
DIRECTIONS = [("2024to2026", "fwd"), ("2026to2024", "rev")]

# DORGA 백본(MRM) 기본 경로 (서버).
MRM_DEFAULT = "/shared/home/mai/JeongGeon/MICCAI2026/MRM.pth"

# native [RT,RB,LT,LB] → display [RT,LT,RB,LB] 치환. 위치 1↔2 교환이라 자기역함수.
NATIVE_ROI = ["RT", "RB", "LT", "LB"]
DISPLAY_ROI = ["RT", "LT", "RB", "LB"]
PERM = [0, 2, 1, 3]                           # display[k] = native[PERM[k]]

WEIGHTS_DIR = A.A1_OUT.parent / "weights"     # D:\npjDM2026\weights
STAGE_ROOT = A.A1_OUT / "W6"                  # D:\npjDM2026\runs\W6


def apply_patches(mrm_path):
    """순서를 [RT,LT,RB,LB] 로 바꾸고 MRM 경로를 지정하는 monkeypatch.

    베이스 스크립트의 order-carrying 전역 3개를 같은 치환 PERM 으로 맞춘다.
    run_one 하류(dataset·gen_patterns·compute_priors·roi_class·per-region net·
    출력 logits)가 전부 이 전역들을 참조하므로 region index 가 일관되게 재배열된다.
    """
    assert [NATIVE_ROI[p] for p in PERM] == DISPLAY_ROI
    if list(A.B.ROI) != NATIVE_ROI:
        raise RuntimeError(f"베이스 ROI 가 예상 native 순서가 아님: {A.B.ROI}")

    # 1) 라벨 순서
    A.B.ROI = list(DISPLAY_ROI)
    # 2) 박스 기하: split_lungs_to_four 반환을 PERM 으로 재배열
    _orig_split = A.B.split_lungs_to_four

    def _split_display(mask_bin, min_area=1000):
        out = _orig_split(mask_bin, min_area)
        return None if out is None else [out[p] for p in PERM]

    A.B.split_lungs_to_four = _split_display
    # 3) 폴백 박스
    A.B.REL_BOXES = A.B.REL_BOXES[PERM].contiguous()
    # 4) 백본(MRM) 경로
    A.B.MRM_W = Path(mrm_path)
    print(f"[patch] 영역순서 → {DISPLAY_ROI}  |  MRM → {A.B.MRM_W}")


def _load_result(run_dir: Path) -> dict:
    """run_one 이 남긴 results.json 에서 요약 지표를 뽑는다(없으면 빈 dict)."""
    p = run_dir / "results.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text(encoding="utf-8"))
    return {"best_val_epoch": r.get("best_val_epoch"),
            "best_val_mae": r.get("best_val_mae"),
            "final_bias": r.get("bias"),
            "n_train": r.get("n_train"), "n_test": r.get("n_test"),
            "train_year": r.get("train_year"), "test_year": r.get("test_year")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=3, default=DEFAULT_SEEDS,
                    help="번호 1,2,3 에 매핑될 seed 3개 (기본 1 2 42)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ckpt", choices=["best_val", "final"], default="best_val",
                    help="6개로 저장할 체크포인트 (기본 best_val)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="weights/ 에 대상 파일이 이미 있으면 그 arm 재학습 생략")
    ap.add_argument("--keep-staging", action="store_true",
                    help="runs/W6 의 학습 .pth 를 지우지 않고 보존")
    ap.add_argument("--mrm", default=MRM_DEFAULT, help="DORGA 백본(MRM) 경로")
    args = ap.parse_args()

    ckpt_file = f"{args.ckpt}_model.pth"       # best_val_model.pth / final_model.pth
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)

    apply_patches(args.mrm)                     # [RT,LT,RB,LB] 순서 + MRM 경로

    # 무거운 2026 캐시는 6개 arm 이 공유 (E0 와 동일)
    cache = None
    manifest = {}

    for mode, prefix in DIRECTIONS:
        for idx, seed in enumerate(args.seeds, start=1):
            name = f"{prefix}{idx}"            # fwd1 … rev3
            dst = WEIGHTS_DIR / f"{name}.pth"

            if args.skip_existing and dst.exists():
                print(f"[skip] {name} (이미 존재: {dst.name})")
                run_dir = STAGE_ROOT / f"{mode}_s{seed}"
                manifest[name] = {"file": dst.name, "mode": mode, "seed": seed,
                                  "direction": prefix, "ckpt": args.ckpt,
                                  "roi_order": DISPLAY_ROI, "mrm": str(A.B.MRM_W),
                                  "skipped": True, **_load_result(run_dir)}
                continue

            if cache is None:                  # 실제 학습이 필요한 첫 시점에만 빌드
                print("[cache] 2026 영상 캐시 빌드 중…")
                cache = A.build_full_2026_cache()

            print("\n" + "=" * 78)
            print(f"[train] {name}  mode={mode}  seed={seed}  "
                  f"(epochs={args.epochs}, ckpt={args.ckpt})")
            print("=" * 78)
            A.B.run_one(mode, seed, args.epochs, cache, STAGE_ROOT)

            run_dir = STAGE_ROOT / f"{mode}_s{seed}"
            src = run_dir / ckpt_file
            if not src.exists():
                raise FileNotFoundError(
                    f"{src} 없음. run_one 이 {ckpt_file} 을 저장하지 못했다.")
            shutil.copy2(src, dst)
            print(f"[save] {src.name} → weights/{dst.name}")

            manifest[name] = {"file": dst.name, "mode": mode, "seed": seed,
                              "direction": prefix, "ckpt": args.ckpt,
                              "roi_order": DISPLAY_ROI, "mrm": str(A.B.MRM_W),
                              "skipped": False, **_load_result(run_dir)}

            if not args.keep_staging:          # 스테이징 .pth 정리 (json/npz 는 보존)
                for f in run_dir.glob("*.pth"):
                    f.unlink()

    (WEIGHTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"완료 — weights/ 에 {len([m for m in manifest.values()])}개 매핑")
    for name, m in manifest.items():
        vm = m.get("best_val_mae")
        vm = f"{vm:.4f}" if isinstance(vm, (int, float)) else "?"
        print(f"  {name}.pth  ←  {m['mode']} s{m['seed']}  "
              f"(val_mae {vm}, best@ep {m.get('best_val_epoch')})")
    print(f"  manifest: {WEIGHTS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
