# -*- coding: utf-8 -*-
r"""
temp.py — 병합/재배선 서버 검증 하네스

목적: npjDM2026 → Private 병합 후 Model(순수 모델) / Experiment(core·드라이버) 배치가
실제 서버 데이터(labels.csv·images_normalize·masks·MRM)로 끝까지 도는지 단계별 확인.

실행:
    cd <repo 루트>        # 예: /shared/home/mai/JeongGeon/Private
    python temp.py

동작:
  STEP1  학습·집계 모듈 12개 import (offline 가능)
  STEP2  미포팅 추출기 3개 구문검사(ast) — import 시 최상단 실행이라 syntax만
  STEP3  데이터·가중치 경로 존재 확인 (core 상수 기준)
  STEP4  실데이터 스모크: split → dataset 1샘플 → DORGA/BSNet/PAFE forward
         (데이터/GPU 없으면 자동 skip → offline 에서도 STEP1·2 는 통과)

종료코드: FAIL 0건이면 0, 있으면 1.
"""
import ast
import importlib.util
import os
import sys
import traceback
from pathlib import Path

# Windows cp949 콘솔에서 유니코드(— · 한글) 출력 크래시 방지
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parent
EXP = REPO / "Experiment"
# core / e_common / Model 해석용 경로 (각 모듈도 자체 shim 이 있으나 진입점에서 보강)
for _p in (str(EXP), str(EXP / "E"), str(REPO / "Model"), str(REPO / "Model" / "DORGA")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PASS, FAIL, SKIP = [], [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name); print(f"  [PASS] {name}")
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")


def _import(relpath, modname):
    spec = importlib.util.spec_from_file_location(modname, str(REPO / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────
print("=" * 72)
print("STEP 1 — 학습·집계 모듈 import (12)")
print("=" * 72)
CORE = {}
def _imp_core():
    CORE["core"] = _import("Experiment/core.py", "core")
check("core.py", _imp_core)
for rel, name in [
    ("Experiment/E/e_common.py", "e_common"),
    ("Experiment/E/e0_cross.py", "e0_cross"),
    ("Experiment/E/e1_indomain_kfold.py", "e1_indomain_kfold"),
    ("Experiment/E/e2_matched_indomain.py", "e2_matched_indomain"),
    ("Experiment/E/e3_positive_control.py", "e3_positive_control"),
    ("Experiment/E/e4_negative_control.py", "e4_negative_control"),
    ("Experiment/E/summary.py", "e_summary"),
    ("Experiment/A/run_all.py", "run_all"),
    ("Experiment/A/summary.py", "a_summary"),
    ("Experiment/L/summary.py", "l_summary"),
]:
    check(name, lambda rel=rel, name=name: _import(rel, name))

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("STEP 2 — 미구현/미포팅 L·A 코드 구문검사 (서버 데이터로 구현 예정)")
print("=" * 72)
for rel in ["Experiment/A/recompute_bestval.py",
            "Experiment/L/l3_features.py",
            "Experiment/L/l3_features_matched.py",
            "Experiment/L/l1_structure.py",
            "Experiment/L/l4_reproducibility.py"]:
    check(f"syntax {Path(rel).name}",
          lambda rel=rel: ast.parse((REPO / rel).read_text(encoding="utf-8")))

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("STEP 3 — 데이터·가중치 경로 존재 확인")
print("=" * 72)
core = CORE.get("core")
if core is None:
    print("  [SKIP] core import 실패로 STEP 3·4 생략")
    SKIP += ["STEP3", "STEP4"]
else:
    paths = {"labels.csv": Path(core.CSV_PATH), "images_normalize": core.IMG_DIR,
             "masks": core.MASK_DIR, "MRM 백본": core.MRM_W}
    for label, p in paths.items():
        exists = Path(p).exists()
        (PASS if exists else SKIP).append(f"경로 {label}")
        print(f"  [{'PASS' if exists else 'SKIP'}] {label}: {p}")
    data_ok = Path(core.CSV_PATH).exists() and core.IMG_DIR.exists() and core.MASK_DIR.exists()

    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("STEP 4 — 실데이터 스모크 (split → dataset → forward)")
    print("=" * 72)
    if not data_ok:
        print("  [SKIP] 데이터 경로 없음 → offline. 서버에서 재실행 요망.")
        SKIP.append("STEP4")
    else:
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader

        state = {}

        def s_split():
            df = pd.read_csv(core.CSV_PATH)
            for mode in ("2024to2026", "2026to2024"):
                d, ty, ey = core.make_split(df.copy(), mode, seed=1)
                n = {s: int((d.split == s).sum()) for s in ("train", "val", "test")}
                assert n["train"] > 0 and n["test"] > 0, f"{mode} split 비었음: {n}"
                print(f"        {mode}: {n} (train={ty}→test={ey})")
            state["df"] = core.gen_patterns(core.make_split(df, "2024to2026", 1)[0], 1)
        check("split(양방향) 비어있지 않음", s_split)

        def s_dorga_ds():
            d = state["df"]
            ds = core.DorgaMaskDataset(d[d.split == "test"])
            x, y, rel, roi_masks, pat, idx = ds[0]
            assert x.shape == (1, core.IMG_SIZE, core.IMG_SIZE), x.shape
            assert rel.shape == (core.R, 4) and roi_masks.shape[0] == core.R
            print(f"        DorgaMaskDataset[0]: img{tuple(x.shape)} rel{tuple(rel.shape)} "
                  f"roi_masks{tuple(roi_masks.shape)} y={y.tolist()}")
            state["dorga_ds"] = ds
        check("DorgaMaskDataset 1샘플 로드", s_dorga_ds)

        def s_dorga_fwd():
            d = state["df"]
            pi_g, pi_p = core.compute_priors(d)
            vit = core.load_mrm_vit(core.MRM_W, num_classes=core.C, in_chans=1)
            model = core.BrixiaViT512Dynamic(
                vit, num_regions=core.R, num_classes=core.C, num_patterns=core.K,
                proj_dim=core.PROJ_DIM, pi_global=pi_g, pi_patterns=pi_p,
                gnn_num_heads=4, gnn_dropout=0.1).to(core.DEVICE).eval()
            ld = DataLoader(state["dorga_ds"], batch_size=2, num_workers=0)
            imgs, lab, rel, roi_masks, pat, _ = next(iter(ld))
            with torch.no_grad():
                out = model(imgs.to(core.DEVICE), rel.to(core.DEVICE),
                            masks=roi_masks.to(core.DEVICE))
            assert out["logits_s3"].shape[-2:] == (core.R, core.C), out["logits_s3"].shape
            print(f"        DORGA forward: logits_s3 {tuple(out['logits_s3'].shape)} (=[B,R,C])")
        check("DORGA 모델 forward", s_dorga_fwd)

        def s_scorer_fwd():
            d = state["df"]
            ds = core.ScorerDataset(d[d.split == "test"])
            ld = DataLoader(ds, batch_size=2, num_workers=0)
            img, m, y = next(iter(ld))
            for nm in ("bsnet", "pafe"):
                sc = core.load_private_scorer(nm)(classes=core.C).to(core.DEVICE).eval()
                with torch.no_grad():
                    out = sc(img.to(core.DEVICE), mask=m.to(core.DEVICE))
                assert out["logits"].shape[-2:] == (core.R, core.C), (nm, out["logits"].shape)
                print(f"        {nm} forward: logits {tuple(out['logits'].shape)} (=[B,R,C])")
        check("BSNet/PAFE 스코어러 forward", s_scorer_fwd)

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"결과: PASS {len(PASS)} · FAIL {len(FAIL)} · SKIP {len(SKIP)}")
if FAIL:
    print("실패 항목:")
    for name, err in FAIL:
        print(f"  - {name}: {err}")
if SKIP:
    print(f"건너뜀(데이터/GPU 부재): {', '.join(SKIP)}")
print("=" * 72)
sys.exit(1 if FAIL else 0)
