"""
통합 학습 스크립트 — BSNet / PAFE / DORGA 를 같은 코드로 학습.

모델별 차이(입력 채널, 마스크, forward 시그니처, 출력 형태, 손실)는 각 폴더
scorer.py 의 wrapper(common.ScorerBase 계약)가 전부 흡수한다. train.py 는
어느 모델이든 아래 두 호출만 한다. 모델별 분기 없음.

    out = scorer(img, mask=mask, target=target)     # {"logits": [B,4,C]}
    loss, parts = scorer.compute_loss(out, target, extras)

영역 순서는 전 모델 [RT, LT, RB, LB].

폴더:
  CXR/Merged/images_normalize   정렬 이미지 (512, uint8)
  CXR/Merged/masks              정렬 마스크 (needs_mask 모델용, cache_masks.py 산출)
  CXR/Merged/labels.csv         uid, patient_id, RT, LT, RB, LB (0~4)
  Model/<Sub>/scorer.py         build_scorer(classes, **kw) -> ScorerBase

노트북:
    from train import train
    train("bsnet")
    train("pafe", epochs=120, lr=5e-4)
    train("dorga", weights_path=".../best_model_native.pth")   # 파인튜닝 권장
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ───────────────────────── 경로 ─────────────────────────
BASE = "/shared/home/mai/JeongGeon/Private"
IMG_DIR = f"{BASE}/CXR/Merged/images_normalize"
MASK_DIR = f"{BASE}/CXR/Merged/masks"
CSV_PATH = f"{BASE}/CXR/Merged/labels.csv"
MODEL_ROOT = f"{BASE}/Model"
OUT_DIR = f"{BASE}/Model/checkpoints"

if MODEL_ROOT not in sys.path:
    sys.path.insert(0, MODEL_ROOT)

from common import CLASSES, SCORE_COLS, metrics          # noqa: E402

# ───────────────────────── 모델 레지스트리 ─────────────────────────
# 폴더명만 등록한다. 세 폴더의 scorer.py 는 파일명이 같아 일반 import 가
# 충돌하므로 파일 경로로 직접 로드한다.
MODELS = {"bsnet": "BSNet", "pafe": "PAFE", "dorga": "DORGA"}


def build_scorer(name, classes=CLASSES, **kw):
    sub = MODELS[name]
    path = os.path.join(MODEL_ROOT, sub, "scorer.py")
    spec = importlib.util.spec_from_file_location(f"_scorer_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_scorer(classes=classes, **kw)


# ═════════════════════ 데이터셋 (inha_dataset.py 사용) ═════════════════════
# InhaUHDataset: (img, mask, target[2,2]) 또는 (img, target[2,2]) 반환.
# target 은 [[RT,LT],[RB,LB]] -> 학습 시 (B,4) [RT,LT,RB,LB] 로 편다.
from inha_dataset import InhaUHDataset, split_indices_by_patient   # noqa: E402


def make_dataset(rows, mode, use_mask):
    return InhaUHDataset(IMG_DIR, CSV_PATH, mode=mode, rows=rows,
                         mask_dir=(MASK_DIR if use_mask else None))


class WithExtras(Dataset):
    """base 배치 끝에 per-sample extras(예: DORGA 패턴 id)를 붙인다.
    extras 는 base 의 행 순서에 맞춘 1D 정수 배열."""

    def __init__(self, base, extras):
        assert len(base) == len(extras)
        self.base = base
        self.extras = extras

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        items = self.base[idx]
        ex = torch.tensor(int(self.extras[idx]), dtype=torch.long)
        return (*items, ex)


def unpack_batch(batch, use_mask, has_extras, device):
    items = list(batch)
    extras = items.pop().to(device) if has_extras else None
    if use_mask:
        img, mask, target = items
        mask = mask.to(device)
    else:
        img, target = items
        mask = None
    img, target = img.to(device), target.to(device)
    target = target.reshape(target.size(0), -1)      # (B,2,2) -> (B,4)
    return img, mask, target, extras


# ═════════════════════ 학습 루프 (전 모델 공통) ═════════════════════
def run_epoch(scorer, loader, device, optimizer=None, frozen=False):
    train = optimizer is not None
    scorer.train(train)
    if train and frozen:
        scorer.backbone_bn_eval()

    has_extras = isinstance(loader.dataset, WithExtras)
    tot, agg = 0, {}
    for batch in loader:
        img, mask, target, extras = unpack_batch(batch, scorer.needs_mask,
                                                 has_extras, device)
        with torch.set_grad_enabled(train):
            out = scorer(img, mask=mask, target=target)
            loss, parts = scorer.compute_loss(out, target, extras)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        bs = img.size(0); tot += bs
        row = {"loss": loss.item(),
               **{k: float(v) for k, v in parts.items()},
               **metrics(out["logits"], target)}
        for k, v in row.items():
            agg[k] = agg.get(k, 0.0) + v * bs
    return {k: v / tot for k, v in agg.items()}


def _fmt(stats, keys):
    return " ".join(f"{k} {stats[k]:.3f}" for k in keys if k in stats)


def train(model_name="bsnet", epochs=80, batch=8, lr=None, backbone_lr=None,
          head_epochs=20, classes=CLASSES, workers=4, seed=0, **model_kw):
    """학습률은 전 모델 공통 고정값: 헤드 1e-4 / 백본 인코더 1e-5.
    (ScorerBase.default_lr / default_backbone_lr. lr·backbone_lr 인자로 덮어쓰기 가능)
    freeze_stage 모델은 head_epochs 동안 백본 frozen + 헤드 lr, 해제 후 백본 그룹 추가.

    model_kw 는 해당 scorer 의 build_scorer 로 그대로 전달된다.
    예: train("bsnet", vertical_overlap=0.25), train("dorga", K=7, weights_path=...)"""
    assert model_name in MODELS, f"model 은 {list(MODELS)} 중 하나"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    scorer = build_scorer(model_name, classes=classes, **model_kw).to(device)
    lr = lr if lr is not None else scorer.default_lr
    backbone_lr = (backbone_lr if backbone_lr is not None
                   else scorer.default_backbone_lr)
    if scorer.needs_mask and not os.path.isdir(MASK_DIR):
        raise FileNotFoundError(f"마스크 폴더 없음: {MASK_DIR}. 먼저 cache_masks.py 실행.")

    tr_idx, va_idx = split_indices_by_patient(CSV_PATH, val_frac=0.15, seed=seed)
    labels_all = pd.read_csv(CSV_PATH)[SCORE_COLS].to_numpy()
    extras_full = scorer.prepare_training(labels_all, tr_idx, va_idx, device)

    tr_ds = make_dataset(tr_idx, "train", scorer.needs_mask)
    va_ds = make_dataset(va_idx, "val", scorer.needs_mask)
    if extras_full is not None:
        tr_ds = WithExtras(tr_ds, extras_full[tr_idx])
        va_ds = WithExtras(va_ds, extras_full[va_idx])
    train_ld = DataLoader(tr_ds, batch, shuffle=True, num_workers=workers,
                          pin_memory=True, drop_last=True)
    val_ld = DataLoader(va_ds, batch, shuffle=False, num_workers=workers,
                        pin_memory=True)

    print(f"[model] {model_name}  in_ch={scorer.in_channels}  "
          f"mask={scorer.needs_mask}  lr={lr}  backbone_lr={backbone_lr}  "
          f"freeze_stage={scorer.freeze_stage}")

    best, optimizer, frozen = float("inf"), None, None
    for ep in range(1, epochs + 1):
        if scorer.freeze_stage:
            want_freeze = ep <= head_epochs
            if want_freeze != frozen:
                scorer.freeze_backbone(want_freeze)
                frozen = want_freeze
                # frozen 이면 백본 그룹이 자동으로 빠져 헤드 그룹만 생성됨
                optimizer = scorer.make_optimizer(lr, backbone_lr=backbone_lr)
                print(f"[stage] epoch {ep}: backbone_frozen={want_freeze}, "
                      f"head_lr={lr}, backbone_lr={backbone_lr}")
        elif optimizer is None:
            frozen = False
            optimizer = scorer.make_optimizer(lr, backbone_lr=backbone_lr)

        tr = run_epoch(scorer, train_ld, device, optimizer, frozen)
        va = run_epoch(scorer, val_ld, device, None)
        extra_keys = [k for k in tr if k not in
                      ("loss", "mae", "acc", "global_mae")]
        print(f"E{ep:03d} | train loss {tr['loss']:.3f} mae {tr['mae']:.3f}"
              f" ({_fmt(tr, extra_keys)})"
              f" | val loss {va['loss']:.3f} mae {va['mae']:.3f}"
              f" gMAE {va['global_mae']:.3f} acc {va['acc']:.3f}")

        if va["global_mae"] < best:
            best = va["global_mae"]
            path = os.path.join(OUT_DIR, f"{model_name}_best.pth")
            scorer.save_weights(path)
            print(f"  saved {os.path.basename(path)} (val global_mae {best:.3f})")

    print(f"완료 [{model_name}]. best val global_mae = {best:.3f}")
    return scorer


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="bsnet", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--backbone-lr", type=float, default=None)
    a = ap.parse_args()
    train(a.model, epochs=a.epochs, batch=a.batch, lr=a.lr, backbone_lr=a.backbone_lr)
