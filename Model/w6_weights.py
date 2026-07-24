# -*- coding: utf-8 -*-
r"""
W6 — 방향별 3-seed 가중치 저장 (DORGA · BSNet · PAFE, 자립 실행판)

각 모델을 2024→2026(fwd) / 2026→2024(rev) × seed 1·2·42 로 학습하고, best_val
체크포인트를 fwd1..3 / rev1..3 으로 모은다. 영역 순서 전 모델 [RT, LT, RB, LB].

  fwd1 = 2024→2026 s1   fwd2 = 2024→2026 s2   fwd3 = 2024→2026 s42
  rev1 = 2026→2024 s1   rev2 = 2026→2024 s2   rev3 = 2026→2024 s42

저장 위치: <OUT_ROOT>/weights/<model>/{fwd,rev}{1,2,3}.pth  (model = dorga|bsnet|pafe)

데이터 (CXR/Merged, build_unified_dataset.py 산출):
  · labels.csv  : uid, patient_id, RT, LT, RB, LB, …  (patient_id 접두어 24_/26_ = 연도)
  · images/<uid>.png : 정렬 완료 이미지 (연도 무관 한 폴더). seg+STN 이미 적용됨 → 재정렬 없음.
  · masks/<uid>.png  : 정렬 마스크 (DORGA rel/roi · BSNet 하드어텐션용).
  cross-domain split = patient_id 접두어(24_/26_)로 학습/테스트 연도를 가른다.

모델별 학습:
  · DORGA — Private DORGA(Model/DORGA/DORGA.py) + MRM.pth 백본, 5-손실·prior
  · BSNet — Private scorer(ResNet18, 하드어텐션), BrixiaLoss
  · PAFE  — Private scorer(ResNet34+ViT, 3ch 내부복제), CE

완전 자립: 외부 dorga 패키지(C:\Code\DORGA)·npjDM 하네스 의존 없음. Private repo + timm.

경로는 아래 CONFIG 상수만 서버 실경로로 맞추면 됨.
실행:
  python w6_weights.py                     # 3모델 × 6 arm
  python w6_weights.py --models bsnet pafe  # 일부만
  python w6_weights.py --skip-existing      # 중단 후 이어서
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ── DORGA 모델·손실: Private repo 자립 코드 ──
_HERE = os.path.dirname(os.path.abspath(__file__))          # Model
_MODEL_ROOT = _HERE                                         # Model (BSNet/PAFE scorer 경로 기준)
_DORGA_DIR = os.path.join(_HERE, "DORGA")                   # Model/DORGA (DORGA.py·scorer.py)
for _p in (_MODEL_ROOT, _DORGA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from DORGA import BrixiaViT512Dynamic, vit_base_patch16_512          # noqa: E402  Model/DORGA/DORGA.py
from scorer import (                                                # noqa: E402  Model/DORGA/scorer.py 손실
    compute_alpha_oracle, kl_attention_loss,
    loss_function, loss_function_projection, pattern_loss,
)

# ═══════════════ 경로 — 서버 기준. 여기만 맞게 수정 ═══════════════
BASE      = "/shared/home/mai/JeongGeon/Private"
MERGED    = Path(f"{BASE}/CXR/Merged")               # labels.csv image_path 의 기준 폴더
IMG_DIR   = Path(f"{BASE}/CXR/Merged/images_normalize")   # ★ seg+STN 정렬 완료 이미지 <uid>.png
MASK_DIR  = Path(f"{BASE}/CXR/Merged/masks")              # ★ 정렬 마스크 <uid>.png
CSV_PATH  = f"{BASE}/CXR/Merged/labels.csv"          # uid, patient_id, RT, LT, RB, LB, image_path, …
MRM_W     = Path("/shared/home/mai/JeongGeon/IEEETMI/weight/DORGA_Brixia.pth")   # DORGA 백본
OUT_ROOT  = Path(f"{BASE}/w6_out")                   # 산출물

# ── 하이퍼파라미터 ──
ROI          = ["RT", "LT", "RB", "LB"]              # labels.csv 컬럼 = 영역 순서
PATIENT_COL  = "patient_id"
UID_COL      = "uid"
EXT          = ".png"
IMG_SIZE     = 512
R, C, K      = 4, 5, 7
PROJ_DIM     = 768
BATCH_SIZE   = 32
FREEZE_BLOCKS = 6         
VAL_FRAC     = 0.10          # train:val = 9:1
TAIL_EPOCHS  = 5
EARLYSTOP_PATIENCE = 10
NORM_MEAN, NORM_STD = [0.56], [0.17]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 폐 분할 실패 시 폴백 박스 [RT, LT, RB, LB]
REL_BOXES = torch.tensor([
    [0.0, 0.0, 0.5, 0.5],   # RT
    [0.0, 0.5, 0.5, 1.0],   # LT
    [0.5, 0.0, 1.0, 0.5],   # RB
    [0.5, 0.5, 1.0, 1.0],   # LB
], dtype=torch.float32)


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════
# DORGA 백본 로더 (timm ViT + 체크포인트 vit. 키, 인라인)
# ═══════════════════════════════════════════════════════════
def load_mrm_vit(mrm_path, num_classes=C, in_chans=1):
    vit = vit_base_patch16_512(num_classes=num_classes, in_chans=in_chans,
                               drop_path_rate=0.1, global_pool="avg")
    ck = torch.load(str(mrm_path), map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    vit_sd = {k.replace("vit.", ""): v for k, v in sd.items() if k.startswith("vit.")}
    if not vit_sd:
        vit_sd = dict(sd)
    vit_sd.pop("head.weight", None); vit_sd.pop("head.bias", None)
    m, u = vit.load_state_dict(vit_sd, strict=False)
    print(f"[backbone] MRM({Path(mrm_path).name}) loaded: missing={len(m)} unexpected={len(u)}")
    return vit


def _set_trainable(model, freeze_blocks=FREEZE_BLOCKS):
    for p in model.parameters():
        p.requires_grad = True
    for blk in model.vit.blocks[:freeze_blocks]:
        for p in blk.parameters():
            p.requires_grad = False


def _make_optimizer(model, enc_lr=1e-5, head_lr=1e-4):
    """encoder(vit.) 와 헤드 분리 lr (백본 1e-5 / 헤드 1e-4)."""
    enc = [p for n, p in model.named_parameters() if n.startswith("vit.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("vit.") and p.requires_grad]
    groups = []
    if enc:
        groups.append({"params": enc, "lr": enc_lr, "weight_decay": 5e-5})
    if head:
        groups.append({"params": head, "lr": head_lr, "weight_decay": 5e-4})
    return torch.optim.AdamW(groups)


# ═══════════════════════════════════════════════════════════
# cross-domain split — patient_id 접두어(24_/26_)로 연도 구분
# ═══════════════════════════════════════════════════════════
def _year_mask(df, yr):
    return df[PATIENT_COL].astype(str).str.startswith(f"{yr % 100}_")   # 2024->"24_"


def make_split(df, mode, seed):
    if mode == "2024to2026":
        train_year, test_year = 2024, 2026
    elif mode == "2026to2024":
        train_year, test_year = 2026, 2024
    else:
        raise ValueError(f"지원하지 않는 mode: {mode}")

    tr_year, te_year = _year_mask(df, train_year), _year_mask(df, test_year)
    df = df.copy()
    df["split"] = None
    df.loc[te_year, "split"] = "test"
    rng = np.random.default_rng(seed)
    pool = np.asarray(df.loc[tr_year, PATIENT_COL].unique(), dtype=object)
    rng.shuffle(pool)
    n_val = max(2, int(round(len(pool) * VAL_FRAC)))
    val_pat = set(pool[:n_val])
    df.loc[tr_year & df[PATIENT_COL].isin(val_pat), "split"] = "val"
    df.loc[tr_year & ~df[PATIENT_COL].isin(val_pat), "split"] = "train"
    df = df[df.split.notna()].copy()
    return df, train_year, test_year


# ═══════════════════════════════════════════════════════════
# patterns / priors (DORGA, train split 만 fit)
# ═══════════════════════════════════════════════════════════
def gen_patterns(df, seed=42):
    Xtr = df.loc[df.split == "train", ROI].to_numpy("float32")
    km = KMeans(n_clusters=K, random_state=seed, n_init=20).fit(Xtr)
    sev = sorted(range(K), key=lambda k: Xtr[km.labels_ == k].sum(1).mean()
                 if (km.labels_ == k).any() else 1e9)
    remap = {old: new for new, old in enumerate(sev)}
    df[f"pattern{K}"] = [remap[c] for c in km.predict(df[ROI].to_numpy("float32"))]
    return df


def compute_priors(df):
    EPS = 1e-7
    tv = df[df.split == "train"]
    labels = tv[ROI].to_numpy(np.int32); pat = tv[f"pattern{K}"].to_numpy(np.int32)
    pi_g = np.zeros((R, R, C, C), np.float32)
    for i in range(R):
        for j in range(R):
            for d in range(C):
                md = labels[:, j] == d; nd = md.sum()
                cnt = np.bincount(labels[md, i], minlength=C).astype(np.float32) if nd else np.zeros(C)
                pi_g[i, j, :, d] = (cnt + EPS) / (nd + EPS * C)
    pi_p = np.zeros((K, R, R, C, C), np.float32)
    for k in range(K):
        lk = labels[pat == k]
        if len(lk) < 10:
            pi_p[k] = pi_g; continue
        for i in range(R):
            for j in range(R):
                for d in range(C):
                    md = lk[:, j] == d; nd = md.sum()
                    if nd == 0:
                        pi_p[k, i, j, :, d] = pi_g[i, j, :, d]
                    else:
                        cnt = np.bincount(lk[md, i], minlength=C).astype(np.float32)
                        pi_p[k, i, j, :, d] = (cnt + EPS) / (nd + EPS * C)
    return torch.from_numpy(pi_g), torch.from_numpy(pi_p)


# ═══════════════════════════════════════════════════════════
# 폐 마스크 4-ROI 분할 — 반환 [RT, LT, RB, LB]
# ═══════════════════════════════════════════════════════════
def split_lungs_to_four(mask_bin, min_area=1000):
    comps = [p for p in regionprops(label((mask_bin > 0).astype(np.uint8))) if p.area >= min_area]
    if len(comps) < 2:
        return None
    R_lung = min(comps, key=lambda p: p.centroid[1])   # 영상 좌측 = 환자 우폐
    L_lung = max(comps, key=lambda p: p.centroid[1])
    H, W = mask_bin.shape

    def halves(reg):
        y0, x0, y1, x1 = reg.bbox
        hs = np.linspace(y0, y1, 3)
        top = (int(round(hs[0])) / H, x0 / W, int(round(hs[1])) / H, x1 / W)
        bot = (int(round(hs[1])) / H, x0 / W, int(round(hs[2])) / H, x1 / W)
        return top, bot

    R_top, R_bot = halves(R_lung)
    L_top, L_bot = halves(L_lung)
    return [R_top, L_top, R_bot, L_bot]                # [RT, LT, RB, LB]


# ═══════════════════════════════════════════════════════════
# 공용 로더 — 정렬 이미지 <uid>.png + 마스크 <uid>.png
# ═══════════════════════════════════════════════════════════
_PHOTO = transforms.Compose([transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])
_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG")


def _first_existing(cands):
    for c in cands:
        if c is not None and Path(c).exists():
            return Path(c)
    return None


def _resolve_img(uid, image_path=None):
    """정렬 이미지(images_normalize)의 <uid>.<확장자>를 우선. image_path 컬럼은
    정렬 안 된 images/ 를 가리키므로 마지막 폴백으로만 둔다."""
    cands = [IMG_DIR / f"{uid}{e}" for e in _EXTS]
    if isinstance(image_path, str) and image_path:
        cands += [MERGED / image_path, Path(image_path)]
    p = _first_existing(cands)
    if p is None:
        raise FileNotFoundError(
            f"이미지 못 찾음 uid={uid}. 시도: {[str(c) for c in cands[:3]]} …")
    return p


def _resolve_mask(uid):
    p = _first_existing([MASK_DIR / f"{uid}{e}" for e in _EXTS])
    if p is None:
        raise FileNotFoundError(f"마스크 못 찾음 uid={uid} in {MASK_DIR}")
    return p


def _load_img(uid, image_path=None):
    im = Image.open(_resolve_img(uid, image_path)).convert("L")
    if im.size != (IMG_SIZE, IMG_SIZE):
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return im


def _load_mask_np(uid):
    m = Image.open(_resolve_mask(uid)).convert("L")
    if m.size != (IMG_SIZE, IMG_SIZE):
        m = m.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return (np.asarray(m, dtype=np.float32) > 0).astype(np.float32)


class DorgaMaskDataset(Dataset):
    """DORGA용: (img, y[4], rel, roi_masks, pat, idx). 마스크에서 rel/roi 생성."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.pat_col = f"pattern{K}"
        self.has_ip = "image_path" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row[UID_COL])
        x = _PHOTO(_load_img(uid, row["image_path"] if self.has_ip else None))
        if torch.isnan(x).any():
            x = torch.zeros_like(x)
        mask_np = _load_mask_np(uid)
        coords = split_lungs_to_four(mask_np) or REL_BOXES.tolist()
        rel = torch.tensor(coords, dtype=torch.float32)
        roi_masks = np.zeros((R, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for i in range(R):
            y0, x0, y1, x1 = coords[i]
            y0, y1 = int(y0 * IMG_SIZE), int(y1 * IMG_SIZE)
            x0, x1 = int(x0 * IMG_SIZE), int(x1 * IMG_SIZE)
            roi_masks[i, y0:y1, x0:x1] = mask_np[y0:y1, x0:x1]
        y = torch.tensor([int(row[c]) for c in ROI], dtype=torch.long)   # [RT,LT,RB,LB]
        pat = torch.tensor(int(row[self.pat_col]), dtype=torch.long)
        return x, y, rel, torch.from_numpy(roi_masks), pat, idx


class ScorerDataset(Dataset):
    """BSNet/PAFE용: (img, lung_mask[1,H,W], y[4])."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.has_ip = "image_path" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row[UID_COL])
        x = _PHOTO(_load_img(uid, row["image_path"] if self.has_ip else None))
        if torch.isnan(x).any():
            x = torch.zeros_like(x)
        m = torch.from_numpy(_load_mask_np(uid))[None]                   # (1,H,W)
        y = torch.tensor([int(row[c]) for c in ROI], dtype=torch.long)   # [RT,LT,RB,LB]
        return x, m, y


# ═══════════════════════════════════════════════════════════
# Eval / 통계
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_dorga(model, loader):
    model.eval(); P, Y = [], []
    for imgs, lab, rel, roi_masks, pat, _ in loader:
        imgs, rel, roi_masks = imgs.to(DEVICE), rel.to(DEVICE), roi_masks.to(DEVICE)
        out = model(imgs, rel, masks=roi_masks)
        P.append(out["logits_s3"].argmax(-1).cpu()); Y.append(lab)
    return _metrics(torch.cat(P).numpy(), torch.cat(Y).numpy())


@torch.no_grad()
def evaluate_scorer(scorer, loader):
    scorer.eval(); P, Y = [], []
    for img, mask, y in loader:
        out = scorer(img.to(DEVICE), mask=mask.to(DEVICE))
        P.append(out["logits"].argmax(-1).cpu()); Y.append(y)
    return _metrics(torch.cat(P).numpy(), torch.cat(Y).numpy())


def _metrics(P, Y):
    acc = (P == Y).mean(); mae = np.abs(P - Y).mean(); bias = (P - Y).mean()
    per = {ROI[i]: (float((P[:, i] == Y[:, i]).mean()), float(np.abs(P[:, i] - Y[:, i]).mean()),
                    float((P[:, i] - Y[:, i]).mean())) for i in range(R)}
    return acc, mae, bias, per, P, Y


def patient_bootstrap_ci(P, Y, patients, n_boot=5000, seed=0):
    d = (P - Y).mean(axis=1)
    pats = np.asarray(patients); uniq = np.unique(pats)
    per_pat = np.array([d[pats == u].mean() for u in uniq])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = per_pat[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(bias=float(per_pat.mean()), ci=(float(lo), float(hi)), n_pat=len(uniq))


# ═══════════════════════════════════════════════════════════
# DORGA arm 학습
# ═══════════════════════════════════════════════════════════
def run_one_dorga(mode, seed, epochs, root, eval_test_every=1):
    set_seed(seed)
    df = pd.read_csv(CSV_PATH)
    df, TRAIN_Y, TEST_Y = make_split(df, mode, seed)
    df = gen_patterns(df, seed)
    pi_g, pi_p = compute_priors(df)

    print("\n" + "━" * 78)
    print(f"▶ dorga arm: mode={mode} seed={seed} (train={TRAIN_Y} → test={TEST_Y})")
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        print(f"  [{s:<5}] {len(sub):>4}장 / {sub[PATIENT_COL].nunique():>3}명  "
              f"평균등급 {sub[ROI].to_numpy().mean():.4f}")
    print("━" * 78)

    tv = df[df.split == "train"]
    roi_class = torch.zeros(R, C)
    for r, col in enumerate(ROI):
        for c in range(C):
            roi_class[r, c] = (tv[col] == c).sum()
    roi_class = roi_class.to(DEVICE)

    mk = lambda s, sh, dl: DataLoader(DorgaMaskDataset(df[df.split == s]), batch_size=BATCH_SIZE,
                                      shuffle=sh, drop_last=dl, num_workers=0, pin_memory=True)
    train_loader, val_loader, test_loader = mk("train", True, True), mk("val", False, False), mk("test", False, False)
    test_patients = df[df.split == "test"][PATIENT_COL].to_numpy()

    vit = load_mrm_vit(MRM_W, num_classes=C, in_chans=1)
    model = BrixiaViT512Dynamic(vit, num_regions=R, num_classes=C, num_patterns=K,
                                proj_dim=PROJ_DIM, pi_global=pi_g, pi_patterns=pi_p,
                                gnn_num_heads=4, gnn_dropout=0.1).to(DEVICE)
    _set_trainable(model, freeze_blocks=FREEZE_BLOCKS)
    optimizer = _make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    run_dir = root / f"{mode}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val, best_epoch, hist, since = float("inf"), -1, [], 0
    for epoch in range(1, epochs + 1):
        model.train(); tr_P, tr_Y = [], []
        for imgs, lab, rel, roi_masks, pat_gt, _ in tqdm(train_loader, desc=f"  [{epoch:03d}] train", leave=False):
            imgs, lab, rel = imgs.to(DEVICE), lab.to(DEVICE), rel.to(DEVICE)
            roi_masks, pat_gt = roi_masks.to(DEVICE), pat_gt.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out = model(imgs, rel, masks=roi_masks, labels=lab)
            loss_s1 = pattern_loss(out["rho"], pat_gt)
            loss_s2_ce = loss_function(out["logits_s2"], lab, roi_class, "RoiClass")
            loss_s2_proj, _ = loss_function_projection(out["u"], lab, model.class_anchors, roi_class)
            a = F.normalize(model.class_anchors, dim=-1)
            w_vec = F.normalize(a[model.C - 1] - a[0], dim=-1)
            v_w_cos = (out["v"] * w_vec).sum(-1).clamp(-1.0, 1.0)
            loss_v_orth = (torch.acos(v_w_cos) - math.pi / 2).abs().mean()
            loss_s2 = loss_s2_ce + loss_s2_proj + loss_v_orth
            loss_s3 = loss_function(out["logits_s3"], lab, roi_class, "RoiClass")
            alpha_pred = out["gnn_out"]["attached"]["alpha"][-1]
            rho_gt = F.one_hot(pat_gt.long(), num_classes=model.K).float()
            pi_oracle = model.dynamic_prior(rho_gt)
            alpha_t, conf = compute_alpha_oracle(pi_gt=pi_oracle.detach(), labels=lab,
                                                 p_model_logits=out["logits_s2"].detach(), oracle_tau=0.1)
            loss_att = kl_attention_loss(alpha_pred, alpha_t, conf)
            total = loss_s1 + loss_s2 + loss_s3 + loss_att
            total.backward(); optimizer.step()
            tr_P.append(out["logits_s3"].detach().argmax(-1).cpu()); tr_Y.append(lab.detach().cpu())
        scheduler.step()

        tr_P, tr_Y = torch.cat(tr_P).numpy(), torch.cat(tr_Y).numpy()
        tr_acc, tr_bias = (tr_P == tr_Y).mean(), (tr_P - tr_Y).mean()
        v_acc, v_mae, v_bias, _, _, _ = evaluate_dorga(model, val_loader)
        tag = ""
        if v_mae < best_val:
            best_val, best_epoch, since = v_mae, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_val_mae": float(best_val)},
                       run_dir / "best_val_model.pth"); tag = "  << BEST(val)"
        else:
            since += 1

        do_test = (epoch % eval_test_every == 0) or (epoch > epochs - TAIL_EPOCHS) or (epoch == epochs)
        if do_test:
            t_acc, t_mae, t_bias, t_per, _, _ = evaluate_dorga(model, test_loader)
            hist.append(dict(epoch=epoch, tr_bias=float(tr_bias), val_bias=float(v_bias),
                             test_bias=float(t_bias), test_acc=float(t_acc), test_mae=float(t_mae)))
            roi_bias = "  ".join(f"{r} {t_per[r][2]:+.3f}" for r in ROI)
            print(f"  [{epoch:03d}/{epochs}] train ACC {tr_acc:.4f} bias {tr_bias:+.4f} | "
                  f"val ACC {v_acc:.4f} MAE {v_mae:.4f} bias {v_bias:+.4f} | "
                  f"TEST({TEST_Y}) ACC {t_acc:.4f} MAE {t_mae:.4f} bias {t_bias:+.4f} | [{roi_bias}]{tag}")
        else:
            hist.append(dict(epoch=epoch, tr_bias=float(tr_bias), val_bias=float(v_bias)))
            print(f"  [{epoch:03d}/{epochs}] train ACC {tr_acc:.4f} bias {tr_bias:+.4f} | "
                  f"val ACC {v_acc:.4f} MAE {v_mae:.4f} bias {v_bias:+.4f}{tag}")

        if EARLYSTOP_PATIENCE and since >= EARLYSTOP_PATIENCE:
            print(f"  ⏹ early stop @epoch {epoch} (best={best_epoch})")
            break

    acc, mae, bias, per, P, Y = evaluate_dorga(model, test_loader)
    _write_results(run_dir, "dorga", mode, seed, TRAIN_Y, TEST_Y, df, epochs, best_epoch,
                   best_val, acc, mae, bias, per, hist, P, Y, test_patients, seed)
    del model, vit, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# BSNet / PAFE arm 학습 (Private ScorerBase)
# ═══════════════════════════════════════════════════════════
def load_private_scorer(name):
    sub = {"bsnet": "BSNet", "pafe": "PAFE"}[name]
    path = Path(_MODEL_ROOT) / sub / "scorer.py"
    spec = importlib.util.spec_from_file_location(f"_scorer_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_scorer


def run_one_scorer(name, mode, seed, epochs, root, eval_test_every=1):
    set_seed(seed)
    df = pd.read_csv(CSV_PATH)
    df, TRAIN_Y, TEST_Y = make_split(df, mode, seed)

    scorer = load_private_scorer(name)(classes=C).to(DEVICE)
    lr, backbone_lr = scorer.default_lr, scorer.default_backbone_lr

    print("\n" + "━" * 78)
    print(f"▶ {name} arm: mode={mode} seed={seed} (train={TRAIN_Y} → test={TEST_Y}) "
          f"lr={lr} backbone_lr={backbone_lr}")
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        print(f"  [{s:<5}] {len(sub):>4}장 / {sub[PATIENT_COL].nunique():>3}명  "
              f"평균등급 {sub[ROI].to_numpy().mean():.4f}")
    print("━" * 78)

    mk = lambda s, sh, dl: DataLoader(ScorerDataset(df[df.split == s]), batch_size=BATCH_SIZE,
                                      shuffle=sh, drop_last=dl, num_workers=0, pin_memory=True)
    train_loader, val_loader, test_loader = mk("train", True, True), mk("val", False, False), mk("test", False, False)
    test_patients = df[df.split == "test"][PATIENT_COL].to_numpy()

    run_dir = root / f"{mode}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # freeze 없음 — 처음부터 백본(1e-5)+헤드(1e-4) 함께 학습
    optimizer = scorer.make_optimizer(lr, backbone_lr=backbone_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val, best_epoch, hist, since = float("inf"), -1, [], 0
    for epoch in range(1, epochs + 1):
        scorer.train(True)
        tr_P, tr_Y = [], []
        for img, mask, y in tqdm(train_loader, desc=f"  [{epoch:03d}] train", leave=False):
            img, mask, y = img.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            out = scorer(img, mask=mask, target=y)
            loss, _ = scorer.compute_loss(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr_P.append(out["logits"].detach().argmax(-1).cpu()); tr_Y.append(y.detach().cpu())
        scheduler.step()

        tr_P, tr_Y = torch.cat(tr_P).numpy(), torch.cat(tr_Y).numpy()
        tr_acc, tr_bias = (tr_P == tr_Y).mean(), (tr_P - tr_Y).mean()
        v_acc, v_mae, v_bias, _, _, _ = evaluate_scorer(scorer, val_loader)
        tag = ""
        if v_mae < best_val:
            best_val, best_epoch, since = v_mae, epoch, 0
            torch.save({"state_dict": scorer.net.state_dict(), "epoch": epoch,
                        "best_val_mae": float(best_val)}, run_dir / "best_val_model.pth"); tag = "  << BEST(val)"
        else:
            since += 1

        do_test = (epoch % eval_test_every == 0) or (epoch > epochs - TAIL_EPOCHS) or (epoch == epochs)
        if do_test:
            t_acc, t_mae, t_bias, t_per, _, _ = evaluate_scorer(scorer, test_loader)
            hist.append(dict(epoch=epoch, tr_bias=float(tr_bias), val_bias=float(v_bias),
                             test_bias=float(t_bias), test_acc=float(t_acc), test_mae=float(t_mae)))
            roi_bias = "  ".join(f"{r} {t_per[r][2]:+.3f}" for r in ROI)
            print(f"  [{epoch:03d}/{epochs}] train ACC {tr_acc:.4f} bias {tr_bias:+.4f} | "
                  f"val ACC {v_acc:.4f} MAE {v_mae:.4f} bias {v_bias:+.4f} | "
                  f"TEST({TEST_Y}) ACC {t_acc:.4f} MAE {t_mae:.4f} bias {t_bias:+.4f} | [{roi_bias}]{tag}")
        else:
            hist.append(dict(epoch=epoch, tr_bias=float(tr_bias), val_bias=float(v_bias)))
            print(f"  [{epoch:03d}/{epochs}] train ACC {tr_acc:.4f} bias {tr_bias:+.4f} | "
                  f"val ACC {v_acc:.4f} MAE {v_mae:.4f} bias {v_bias:+.4f}{tag}")

        if EARLYSTOP_PATIENCE and since >= EARLYSTOP_PATIENCE:
            print(f"  ⏹ early stop @epoch {epoch} (best={best_epoch})")
            break

    acc, mae, bias, per, P, Y = evaluate_scorer(scorer, test_loader)
    _write_results(run_dir, name, mode, seed, TRAIN_Y, TEST_Y, df, epochs, best_epoch,
                   best_val, acc, mae, bias, per, hist, P, Y, test_patients, seed)
    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_results(run_dir, name, mode, seed, TRAIN_Y, TEST_Y, df, epochs, best_epoch,
                   best_val, acc, mae, bias, per, hist, P, Y, test_patients, bseed):
    tail = [h["test_bias"] for h in hist if "test_bias" in h][-TAIL_EPOCHS:]
    bs = patient_bootstrap_ci(P, Y, test_patients, seed=bseed)
    res = dict(model=name, mode=mode, seed=seed, train_year=int(TRAIN_Y), test_year=int(TEST_Y),
               roi_order=ROI, mrm=str(MRM_W) if name == "dorga" else None,
               n_train=int((df.split == "train").sum()), n_test=int((df.split == "test").sum()),
               final_epoch=epochs, best_val_epoch=best_epoch, best_val_mae=float(best_val),
               acc=float(acc), mae=float(mae), bias=float(bias),
               tail_bias=float(np.mean(tail)) if tail else float("nan"),
               pat_bias=bs["bias"], ci_lo=bs["ci"][0], ci_hi=bs["ci"][1],
               per_roi={r: dict(acc=per[r][0], mae=per[r][1], bias=per[r][2]) for r in ROI})
    (run_dir / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    np.savez(run_dir / "test_preds.npz", preds=P, labels=Y, patients=test_patients)
    print(f"  ✔ final bias {bias:+.4f} | best_val_mae {best_val:.4f}@ep{best_epoch}")


# ═══════════════════════════════════════════════════════════
# W6 드라이버
# ═══════════════════════════════════════════════════════════
DEFAULT_SEEDS = [1, 2, 42]
DEFAULT_MODELS = ["dorga", "bsnet", "pafe"]
DIRECTIONS = [("2024to2026", "fwd"), ("2026to2024", "rev")]


def train_arm(model, mode, seed, epochs, stage_root):
    if model == "dorga":
        run_one_dorga(mode, seed, epochs, stage_root)
    else:
        run_one_scorer(model, mode, seed, epochs, stage_root)


def _load_result(run_dir: Path) -> dict:
    p = run_dir / "results.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text(encoding="utf-8"))
    return {"best_val_epoch": r.get("best_val_epoch"), "best_val_mae": r.get("best_val_mae"),
            "final_bias": r.get("bias"), "n_train": r.get("n_train"), "n_test": r.get("n_test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    ap.add_argument("--seeds", type=int, nargs=3, default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--keep-staging", action="store_true")
    ap.add_argument("--mrm", default=None, help="DORGA 백본(MRM) 경로 (기본 CONFIG)")
    args = ap.parse_args()

    global MRM_W
    if args.mrm:
        MRM_W = Path(args.mrm)

    weights_root = OUT_ROOT / "weights"
    runs_root = OUT_ROOT / "runs"
    weights_root.mkdir(parents=True, exist_ok=True)
    ckpt_file = "best_val_model.pth"          # best val-MAE 가중치 하나만 저장
    print(f"[config] models={args.models} ROI={ROI} IMG={IMG_DIR} MRM={MRM_W} OUT={OUT_ROOT}")

    manifest = {}
    for model in args.models:
        weights_dir = weights_root / model
        stage_root = runs_root / model
        weights_dir.mkdir(parents=True, exist_ok=True)
        stage_root.mkdir(parents=True, exist_ok=True)
        for mode, prefix in DIRECTIONS:
            for idx, seed in enumerate(args.seeds, start=1):
                name = f"{prefix}{idx}"
                key = f"{model}/{name}"
                dst = weights_dir / f"{name}.pth"
                run_dir = stage_root / f"{mode}_s{seed}"
                meta = {"file": f"{model}/{name}.pth", "model": model, "mode": mode,
                        "seed": seed, "direction": prefix, "ckpt": "best_val", "roi_order": ROI}
                if args.skip_existing and dst.exists():
                    print(f"[skip] {key} (이미 존재)")
                    manifest[key] = {**meta, "skipped": True, **_load_result(run_dir)}
                    continue
                print("\n" + "=" * 78)
                print(f"[train] {key}  mode={mode} seed={seed} (epochs={args.epochs}, best_val)")
                print("=" * 78)
                train_arm(model, mode, seed, args.epochs, stage_root)
                src = run_dir / ckpt_file
                if not src.exists():
                    raise FileNotFoundError(f"{src} 없음. {ckpt_file} 저장 실패.")
                shutil.copy2(src, dst)
                print(f"[save] {src.name} → weights/{model}/{dst.name}")
                manifest[key] = {**meta, "skipped": False, **_load_result(run_dir)}
                if not args.keep_staging:
                    for f in run_dir.glob("*.pth"):
                        f.unlink()

    (weights_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"완료 — weights/ 에 {len(manifest)}개 매핑  (순서 {ROI})")
    for key, m in manifest.items():
        vm = m.get("best_val_mae")
        vm = f"{vm:.4f}" if isinstance(vm, (int, float)) else "?"
        print(f"  {key}.pth  ←  {m['mode']} s{m['seed']}  (val_mae {vm}, best@ep {m.get('best_val_epoch')})")
    print(f"  manifest: {weights_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
