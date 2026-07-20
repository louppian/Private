# -*- coding: utf-8 -*-
r"""
W6 — 방향별 3-seed DORGA 가중치 6개 저장 (자립 실행판)

2024→2026(fwd) / 2026→2024(rev) 를 seed 1·2·42 로 각각 학습하고, best_val
체크포인트를 fwd1..3 / rev1..3 으로 모은다. 영역 순서 [RT, LT, RB, LB].

  fwd1 = 2024→2026 s1   fwd2 = 2024→2026 s2   fwd3 = 2024→2026 s42
  rev1 = 2026→2024 s1   rev2 = 2026→2024 s2   rev3 = 2026→2024 s42

★ 자립판: npjDM2026 하네스(a1_common · dorga_train_2026to2024.py)에 의존하지
  않는다. 학습 로직(dataset·loss·prior·split·cache·run_one)을 이 파일에 인라인했다.
  DORGA 모델/백본/seg/stn 은 dorga 레포(REPO)에서 import (여긴 유지).

★ 영역 순서 [RT, LT, RB, LB] — 원본 native [RT,RB,LT,LB] 를 데이터 레벨에서 재배열해
  박았다(ROI 라벨 · split_lungs_to_four 박스 · REL_BOXES 폴백 일관). fresh 학습이라 안전.

경로는 아래 CONFIG 상수 또는 환경변수로 지정한다(서버/로컬 이식용):
  DORGA_REPO, W6_MANIFEST, W6_IMG2024, W6_MASK2024,
  W6_SEG, W6_STN, W6_MRM, W6_OUT
MRM 은 --mrm 로도 덮어쓸 수 있다.

실행 (torch + GPU + dorga 레포):
  python w6_weights.py                        # 6개 전부
  python w6_weights.py --skip-existing         # 이미 있으면 그 arm 생략
  python w6_weights.py --ckpt final            # final 체크포인트로 저장
  python w6_weights.py --mrm /path/to/MRM.pth
  python w6_weights.py --seeds 1 2 42 --epochs 50
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
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
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════
# CONFIG — 경로/하이퍼파라미터 (환경변수 우선, 없으면 기본값)
# ═══════════════════════════════════════════════════════════
REPO = os.environ.get("DORGA_REPO", r"C:\Code\DORGA")     # dorga 레포 (모델·seg·stn)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dorga.models.backbone import load_mae_ckpt_to_512               # noqa: E402
from dorga.models.dorga_model import BrixiaViT512Dynamic            # noqa: E402
from dorga.models.pattern_prior import pattern_loss                 # noqa: E402
from dorga.losses.losses import (                                   # noqa: E402
    compute_alpha_oracle, kl_attention_loss,
    loss_function, loss_function_projection,
)
from dorga.utils.training import _make_optimizer, _set_trainable    # noqa: E402
from dorga.preprocessing.segmentation import LungSegmenter          # noqa: E402
from dorga.preprocessing.alignment import SpatialAligner            # noqa: E402

MANIFEST     = os.environ.get("W6_MANIFEST", r"D:\InhaUH_CXR\2026.05 CXRs\split_manifest.csv")
IMG2024_DIR  = Path(os.environ.get("W6_IMG2024",  r"D:\MICCAI2026\inhauh\image_normalize"))
MASK2024_DIR = Path(os.environ.get("W6_MASK2024", r"D:\MICCAI2026\inhauh\mask_normalize"))
SEG_W        = Path(os.environ.get("W6_SEG", str(Path(REPO) / "assets/weights/seg_weights.pt")))
STN_W        = Path(os.environ.get("W6_STN", str(Path(REPO) / "assets/weights/stn_weights.pth")))
MRM_W        = Path(os.environ.get("W6_MRM", "/shared/home/mai/JeongGeon/MICCAI2026/MRM.pth"))
OUT_ROOT     = Path(os.environ.get("W6_OUT", "./w6_out"))

# 영역 순서 [RT, LT, RB, LB] (display). 라벨·박스·폴백 모두 이 순서로 일관.
ROI          = ["RT", "LT", "RB", "LB"]
R, C, K      = 4, 5, 7
PROJ_DIM     = 768
BATCH_SIZE   = 32
FREEZE_BLOCKS = 6
VAL_FRAC     = 0.15
TAIL_EPOCHS  = 5
EARLYSTOP_PATIENCE = 10          # E0 규약과 동일 (val MAE 10에폭 개선 없으면 중단)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 폐 분할 실패 시 폴백 박스 — [RT, LT, RB, LB] 순서
#   RT 좌상(=환자우/상), LT 우상(=환자좌/상), RB 좌하, LB 우하
REL_BOXES = torch.tensor([
    [0.0, 0.0, 0.5, 0.5],   # RT
    [0.0, 0.5, 0.5, 1.0],   # LT
    [0.5, 0.0, 1.0, 0.5],   # RB
    [0.5, 0.5, 1.0, 1.0],   # LB
], dtype=torch.float32)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════
# 2026 in-code 전처리 (seg + STN align) — 저장 없이 RAM 캐시
# ═══════════════════════════════════════════════════════════
def raw_path_from_npz(npz_path: str) -> str:
    d = os.path.dirname(os.path.dirname(npz_path))
    fn = os.path.basename(npz_path)
    raw = re.match(r"\d+_([A-Za-z]+\d+)\.png", fn).group(1).upper() + ".jpg"
    return os.path.join(d, "RAW_IMAGE", raw)


@torch.no_grad()
def build_2026_cache(raw_paths):
    from dorga.models.stn import STN
    seg = LungSegmenter(str(SEG_W), device=str(DEVICE))
    aln = SpatialAligner(str(STN_W), device=str(DEVICE))
    cache = {}
    for p in tqdm(raw_paths, desc="2026 seg+align (in-code)"):
        arr = np.array(Image.open(p).convert("L")).astype(np.float32) / 255.0
        t = torch.from_numpy(arr)[None, None]
        im1024 = F.interpolate(t, size=(1024, 1024), mode="bilinear", align_corners=False)
        im512  = F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False).to(DEVICE)
        mask1024 = seg(im1024.to(DEVICE))
        mask512  = F.interpolate(mask1024.float(), size=(512, 512), mode="nearest").to(DEVICE)
        theta = aln.stn(mask512)
        aligned_img  = STN.transform(im512, theta).cpu()
        aligned_mask = STN.transform(mask512, theta).cpu()
        cache[p] = ((aligned_img[0, 0].numpy() * 255).clip(0, 255).astype(np.uint8),
                    ((aligned_mask[0, 0].numpy() > 0.5) * 255).astype(np.uint8))
    del seg, aln
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def build_full_2026_cache():
    df = pd.read_csv(MANIFEST)
    raw = sorted({raw_path_from_npz(p) for p in df.loc[df.year == 2026, "image_path"]})
    return build_2026_cache(raw)


# ═══════════════════════════════════════════════════════════
# 폐 마스크 기반 4-ROI 분할 — 반환 순서 [RT, LT, RB, LB]
# ═══════════════════════════════════════════════════════════
def split_lungs_to_four(mask_bin, min_area=1000):
    """폐 마스크 → 좌/우 폐 각각 상/하 반 → 4 box (정규화 y0,x0,y1,x1).
    반환 [RT, LT, RB, LB]. 영상 좌측(min centroid_x)=환자 우폐(R). 실패 시 None."""
    comps = [p for p in regionprops(label((mask_bin > 0).astype(np.uint8))) if p.area >= min_area]
    if len(comps) < 2:
        return None
    R_lung = min(comps, key=lambda p: p.centroid[1])   # 영상 좌측 = 환자 우폐
    L_lung = max(comps, key=lambda p: p.centroid[1])   # 영상 우측 = 환자 좌폐
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
# Dataset
# ═══════════════════════════════════════════════════════════
class InhaUHMaskDataset(Dataset):
    def __init__(self, df, cache2026, mode="train"):
        self.df = df.reset_index(drop=True)
        self.cache = cache2026
        self.mode = mode
        self.pat_col = f"pattern{K}"
        self.photo = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.56], [0.17])])
        self.rot = 10 if mode == "train" else 0

    def __len__(self):
        return len(self.df)

    def _load(self, row):
        if int(row["year"]) == 2024:
            stem = Path(row["image_path"]).name
            img = Image.open(IMG2024_DIR / stem).convert("L")
            mask = np.array(Image.open(MASK2024_DIR / stem).convert("L"))
        else:
            img_u8, mask_u8 = self.cache[raw_path_from_npz(row["image_path"])]
            img = Image.fromarray(img_u8); mask = mask_u8
        return img, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img, mask_np = self._load(row)
        mask = Image.fromarray(mask_np)
        if img.size != (512, 512):
            img = img.resize((512, 512), Image.BILINEAR)
        if mask.size != (512, 512):
            mask = mask.resize((512, 512), Image.NEAREST)
        if self.mode == "train" and self.rot > 0:
            ang = random.uniform(-self.rot, self.rot)
            img = TF.rotate(img, ang, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, ang, interpolation=TF.InterpolationMode.NEAREST)
        mask_np = (np.array(mask) > 0).astype(np.float32)

        coords = split_lungs_to_four(mask_np)
        if coords is None:
            coords = REL_BOXES.tolist()
        rel = torch.tensor(coords, dtype=torch.float32)
        roi_masks = np.zeros((R, 512, 512), dtype=np.float32)
        for i in range(R):
            y0, x0, y1, x1 = coords[i]
            y0, y1 = int(y0 * 512), int(y1 * 512); x0, x1 = int(x0 * 512), int(x1 * 512)
            roi_masks[i, y0:y1, x0:x1] = mask_np[y0:y1, x0:x1]
        roi_masks = torch.from_numpy(roi_masks)

        x = self.photo(img)
        if torch.isnan(x).any():
            x = torch.zeros_like(x)
        y = torch.tensor([int(row[c]) for c in ROI], dtype=torch.long)   # [RT,LT,RB,LB]
        pat = torch.tensor(int(row[self.pat_col]), dtype=torch.long)
        return x, y, rel, roi_masks, pat, idx


# ═══════════════════════════════════════════════════════════
# patterns / priors (train split 만 보고 fit — test 누수 방지)
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
# Eval / 통계
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader):
    model.eval(); P, Y = [], []
    for imgs, lab, rel, roi_masks, pat, _ in loader:
        imgs, rel, roi_masks = imgs.to(DEVICE), rel.to(DEVICE), roi_masks.to(DEVICE)
        out = model(imgs, rel, masks=roi_masks)
        P.append(out["logits_s3"].argmax(-1).cpu()); Y.append(lab)
    P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy()
    acc = (P == Y).mean(); mae = np.abs(P - Y).mean(); bias = (P - Y).mean()
    per = {ROI[i]: ((P[:, i] == Y[:, i]).mean(), np.abs(P[:, i] - Y[:, i]).mean(),
                    (P[:, i] - Y[:, i]).mean()) for i in range(R)}
    return acc, mae, bias, per, P, Y


def patient_bootstrap_ci(P, Y, patients, n_boot=5000, seed=0):
    d = (P - Y).mean(axis=1)
    pats = np.asarray(patients); uniq = np.unique(pats)
    per_pat = np.array([d[pats == u].mean() for u in uniq])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = per_pat[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    from scipy.stats import binomtest, wilcoxon
    pos = int((per_pat > 0).sum()); neg = int((per_pat < 0).sum())
    p_sign = binomtest(pos, pos + neg, 0.5).pvalue if (pos + neg) else float("nan")
    p_wil = wilcoxon(per_pat[per_pat != 0]).pvalue if (per_pat != 0).sum() else float("nan")
    return dict(bias=float(per_pat.mean()), ci=(float(lo), float(hi)),
                n_pat=len(uniq), pos=pos, neg=neg, p_sign=float(p_sign), p_wilcoxon=float(p_wil))


def stratified_bias(P, Y):
    out = {}
    for i, roi in enumerate(ROI):
        num, den = 0.0, 0
        for g in range(C):
            m = Y[:, i] == g
            if m.sum() < 8:
                continue
            b = float((P[m, i] - g).mean())
            num += b * m.sum(); den += m.sum()
        out[roi] = dict(adjusted=(num / den if den else float("nan")),
                        marginal=float((P[:, i] - Y[:, i]).mean()))
    return out


# ═══════════════════════════════════════════════════════════
# split (cross-domain: 2024to2026 / 2026to2024)
# ═══════════════════════════════════════════════════════════
def make_split(df, mode, seed):
    rng = np.random.default_rng(seed)
    if mode == "2026to2024":
        train_year, test_year = 2026, 2024
    elif mode == "2024to2026":
        train_year, test_year = 2024, 2026
    else:
        raise ValueError(f"지원하지 않는 mode: {mode}")

    df["split"] = None
    df.loc[df.year == test_year, "split"] = "test"
    # np.asarray(dtype=object) 필수 — pandas StringArray shuffle 시 뷰 시맨틱으로 누수 위험
    pool = np.asarray(df.loc[df.year == train_year, "patient"].unique(), dtype=object)
    rng.shuffle(pool)
    n_val = max(2, int(round(len(pool) * VAL_FRAC)))
    val_pat = set(pool[:n_val])
    df.loc[(df.year == train_year) & (df.patient.isin(val_pat)), "split"] = "val"
    df.loc[(df.year == train_year) & (~df.patient.isin(val_pat)), "split"] = "train"
    df = df[df.split.notna()].copy()
    return df, train_year, test_year


# ═══════════════════════════════════════════════════════════
# 한 arm 학습 (mode × seed) → run_dir 에 best_val/final 저장
# ═══════════════════════════════════════════════════════════
def run_one(mode, seed, epochs, cache2026, root, eval_test_every=1):
    set_seed(seed)
    df = pd.read_csv(MANIFEST)
    df, TRAIN_Y, TEST_Y = make_split(df, mode, seed)
    df = gen_patterns(df, seed)
    pi_g, pi_p = compute_priors(df)

    print("\n" + "━" * 78)
    print(f"▶ arm: mode={mode}  seed={seed}   (train={TRAIN_Y} → test={TEST_Y})")
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        print(f"  [{s:<5}] {len(sub):>4}장 / {sub.patient.nunique():>3}명  "
              f"평균등급 {sub[ROI].to_numpy().mean():.4f}")
    print("━" * 78)

    tv = df[df.split == "train"]
    roi_class = torch.zeros(R, C)
    for r, col in enumerate(ROI):
        for c in range(C):
            roi_class[r, c] = (tv[col] == c).sum()
    roi_class = roi_class.to(DEVICE)

    mk = lambda s, m: DataLoader(InhaUHMaskDataset(df[df.split == s], cache2026, m),
                                 batch_size=BATCH_SIZE, shuffle=(m == "train"),
                                 drop_last=(m == "train"), num_workers=0, pin_memory=True)
    train_loader, val_loader, test_loader = mk("train", "train"), mk("val", "eval"), mk("test", "eval")
    test_patients = df[df.split == "test"].patient.to_numpy()

    vit = load_mae_ckpt_to_512(str(MRM_W), num_classes=C, in_chans=1,
                               drop_path_rate=0.1, verbose=False)
    model = BrixiaViT512Dynamic(vit, num_regions=R, num_classes=C, num_patterns=K,
                                proj_dim=PROJ_DIM, pi_global=pi_g, pi_patterns=pi_p,
                                gnn_num_heads=4, gnn_dropout=0.1).to(DEVICE)
    _set_trainable(model, freeze_blocks=FREEZE_BLOCKS)
    optimizer = _make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    run_dir = root / f"{mode}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val, best_epoch, hist = float("inf"), -1, []
    since_improve = 0
    for epoch in range(1, epochs + 1):
        model.train(); tr_P, tr_Y = [], []
        for imgs, lab, rel, roi_masks, pat_gt, _ in tqdm(train_loader, desc=f"  [{epoch:03d}] train", leave=False):
            imgs, lab, rel = imgs.to(DEVICE), lab.to(DEVICE), rel.to(DEVICE)
            roi_masks, pat_gt = roi_masks.to(DEVICE), pat_gt.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                out = model(imgs, rel, masks=roi_masks, labels=lab)
                loss_s1 = pattern_loss(out["rho"], pat_gt)
                loss_s2_ce = loss_function(out["logits_s2"], lab, roi_class=roi_class, option="RoiClass")
                loss_s2_proj, _ = loss_function_projection(out["u"], lab, model.class_anchors, roi_class)
                a = F.normalize(model.class_anchors, dim=-1)
                w_vec = F.normalize(a[model.C - 1] - a[0], dim=-1)
                v_w_cos = (out["v"] * w_vec).sum(-1).clamp(-1.0, 1.0)
                loss_v_orth = (torch.acos(v_w_cos) - math.pi / 2).abs().mean()
                loss_s2 = loss_s2_ce + loss_s2_proj + loss_v_orth
                loss_s3 = loss_function(out["logits_s3"], lab, roi_class=roi_class, option="RoiClass")
                alpha_pred = out["gnn_out"]["attached"]["alpha"][-1]
                rho_gt = F.one_hot(pat_gt.long(), num_classes=model.K).float()
                pi_oracle = model.dynamic_prior(rho_gt)
                alpha_t, conf = compute_alpha_oracle(pi_gt=pi_oracle.detach(), labels=lab,
                                                     p_model_logits=out["logits_s2"].detach(), oracle_tau=0.1)
                loss_att = kl_attention_loss(alpha_pred, alpha_t, conf)
                total = loss_s1 + loss_s2 + loss_s3 + loss_att
            scaler.scale(total).backward(); scaler.step(optimizer); scaler.update()
            tr_P.append(out["logits_s3"].detach().argmax(-1).cpu()); tr_Y.append(lab.detach().cpu())
        scheduler.step()

        tr_P, tr_Y = torch.cat(tr_P).numpy(), torch.cat(tr_Y).numpy()
        tr_acc, tr_bias = (tr_P == tr_Y).mean(), (tr_P - tr_Y).mean()
        v_acc, v_mae, v_bias, _, _, _ = evaluate(model, val_loader)

        tag = ""
        if v_mae < best_val:                        # ★ val 로만 선택 — test 누수 없음
            best_val, best_epoch = v_mae, epoch
            since_improve = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_val_mae": best_val},
                       run_dir / "best_val_model.pth"); tag = "  << BEST(val)"
        else:
            since_improve += 1

        do_test = (epoch % eval_test_every == 0) or (epoch > epochs - TAIL_EPOCHS) or (epoch == epochs)
        if do_test:
            t_acc, t_mae, t_bias, t_per, _, _ = evaluate(model, test_loader)
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

        if EARLYSTOP_PATIENCE and since_improve >= EARLYSTOP_PATIENCE:
            print(f"  ⏹ early stop @epoch {epoch} (val {EARLYSTOP_PATIENCE} 에폭 개선 없음, best={best_epoch})")
            break

    torch.save({"state_dict": model.state_dict(), "epoch": epochs}, run_dir / "final_model.pth")

    acc, mae, bias, per, P, Y = evaluate(model, test_loader)
    tail = [h["test_bias"] for h in hist if "test_bias" in h][-TAIL_EPOCHS:]
    bs = patient_bootstrap_ci(P, Y, test_patients, seed=seed)
    strat = stratified_bias(P, Y)

    res = dict(mode=mode, seed=seed, train_year=int(TRAIN_Y), test_year=int(TEST_Y),
               roi_order=ROI, mrm=str(MRM_W),
               n_train=int((df.split == "train").sum()), n_test=int((df.split == "test").sum()),
               final_epoch=epochs, best_val_epoch=best_epoch, best_val_mae=float(best_val),
               acc=float(acc), mae=float(mae), bias=float(bias),
               tail_bias=float(np.mean(tail)) if tail else float("nan"),
               pat_bias=bs["bias"], ci_lo=bs["ci"][0], ci_hi=bs["ci"][1],
               per_roi={r: dict(acc=float(per[r][0]), mae=float(per[r][1]), bias=float(per[r][2]),
                                bias_adj=float(strat[r]["adjusted"])) for r in ROI})
    (run_dir / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    np.savez(run_dir / "test_preds.npz", preds=P, labels=Y, patients=test_patients)

    print(f"  ✔ final bias {bias:+.4f} | best_val_mae {best_val:.4f}@ep{best_epoch}")
    del model, vit, optimizer, scaler, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# W6 드라이버 — 6 arm 학습 후 best_val 체크포인트 수집
# ═══════════════════════════════════════════════════════════
DEFAULT_SEEDS = [1, 2, 42]
DIRECTIONS = [("2024to2026", "fwd"), ("2026to2024", "rev")]


def _load_result(run_dir: Path) -> dict:
    p = run_dir / "results.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text(encoding="utf-8"))
    return {"best_val_epoch": r.get("best_val_epoch"), "best_val_mae": r.get("best_val_mae"),
            "final_bias": r.get("bias"), "n_train": r.get("n_train"), "n_test": r.get("n_test"),
            "train_year": r.get("train_year"), "test_year": r.get("test_year")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=3, default=DEFAULT_SEEDS,
                    help="번호 1,2,3 에 매핑될 seed 3개 (기본 1 2 42)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ckpt", choices=["best_val", "final"], default="best_val")
    ap.add_argument("--skip-existing", action="store_true",
                    help="weights/ 에 대상 파일이 있으면 그 arm 재학습 생략")
    ap.add_argument("--keep-staging", action="store_true",
                    help="runs 스테이징의 학습 .pth 를 지우지 않고 보존")
    ap.add_argument("--mrm", default=None, help="DORGA 백본(MRM) 경로 (기본 CONFIG MRM_W)")
    args = ap.parse_args()

    global MRM_W
    if args.mrm:
        MRM_W = Path(args.mrm)

    weights_dir = OUT_ROOT / "weights"
    stage_root = OUT_ROOT / "runs"
    weights_dir.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)
    ckpt_file = f"{args.ckpt}_model.pth"
    print(f"[config] ROI={ROI}  MRM={MRM_W}  OUT={OUT_ROOT.resolve()}")

    cache = None
    manifest = {}
    for mode, prefix in DIRECTIONS:
        for idx, seed in enumerate(args.seeds, start=1):
            name = f"{prefix}{idx}"
            dst = weights_dir / f"{name}.pth"
            run_dir = stage_root / f"{mode}_s{seed}"

            if args.skip_existing and dst.exists():
                print(f"[skip] {name} (이미 존재)")
                manifest[name] = {"file": dst.name, "mode": mode, "seed": seed,
                                  "direction": prefix, "ckpt": args.ckpt, "roi_order": ROI,
                                  "mrm": str(MRM_W), "skipped": True, **_load_result(run_dir)}
                continue

            if cache is None:
                print("[cache] 2026 영상 캐시 빌드 중…")
                cache = build_full_2026_cache()

            print("\n" + "=" * 78)
            print(f"[train] {name}  mode={mode}  seed={seed}  (epochs={args.epochs}, ckpt={args.ckpt})")
            print("=" * 78)
            run_one(mode, seed, args.epochs, cache, stage_root)

            src = run_dir / ckpt_file
            if not src.exists():
                raise FileNotFoundError(f"{src} 없음. run_one 이 {ckpt_file} 을 저장하지 못했다.")
            shutil.copy2(src, dst)
            print(f"[save] {src.name} → weights/{dst.name}")
            manifest[name] = {"file": dst.name, "mode": mode, "seed": seed,
                              "direction": prefix, "ckpt": args.ckpt, "roi_order": ROI,
                              "mrm": str(MRM_W), "skipped": False, **_load_result(run_dir)}

            if not args.keep_staging:
                for f in run_dir.glob("*.pth"):
                    f.unlink()

    (weights_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"완료 — weights/ 에 {len(manifest)}개 매핑  (순서 {ROI})")
    for name, m in manifest.items():
        vm = m.get("best_val_mae")
        vm = f"{vm:.4f}" if isinstance(vm, (int, float)) else "?"
        print(f"  {name}.pth  ←  {m['mode']} s{m['seed']}  (val_mae {vm}, best@ep {m.get('best_val_epoch')})")
    print(f"  manifest: {weights_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
