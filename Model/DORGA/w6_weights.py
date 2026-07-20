# -*- coding: utf-8 -*-
r"""
W6 — 방향별 3-seed 가중치 저장 (DORGA · BSNet · PAFE, 자립 실행판)

각 모델을 2024→2026(fwd) / 2026→2024(rev) × seed 1·2·42 로 학습하고, best_val
체크포인트를 fwd1..3 / rev1..3 으로 모은다. 영역 순서 전 모델 [RT, LT, RB, LB].

  fwd1 = 2024→2026 s1   fwd2 = 2024→2026 s2   fwd3 = 2024→2026 s42
  rev1 = 2026→2024 s1   rev2 = 2026→2024 s2   rev3 = 2026→2024 s42

저장 위치: weights/<model>/{fwd,rev}{1,2,3}.pth  (model = dorga|bsnet|pafe)

세 모델의 데이터 파이프라인(manifest · 2026 seg+STN 캐시 · cross-domain split)은
공유하고, 모델별 학습만 갈라진다:
  · DORGA — dorga 레포 + MRM.pth 백본, 5-손실·prior (기존 검증 파이프라인 그대로)
  · BSNet — Private 통합 scorer(ResNet18 백본, 하드어텐션), BrixiaLoss
  · PAFE  — Private 통합 scorer(ResNet34+ViT, 3ch 내부복제), CE

★ 완전 자립: 외부 dorga 패키지(C:\Code\DORGA) 의존 없음. 전부 Private repo + timm 로 해결.
  · DORGA 모델·손실 → Model/DORGA/{DORGA.py, scorer.py}
  · DORGA 백본 → timm ViT + MRM 체크포인트 vit. 키 로드(load_mrm_vit, 인라인)
  · 2026 seg+STN 캐시 → Model/SegSTN/pipeline.py (PreprocessPipeline)
  npjDM2026 하네스(a1_common · dorga_train_2026to2024.py)에도 의존하지 않는다.

★ 영역 순서 [RT, LT, RB, LB] — DORGA 는 native [RT,RB,LT,LB] 를 데이터 레벨에서 재배열해
  박았고(라벨·박스·폴백 일관), BSNet/PAFE 의 Private scorer 는 애초에 이 순서로 출력한다.

경로는 아래 CONFIG 상수 또는 환경변수로 지정한다(서버/로컬 이식용):
  W6_MANIFEST, W6_IMG2024, W6_MASK2024, W6_SEG, W6_STN, W6_MRM, W6_OUT
MRM 은 --mrm 로도 덮어쓸 수 있다.

실행 (torch + GPU + dorga 레포):
  python w6_weights.py                         # 3모델 × 6 arm 전부
  python w6_weights.py --models bsnet pafe      # 일부 모델만
  python w6_weights.py --skip-existing          # 이미 있는 arm 생략
  python w6_weights.py --ckpt final --seeds 1 2 42 --epochs 50
"""
from __future__ import annotations

import argparse
import importlib.util
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
# ── DORGA 모델·손실: Private repo 자립 코드 (C:\Code\DORGA 패키지 불필요) ──
_HERE = os.path.dirname(os.path.abspath(__file__))          # Model/DORGA
_MODEL_ROOT = os.path.dirname(_HERE)                         # Model
for _p in (_HERE, _MODEL_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from DORGA import BrixiaViT512Dynamic, vit_base_patch16_512          # noqa: E402  모델·ViT 팩토리
from scorer import (                                                 # noqa: E402  손실 (Model/DORGA/scorer.py)
    compute_alpha_oracle, kl_attention_loss,
    loss_function, loss_function_projection, pattern_loss,
)
# seg+STN(2026 캐시)은 Model/SegSTN/pipeline.py 의 PreprocessPipeline 을 build_2026_cache 에서 로드.

# ═══════════════ 경로 — 서버 기준. 여기만 맞게 수정하면 됨 ═══════════════
BASE         = "/shared/home/mai/JeongGeon"
# ↓ 데이터 경로 (cross-domain manifest + 2024 정렬 이미지/마스크). 서버 실경로로 확인·수정.
MANIFEST     = f"{BASE}/MICCAI2026/inhauh/split_manifest.csv"     # 열: patient, year, image_path, RT,RB,LT,LB
IMG2024_DIR  = Path(f"{BASE}/MICCAI2026/inhauh/image_normalize")  # 2024 정렬 이미지
MASK2024_DIR = Path(f"{BASE}/MICCAI2026/inhauh/mask_normalize")   # 2024 정렬 마스크
# ↓ 가중치 (seg/stn 은 Private SegSTN, MRM 은 지정 경로) — 확인된 서버 경로
SEG_W        = Path(f"{BASE}/Private/Model/SegSTN/weights/finetuned_9601.pt")
STN_W        = Path(f"{BASE}/Private/Model/SegSTN/weights/stn_weights.pth")
MRM_W        = Path(f"{BASE}/MICCAI2026/MRM.pth")
# ↓ 산출물 (가중치·로그)
OUT_ROOT     = Path(f"{BASE}/Private/w6_out")

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


# ── DORGA 백본 로더 (dorga_inhauh_full.py line 756-768 방식, 통짜/인라인) ──
def load_mrm_vit(mrm_path, num_classes=C, in_chans=1):
    """timm ViT 생성 후 체크포인트의 vit. 키를 로드. C:\\Code\\DORGA 불필요."""
    vit = vit_base_patch16_512(num_classes=num_classes, in_chans=in_chans,
                               drop_path_rate=0.1, global_pool="avg")
    ck = torch.load(str(mrm_path), map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    vit_sd = {k.replace("vit.", ""): v for k, v in sd.items() if k.startswith("vit.")}
    if not vit_sd:                       # vit. 접두어 없는 순수 ViT 체크포인트인 경우
        vit_sd = dict(sd)
    vit_sd.pop("head.weight", None); vit_sd.pop("head.bias", None)
    m, u = vit.load_state_dict(vit_sd, strict=False)
    print(f"[backbone] MRM({Path(mrm_path).name}) loaded: missing={len(m)} unexpected={len(u)}")
    return vit


# ── freeze / optimizer (dorga_inhauh_full.py set_trainable/make_optimizer 이식) ──
def _set_trainable(model, freeze_blocks=FREEZE_BLOCKS):
    for p in model.parameters():
        p.requires_grad = True
    for blk in model.vit.blocks[:freeze_blocks]:
        for p in blk.parameters():
            p.requires_grad = False


def _make_optimizer(model, enc_lr=1e-5, head_lr=1e-4):
    """encoder(vit.) 와 헤드 분리 lr. 공통 규약(헤드 1e-4 / 백본 1e-5)."""
    enc = [p for n, p in model.named_parameters() if n.startswith("vit.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("vit.") and p.requires_grad]
    groups = []
    if enc:
        groups.append({"params": enc, "lr": enc_lr, "weight_decay": 5e-5})
    if head:
        groups.append({"params": head, "lr": head_lr, "weight_decay": 5e-4})
    return torch.optim.AdamW(groups)


# ═══════════════════════════════════════════════════════════
# 2026 in-code 전처리 (seg + STN align) — 저장 없이 RAM 캐시
# ═══════════════════════════════════════════════════════════
def raw_path_from_npz(npz_path: str) -> str:
    d = os.path.dirname(os.path.dirname(npz_path))
    fn = os.path.basename(npz_path)
    raw = re.match(r"\d+_([A-Za-z]+\d+)\.png", fn).group(1).upper() + ".jpg"
    return os.path.join(d, "RAW_IMAGE", raw)


def _load_preprocess_pipeline():
    """Model/SegSTN/pipeline.py 의 PreprocessPipeline 을 파일 경로로 로드 (이름 충돌 회피)."""
    path = os.path.join(_MODEL_ROOT, "SegSTN", "pipeline.py")
    spec = importlib.util.spec_from_file_location("_segstn_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                         # pipeline.py 가 seg/stn 을 자체 경로로 import
    return mod.PreprocessPipeline


@torch.no_grad()
def build_2026_cache(raw_paths):
    """RAW → Private SegSTN(GUNet seg + STN align) → 정렬 512 img + mask 캐시."""
    PreprocessPipeline = _load_preprocess_pipeline()
    pipe = PreprocessPipeline(str(SEG_W), str(STN_W), device=str(DEVICE))
    cache = {}
    for p in tqdm(raw_paths, desc="2026 seg+align (Private SegSTN)"):
        arr = np.array(Image.open(p).convert("L")).astype(np.float32) / 255.0
        t = torch.from_numpy(arr)[None, None]           # [1,1,H,W]
        aligned, masks = pipe(t)                         # aligned [1,1,512,512], masks [1,1,1024,1024] (cpu)
        mask512 = F.interpolate(masks.float(), size=(512, 512), mode="nearest")[0, 0].numpy()
        cache[p] = ((aligned[0, 0].numpy() * 255).clip(0, 255).astype(np.uint8),
                    ((mask512 > 0.5) * 255).astype(np.uint8))
    del pipe
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
# DORGA arm 학습 (mode × seed) → run_dir 에 best_val/final 저장
#   dorga 레포 모델 + MRM 백본 + 5-손실·prior (기존 검증 파이프라인)
# ═══════════════════════════════════════════════════════════
def run_one_dorga(mode, seed, epochs, cache2026, root, eval_test_every=1):
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

    vit = load_mrm_vit(MRM_W, num_classes=C, in_chans=1)
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
# BSNet / PAFE 용 공용 cross-domain 데이터셋
#   (img, lung_mask, labels[RT,LT,RB,LB]) 를 낸다. DORGA 와 같은 소스/전처리.
#   BSNet 은 lung_mask 로 하드어텐션, PAFE 는 mask 무시(내부 3ch 복제).
# ═══════════════════════════════════════════════════════════
class CrossDomainDataset(Dataset):
    def __init__(self, df, cache2026, mode="train"):
        self.df = df.reset_index(drop=True)
        self.cache = cache2026
        self.mode = mode
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

        x = self.photo(img)
        if torch.isnan(x).any():
            x = torch.zeros_like(x)
        m = torch.from_numpy((np.array(mask) > 0).astype(np.float32))[None]   # (1,H,W)
        y = torch.tensor([int(row[c]) for c in ROI], dtype=torch.long)        # [RT,LT,RB,LB]
        return x, m, y


@torch.no_grad()
def evaluate_scorer(scorer, loader):
    scorer.eval(); P, Y = [], []
    for img, mask, y in loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        out = scorer(img, mask=mask)
        P.append(out["logits"].argmax(-1).cpu()); Y.append(y)
    P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy()
    acc = (P == Y).mean(); mae = np.abs(P - Y).mean(); bias = (P - Y).mean()
    per = {ROI[i]: (float((P[:, i] == Y[:, i]).mean()), float(np.abs(P[:, i] - Y[:, i]).mean()),
                    float((P[:, i] - Y[:, i]).mean())) for i in range(R)}
    return acc, mae, bias, per, P, Y


def load_private_scorer(name):
    """Model/<Sub>/scorer.py 의 build_scorer 를 파일 경로로 로드 (이름 충돌 회피)."""
    sub = {"bsnet": "BSNet", "pafe": "PAFE"}[name]
    path = Path(__file__).resolve().parent.parent / sub / "scorer.py"   # Model/<Sub>/scorer.py
    spec = importlib.util.spec_from_file_location(f"_scorer_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_scorer


# ═══════════════════════════════════════════════════════════
# BSNet / PAFE arm 학습 (Private ScorerBase 계약) → run_dir 저장
#   freeze_stage: head_epochs 동안 백본 frozen(head lr) → 해제 후 backbone_lr 합류
#   val MAE 로만 모델 선택(test 누수 없음), best_val/final 저장
# ═══════════════════════════════════════════════════════════
def run_one_scorer(name, mode, seed, epochs, cache2026, root,
                   head_epochs=20, lr=None, backbone_lr=None, eval_test_every=1):
    set_seed(seed)
    df = pd.read_csv(MANIFEST)
    df, TRAIN_Y, TEST_Y = make_split(df, mode, seed)

    scorer = load_private_scorer(name)(classes=C).to(DEVICE)
    lr = lr if lr is not None else scorer.default_lr
    backbone_lr = backbone_lr if backbone_lr is not None else scorer.default_backbone_lr

    print("\n" + "━" * 78)
    print(f"▶ {name} arm: mode={mode} seed={seed} (train={TRAIN_Y} → test={TEST_Y}) "
          f"lr={lr} backbone_lr={backbone_lr}")
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        print(f"  [{s:<5}] {len(sub):>4}장 / {sub.patient.nunique():>3}명  "
              f"평균등급 {sub[ROI].to_numpy().mean():.4f}")
    print("━" * 78)

    mk = lambda s, m: DataLoader(CrossDomainDataset(df[df.split == s], cache2026, m),
                                 batch_size=BATCH_SIZE, shuffle=(m == "train"),
                                 drop_last=(m == "train"), num_workers=0, pin_memory=True)
    train_loader, val_loader, test_loader = mk("train", "train"), mk("val", "eval"), mk("test", "eval")
    test_patients = df[df.split == "test"].patient.to_numpy()

    run_dir = root / f"{mode}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val, best_epoch, hist = float("inf"), -1, []
    since_improve, optimizer, frozen = 0, None, None
    for epoch in range(1, epochs + 1):
        if scorer.freeze_stage:
            want_freeze = epoch <= head_epochs
            if want_freeze != frozen:
                scorer.freeze_backbone(want_freeze)
                frozen = want_freeze
                optimizer = scorer.make_optimizer(lr, backbone_lr=backbone_lr)
                print(f"  [stage] epoch {epoch}: backbone_frozen={want_freeze}")
        elif optimizer is None:
            frozen = False
            optimizer = scorer.make_optimizer(lr, backbone_lr=backbone_lr)

        scorer.train(True)
        if frozen:
            scorer.backbone_bn_eval()
        tr_P, tr_Y = [], []
        for img, mask, y in tqdm(train_loader, desc=f"  [{epoch:03d}] train", leave=False):
            img, mask, y = img.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            out = scorer(img, mask=mask, target=y)
            loss, _ = scorer.compute_loss(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr_P.append(out["logits"].detach().argmax(-1).cpu()); tr_Y.append(y.detach().cpu())

        tr_P, tr_Y = torch.cat(tr_P).numpy(), torch.cat(tr_Y).numpy()
        tr_acc, tr_bias = (tr_P == tr_Y).mean(), (tr_P - tr_Y).mean()
        v_acc, v_mae, v_bias, _, _, _ = evaluate_scorer(scorer, val_loader)
        tag = ""
        if v_mae < best_val:
            best_val, best_epoch = v_mae, epoch
            since_improve = 0
            torch.save({"state_dict": scorer.net.state_dict(), "epoch": epoch,
                        "best_val_mae": float(best_val)}, run_dir / "best_val_model.pth")
            tag = "  << BEST(val)"
        else:
            since_improve += 1

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

        if EARLYSTOP_PATIENCE and since_improve >= EARLYSTOP_PATIENCE:
            print(f"  ⏹ early stop @epoch {epoch} (best={best_epoch})")
            break

    torch.save({"state_dict": scorer.net.state_dict(), "epoch": epochs}, run_dir / "final_model.pth")

    acc, mae, bias, per, P, Y = evaluate_scorer(scorer, test_loader)
    tail = [h["test_bias"] for h in hist if "test_bias" in h][-TAIL_EPOCHS:]
    bs = patient_bootstrap_ci(P, Y, test_patients, seed=seed)
    res = dict(model=name, mode=mode, seed=seed, train_year=int(TRAIN_Y), test_year=int(TEST_Y),
               roi_order=ROI, n_train=int((df.split == "train").sum()),
               n_test=int((df.split == "test").sum()),
               final_epoch=epochs, best_val_epoch=best_epoch, best_val_mae=float(best_val),
               acc=float(acc), mae=float(mae), bias=float(bias),
               tail_bias=float(np.mean(tail)) if tail else float("nan"),
               pat_bias=bs["bias"], ci_lo=bs["ci"][0], ci_hi=bs["ci"][1],
               per_roi={r: dict(acc=per[r][0], mae=per[r][1], bias=per[r][2]) for r in ROI})
    (run_dir / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    np.savez(run_dir / "test_preds.npz", preds=P, labels=Y, patients=test_patients)

    print(f"  ✔ final bias {bias:+.4f} | best_val_mae {best_val:.4f}@ep{best_epoch}")
    del scorer, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# W6 드라이버 — 모델 × 6 arm 학습 후 best_val 체크포인트 수집
# ═══════════════════════════════════════════════════════════
DEFAULT_SEEDS = [1, 2, 42]
DEFAULT_MODELS = ["dorga", "bsnet", "pafe"]
DIRECTIONS = [("2024to2026", "fwd"), ("2026to2024", "rev")]


def train_arm(model, mode, seed, epochs, cache, stage_root):
    """모델별 학습 디스패치. DORGA 는 dorga-repo 경로, BSNet/PAFE 는 Private scorer."""
    if model == "dorga":
        run_one_dorga(mode, seed, epochs, cache, stage_root)
    else:
        run_one_scorer(model, mode, seed, epochs, cache, stage_root)


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
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    choices=DEFAULT_MODELS, help="학습할 모델 (기본 dorga bsnet pafe)")
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

    weights_root = OUT_ROOT / "weights"
    runs_root = OUT_ROOT / "runs"
    weights_root.mkdir(parents=True, exist_ok=True)
    ckpt_file = f"{args.ckpt}_model.pth"
    print(f"[config] models={args.models}  ROI={ROI}  MRM={MRM_W}  OUT={OUT_ROOT.resolve()}")

    cache = None
    manifest = {}
    for model in args.models:
        weights_dir = weights_root / model            # weights/<model>/
        stage_root = runs_root / model                # runs/<model>/
        weights_dir.mkdir(parents=True, exist_ok=True)
        stage_root.mkdir(parents=True, exist_ok=True)

        for mode, prefix in DIRECTIONS:
            for idx, seed in enumerate(args.seeds, start=1):
                name = f"{prefix}{idx}"               # fwd1 … rev3
                key = f"{model}/{name}"
                dst = weights_dir / f"{name}.pth"
                run_dir = stage_root / f"{mode}_s{seed}"
                meta = {"file": f"{model}/{name}.pth", "model": model, "mode": mode,
                        "seed": seed, "direction": prefix, "ckpt": args.ckpt, "roi_order": ROI}
                if model == "dorga":
                    meta["mrm"] = str(MRM_W)

                if args.skip_existing and dst.exists():
                    print(f"[skip] {key} (이미 존재)")
                    manifest[key] = {**meta, "skipped": True, **_load_result(run_dir)}
                    continue

                if cache is None:                     # 2026 캐시는 모든 모델/arm 공유
                    print("[cache] 2026 영상 캐시 빌드 중…")
                    cache = build_full_2026_cache()

                print("\n" + "=" * 78)
                print(f"[train] {key}  mode={mode}  seed={seed}  "
                      f"(epochs={args.epochs}, ckpt={args.ckpt})")
                print("=" * 78)
                train_arm(model, mode, seed, args.epochs, cache, stage_root)

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
        print(f"  {key}.pth  ←  {m['mode']} s{m['seed']}  "
              f"(val_mae {vm}, best@ep {m.get('best_val_epoch')})")
    print(f"  manifest: {weights_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
