"""
통합 학습 스크립트 — BSNet / PAFE 를 같은 코드로 학습.

모델별 차이(입력 채널, 마스크 사용 여부, 출력 형태/종류)는 MODELS 레지스트리와
어댑터로 흡수한다. 두 모델의 출력을 공통 형태인 로그확률 [B,4,C] 로 맞춘 뒤 동일한
손실/지표를 적용한다. 영역 순서는 두 모델 모두 [RT, LT, RB, LB].

폴더:
  CXR/Merged/images_normalize   정렬 이미지 (512, uint8)
  CXR/Merged/masks              정렬 마스크 (BSNet 용, cache_masks.py 산출)
  CXR/Merged/labels.csv         uid, patient_id, RT, LT, RB, LB (0~4)
  Model/BSNet/BSNet.py          from BSNet import BSNet
  Model/PAFE/PAFE.py            from PAFE import build_v3_model

노트북:
    from train import train
    train("bsnet")
    train("pafe")
    train("bsnet", epochs=120, lr=5e-4)
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# DORGA 전용 루틴에서 사용
from DORGA import make_rel_and_roimasks, build_masked_model

# ───────────────────────── 경로 ─────────────────────────
BASE      = "/shared/home/mai/JeongGeon/Private"
IMG_DIR   = f"{BASE}/CXR/Merged/images_normalize"
MASK_DIR  = f"{BASE}/CXR/Merged/masks"
CSV_PATH  = f"{BASE}/CXR/Merged/labels.csv"
MODEL_ROOT = f"{BASE}/Model"
OUT_DIR   = f"{BASE}/Model/checkpoints"

# 모델 파일이 각 폴더에 있으므로 import 경로 추가
for sub in ("BSNet", "PAFE", "DORGA"):
    p = os.path.join(MODEL_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

# ───────────────────────── 상수 ─────────────────────────
FNAME_COL, PATIENT_COL = "uid", "patient_id"
SCORE_COLS = ["RT", "LT", "RB", "LB"]          # CSV 순서 = 공통 영역 순서
EXT = ".png"
IMG_SIZE = 512
CLASSES = 5                                     # 라벨 0~4
NORM_MEAN, NORM_STD = [0.56], [0.17]

# DORGA 파인튜닝 시 best_model.pth 경로 지정(없으면 fresh init).
# ViT-base 를 1305장으로 fresh 학습은 사실상 무리 -> 파인튜닝 권장.
# 또한 fresh 면 prior 버퍼(pi_global/pi_patterns)가 0->uniform 으로 degrade 된다.
DORGA_WEIGHTS = None


# ═════════════════════ 모델 빌더 & 어댑터 ═════════════════════
def _build_bsnet(classes, vertical_overlap):
    from BSNet import BSNet
    return BSNet(in_channels=1, classes=classes, hard_attention=True,
                 vertical_overlap=vertical_overlap,
                 pretrained_backbone=True, seg_model=None)


def _build_pafe(classes, vertical_overlap):
    from PAFE import build_v3_model                # vertical_overlap 미사용(2x2 무중첩 고정)
    return build_v3_model(kind="hybrid", backbone="resnet34",
                          num_classes=classes, num_regions=4)


def _build_dorga(classes, vertical_overlap):
    from DORGA import build_masked_model         # 마스크 pooling 버전 (forward(x,rel,masks))
    model = build_masked_model(weights_path=DORGA_WEIGHTS, C=classes, R=4, K=7,
                               verbose=DORGA_WEIGHTS is not None)
    return model


# ── forward 호출 방식 (모델별 시그니처 차이 흡수) ──
def _call_bsnet(model, img, mask):
    return model(img, mask=mask)                   # logits [B,2,2,C]

def _call_pafe(model, img, mask):
    return model(img)                              # softmax 확률 [B,4,C]

def _call_dorga(model, img, mask):
    # 폐 마스크에서 배치별로 rel(영역박스)과 roi_masks 를 동적 생성
    from DORGA import make_rel_and_roimasks
    rels, rois = [], []
    for b in range(img.size(0)):
        r, rm = make_rel_and_roimasks(mask[b, 0], img_size=img.size(-1), num_regions=4)
        rels.append(r); rois.append(rm)
    rel = torch.stack(rels).to(img.device)
    roi = torch.stack(rois).to(img.device)
    return model(img, rel, masks=roi)              # dict, 영역순서 native [RT,RB,LT,LB]


# ── 출력 -> 로그확률 [B,4,C] ──
def _adapt_bsnet(raw):
    B, C = raw.size(0), raw.size(-1)
    return F.log_softmax(raw.reshape(B, 4, C), dim=-1)          # logits -> 로그확률

def _adapt_pafe(raw):
    return raw.clamp_min(1e-8).log()                            # 확률 -> 로그확률

def _adapt_dorga(raw):
    # DORGA.py 가 이미 [RT,LT,RB,LB] 순서 -> 재배열 불필요
    return F.log_softmax(raw["logits_s3"], dim=-1)


MODELS = {
    "bsnet": {"in_channels": 1, "use_mask": True,  "backbone_attr": "backbone",
              "alpha": 0.7,   # 0.7*CE + 0.3*MAEd (BSNet 논문 손실)
              "build": _build_bsnet, "call": _call_bsnet, "adapt": _adapt_bsnet},
    "pafe":  {"in_channels": 3, "use_mask": False, "backbone_attr": "backbone",
              "alpha": 1.0,   # 순수 CE (MAEd 제거)
              "build": _build_pafe,  "call": _call_pafe,  "adapt": _adapt_pafe},
    "dorga": {"in_channels": 1, "use_mask": True,  "backbone_attr": "vit",
              "build": _build_dorga, "call": _call_dorga, "adapt": _adapt_dorga},
}


def forward_model(model, cfg, img, mask):
    """모델별 입력/마스크/시그니처 차이를 흡수하고 로그확률 [B,4,C] 반환."""
    if cfg["in_channels"] == 3 and img.size(1) == 1:
        img = img.repeat(1, 3, 1, 1)               # 흑백 3복제 (PAFE 규약)
    raw = cfg["call"](model, img, mask)
    return cfg["adapt"](raw)


# ═════════════════════ 데이터셋 (inha_dataset.py 사용) ═════════════════════
# InhaUHDataset: (img, mask, target[2,2]) 또는 (img, target[2,2]) 반환.
# target 은 [[RT,LT],[RB,LB]] -> 학습 시 (B,4) [RT,LT,RB,LB] 로 편다.
from inha_dataset import InhaUHDataset, split_indices_by_patient


def make_dataset(rows, mode, use_mask):
    return InhaUHDataset(IMG_DIR, CSV_PATH, mode=mode, rows=rows,
                         mask_dir=(MASK_DIR if use_mask else None))


def split_by_patient(val_frac=0.15, seed=0):
    tr, va = split_indices_by_patient(CSV_PATH, val_frac=val_frac, seed=seed)
    return tr, va


# ═════════════════════ 손실/지표 (로그확률 입력) ═════════════════════
class BrixiaLoss(nn.Module):
    """L = alpha*NLL + (1-alpha)*MAEd.  입력은 로그확률 [B,4,C].
    NLL(로그확률) = 크로스엔트로피. MAEd 는 exp(logprob)=확률로 기대점수 계산."""
    def __init__(self, alpha=0.7, classes=CLASSES):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("levels", torch.arange(classes, dtype=torch.float32))

    def forward(self, logprob, target):
        fl = logprob.reshape(-1, logprob.size(-1))                 # [B*4, C]
        ft = target.reshape(-1).long()
        nll = F.nll_loss(fl, ft)
        exp = (fl.exp() * self.levels).sum(-1)
        mae_d = (ft.float() - exp).abs().mean()
        return self.alpha * nll + (1 - self.alpha) * mae_d, \
            {"nll": nll.detach(), "mae_d": mae_d.detach()}


@torch.no_grad()
def metrics(logprob, target):
    pred = logprob.argmax(-1)                                      # [B,4]
    tgt = target.view(pred.shape)
    mae = (pred - tgt).abs().float().mean()
    acc = (pred == tgt).float().mean()
    gmae = (pred.sum(-1).float() - tgt.sum(-1).float()).abs().mean()
    return {"mae": mae.item(), "acc": acc.item(), "global_mae": gmae.item()}


# ═════════════════════ 백본 freeze (모델별 backbone_attr) ═════════════════════
def set_backbone_grad(model, attr, trainable):
    bb = getattr(model, attr, None)
    if bb is not None:
        for p in bb.parameters():
            p.requires_grad_(trainable)

def backbone_bn_eval(model, attr):
    bb = getattr(model, attr, None)
    if bb is not None:
        for m in bb.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


# ═════════════════════ 학습 루프 ═════════════════════
def run_epoch(model, cfg, loader, criterion, device, optimizer=None, frozen=False):
    train = optimizer is not None
    model.train(train)
    if train and frozen:
        backbone_bn_eval(model, cfg["backbone_attr"])

    tot, agg = 0, {"loss":0,"nll":0,"mae_d":0,"mae":0,"acc":0,"global_mae":0}
    for batch in loader:
        if cfg["use_mask"]:
            img, mask, target = batch
            mask = mask.to(device)
        else:
            img, target = batch
            mask = None
        img, target = img.to(device), target.to(device)
        target = target.reshape(target.size(0), -1)   # (B,2,2) -> (B,4) [RT,LT,RB,LB]
        with torch.set_grad_enabled(train):
            logprob = forward_model(model, cfg, img, mask)
            loss, parts = criterion(logprob, target)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        bs = img.size(0); tot += bs
        m = metrics(logprob, target)
        agg["loss"] += loss.item()*bs
        agg["nll"] += parts["nll"].item()*bs
        agg["mae_d"] += parts["mae_d"].item()*bs
        for k in ("mae","acc","global_mae"): agg[k] += m[k]*bs
    return {k: v/tot for k, v in agg.items()}


def train(model_name="bsnet", epochs=80, batch=8, lr=1e-3, head_epochs=20,
          classes=CLASSES, vertical_overlap=0.25, workers=4, seed=0, K=7):
    assert model_name in MODELS, f"model 은 {list(MODELS)} 중 하나"
    cfg = MODELS[model_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    if cfg["use_mask"] and not os.path.isdir(MASK_DIR):
        raise FileNotFoundError(f"마스크 폴더 없음: {MASK_DIR}. 먼저 cache_masks.py 실행.")

    tr_idx, va_idx = split_by_patient(0.15, seed)

    # ── DORGA 는 5-손실·6튜플·prior 라 별도 루틴으로 분기 ──
    if model_name == "dorga":
        import pandas as pd
        labels_all = pd.read_csv(CSV_PATH)[["RT", "LT", "RB", "LB"]].to_numpy()
        base_tr = make_dataset(tr_idx, "train", use_mask=True)
        base_va = make_dataset(va_idx, "val", use_mask=True)
        return train_dorga(base_tr, base_va, tr_idx, va_idx, labels_all,
                              epochs=epochs, batch=batch, lr=lr, K=K,
                              classes=classes, workers=workers, out_dir=OUT_DIR,
                              weights_path=DORGA_WEIGHTS)

    train_ld = DataLoader(make_dataset(tr_idx, "train", cfg["use_mask"]), batch,
                          shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(make_dataset(va_idx, "val", cfg["use_mask"]), batch,
                        shuffle=False, num_workers=workers, pin_memory=True)

    model = cfg["build"](classes, vertical_overlap).to(device)
    criterion = BrixiaLoss(alpha=cfg.get("alpha", 0.7), classes=classes).to(device)
    print(f"[model] {model_name}  in_ch={cfg['in_channels']}  mask={cfg['use_mask']}")

    best, optimizer, frozen = float("inf"), None, None
    for ep in range(1, epochs + 1):
        want_freeze = ep <= head_epochs
        if want_freeze != frozen:
            set_backbone_grad(model, cfg["backbone_attr"], not want_freeze)
            frozen = want_freeze
            cur_lr = lr if want_freeze else lr * 0.3
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=cur_lr)
            print(f"[stage] epoch {ep}: backbone_frozen={want_freeze}, lr={cur_lr}")

        tr = run_epoch(model, cfg, train_ld, criterion, device, optimizer, frozen)
        va = run_epoch(model, cfg, val_ld, criterion, device, None)
        print(f"E{ep:03d} | train loss {tr['loss']:.3f} mae {tr['mae']:.3f} "
              f"| val loss {va['loss']:.3f} mae {va['mae']:.3f} "
              f"gMAE {va['global_mae']:.3f} acc {va['acc']:.3f} | mae_d {tr['mae_d']:.3f}")

        if va["global_mae"] < best:
            best = va["global_mae"]
            path = os.path.join(OUT_DIR, f"{model_name}_best.pth")
            torch.save(model.state_dict(), path)
            print(f"  saved {os.path.basename(path)} (val global_mae {best:.3f})")

    print(f"완료 [{model_name}]. best val global_mae = {best:.3f}")
    return model




# ═════════════════════ DORGA 전용 (5-손실·6튜플·prior) ═════════════════════
def loss_function(logits, labels, roi_class=None, option="RoiClass"):
    B, R, C = logits.shape
    if option == "CE" or roi_class is None:
        return F.cross_entropy(logits.reshape(-1, C), labels.reshape(-1))
    cnt = roi_class.to(device=logits.device, dtype=logits.dtype).clamp(min=1e-6)
    if option == "Class":
        class_cnt = cnt.sum(dim=0)
        w = (class_cnt.sum() / class_cnt); w = w / w.mean()
        return F.cross_entropy(logits.reshape(-1, C), labels.reshape(-1), weight=w)
    # RoiClass
    w_rc = (cnt.sum(dim=1, keepdim=True) / cnt); w_rc = w_rc / w_rc.mean()
    loss_per = F.cross_entropy(logits.reshape(-1, C), labels.reshape(-1),
                               reduction="none").view(B, R)
    y = labels.long()
    r_idx = torch.arange(R, device=logits.device).view(1, R).expand(B, R)
    w_pos = w_rc[r_idx, y.clamp(min=0)]
    return (loss_per * w_pos).sum() / w_pos.sum().clamp_min(1e-6)


def loss_function_projection(features, labels, anchors, roi_class_counts=None,
                             penalty_weight=10.0):
    device = features.device
    C = anchors.shape[0]
    u = F.normalize(features, p=2, dim=-1)
    a = F.normalize(anchors, p=2, dim=-1)
    target = a[labels]
    direct_cos = (u * target).sum(dim=-1)
    L_align = 1.0 - direct_cos
    w = F.normalize(a[C - 1] - a[0], p=2, dim=-1)
    u_pc = (u * w).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    a_pc = (a * w).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    L_1D = F.l1_loss(torch.acos(u_pc), torch.acos(a_pc)[labels], reduction='none')
    L_obtuse = F.relu(-direct_cos) * penalty_weight
    L_sev = L_align + L_1D + L_obtuse
    if roi_class_counts is not None:
        counts = roi_class_counts.to(device=device, dtype=features.dtype)
        inv = 1.0 / counts.clamp(min=1.0)
        w_rc = (inv / inv.sum(dim=-1, keepdim=True)) * C
        r_idx = torch.arange(u.shape[1], device=device).unsqueeze(0).expand(u.shape[0], -1)
        L_sev_mean = (L_sev * w_rc[r_idx, labels]).mean()
    else:
        L_sev_mean = L_sev.mean()
    L_polar = (1.0 + (a[0] * a[C - 1]).sum()).pow(2)
    target_cos = math.cos(math.pi / (C - 1))
    adj = (a[:-1] * a[1:]).sum(-1)
    L_ord = F.mse_loss(adj, torch.full_like(adj, target_cos))
    return L_sev_mean + L_polar + L_ord, {}


def compute_alpha_oracle(pi_gt, labels, p_model_logits, oracle_tau=0.1):
    B, R = labels.shape
    C = pi_gt.shape[-1]
    device = labels.device
    p_gt = F.one_hot(labels.long(), C).float()
    q_all = torch.einsum("bijcd,bjd->bijc", pi_gt, p_gt)
    p_self = F.softmax(p_model_logits.detach(), dim=-1)
    q_self = torch.einsum("bijcd,bjd->bijc", pi_gt, p_self)
    mask = torch.eye(R, device=device).bool().reshape(1, R, R, 1)
    q_hyb = torch.where(mask, q_self, q_all)
    ci = torch.arange(C, device=device).float()
    E_q = (q_hyb * ci).sum(-1)
    dist = (E_q - labels.float().unsqueeze(-1)).abs()
    alpha_target = F.softmax(-dist / oracle_tau, dim=-1).detach()
    pred = p_model_logits.detach().argmax(-1)
    is_wrong = (pred != labels).float()
    raw_conf = (-dist.min(dim=-1).values).exp()
    confidence = (is_wrong * raw_conf).detach()
    return alpha_target, confidence


def kl_attention_loss(alpha_pred, alpha_target, confidence, eps=1e-9):
    kl = F.kl_div((alpha_pred + eps).log(), alpha_target, reduction='none').sum(-1)
    return (kl * confidence).sum() / confidence.sum().clamp(min=1e-6)


def pattern_loss(rho, pattern_gt):
    return F.cross_entropy(rho.clamp_min(1e-12).log(), pattern_gt.long())


# ══════════════ pattern / prior / roi_class (우리 R=4,C=5 로 이식) ══════════════
SCORE_COLS = ["RT", "LT", "RB", "LB"]


def generate_severity_patterns(labels_np, n_patterns=7, random_state=42):
    """labels_np [N,4] -> pattern id [N]. train/val 전체로 KMeans, 심각도합으로 재정렬."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_patterns, random_state=random_state, n_init=20)
    cl = km.fit_predict(labels_np.astype(np.float32))
    sev = sorted([(k, labels_np[cl == k].sum(1).mean()) for k in range(n_patterns)],
                 key=lambda x: x[1])
    remap = {old: new for new, (old, _) in enumerate(sev)}
    return np.array([remap[c] for c in cl], dtype=np.int64)


def compute_priors(labels_np, pattern_ids, R=4, C=5, K=7, eps=1e-7):
    """패턴별 영역간 전이확률. pi_global [R,R,C,C], pi_patterns [K,R,R,C,C]."""
    lab = labels_np.astype(np.int32)

    def build(sub):
        pi = np.zeros((R, R, C, C), dtype=np.float32)
        for i in range(R):
            for j in range(R):
                for d in range(C):
                    md = (sub[:, j] == d); n_d = md.sum()
                    if n_d == 0:
                        pi[i, j, :, d] = 1.0 / C
                    else:
                        cnt = np.bincount(sub[md, i], minlength=C)
                        pi[i, j, :, d] = (cnt + eps) / (n_d + eps * C)
        return pi

    pi_global = build(lab)
    pi_patterns = np.zeros((K, R, R, C, C), dtype=np.float32)
    for k in range(K):
        sub = lab[pattern_ids == k]
        pi_patterns[k] = pi_global if sub.shape[0] < 10 else build(sub)
    pi_global = np.nan_to_num(pi_global, nan=1.0 / C)
    pi_patterns = np.nan_to_num(pi_patterns, nan=1.0 / C)
    return torch.from_numpy(pi_global), torch.from_numpy(pi_patterns)


def compute_roi_class_counts(labels_np, R=4, C=5):
    """roi_class_counts [R,C]."""
    out = np.zeros((R, C), dtype=np.float32)
    for r in range(R):
        for c in range(C):
            out[r, c] = (labels_np[:, r] == c).sum()
    return torch.from_numpy(out)


# ══════════════ DORGA 전용 데이터셋 (6요소 배치) ══════════════
class DorgaDataset(Dataset):
    """InhaUHDataset 를 감싸 rel/roi_masks/pattern_gt 를 추가로 낸다.
    base 는 (img, mask, target[2,2]) 를 내는 InhaUHDataset (mask_dir 지정 필수).
    patterns: base.df 순서에 맞춘 pattern id 배열 (train/val 각각 슬라이스)."""

    def __init__(self, base, patterns):
        self.base = base
        self.patterns = patterns
        assert base.mask_dir is not None, "DORGA 는 마스크 필요"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask, target = self.base[idx]          # mask (1,H,W), target (2,2)
        lab = target.reshape(-1).long()             # [4] = [RT,LT,RB,LB]
        rel, roi = make_rel_and_roimasks(mask[0], img_size=img.shape[-1], num_regions=4)
        pat = torch.tensor(int(self.patterns[idx]), dtype=torch.long)
        return img, lab, rel, roi, pat, str(idx)


# ══════════════ 지표 ══════════════
@torch.no_grad()
def dorga_metrics(logits, lab):
    pred = logits.argmax(-1)                         # [B,4]
    mae = (pred - lab).abs().float().mean()
    acc = (pred == lab).float().mean()
    gmae = (pred.sum(-1).float() - lab.sum(-1).float()).abs().mean()
    return {"mae": mae.item(), "acc": acc.item(), "global_mae": gmae.item()}


# ══════════════ 학습 루프 ══════════════
def run_dorga_epoch(model, loader, optimizer, device, roi_class, train=True):
    model.train(train)
    tot = 0
    agg = {"loss": 0, "s1": 0, "s2": 0, "s3": 0, "att": 0,
           "mae": 0, "acc": 0, "global_mae": 0}
    for img, lab, rel, roi, pat, _ in loader:
        img, lab = img.to(device), lab.to(device)
        rel, roi, pat = rel.to(device), roi.to(device), pat.to(device)
        with torch.set_grad_enabled(train):
            out = model(img, rel, masks=roi, labels=lab)
            loss_s1 = pattern_loss(out["rho"], pat)
            loss_s2_ce = loss_function(out["logits_s2"], lab, roi_class, "RoiClass")
            loss_s2_proj, _ = loss_function_projection(out["u"], lab, model.class_anchors, roi_class)
            # v_orth (저장소 원본): roi_specific v 가 클래스축(anchor 0->C-1)과 직교하도록.
            v = out["v"]
            a = F.normalize(model.class_anchors, dim=-1)
            w_vec = F.normalize(a[model.C - 1] - a[0], dim=-1)
            v_w_cos = (v * w_vec).sum(dim=-1).clamp(-1.0, 1.0)
            loss_v_orth = (torch.acos(v_w_cos) - math.pi / 2).abs().mean()
            loss_s2 = loss_s2_ce + loss_s2_proj + loss_v_orth
            loss_s3 = loss_function(out["logits_s3"], lab, roi_class, "RoiClass")
            rho_gt = F.one_hot(pat.long(), num_classes=model.K).float()
            pi_oracle = model.dynamic_prior(rho_gt)
            alpha_target, conf = compute_alpha_oracle(
                pi_gt=pi_oracle.detach(), labels=lab,
                p_model_logits=out["logits_s2"].detach(), oracle_tau=0.1)
            alpha_pred = out["gnn_out"]["attached"]["alpha"][0]
            loss_att = kl_attention_loss(alpha_pred, alpha_target, conf)
            total = loss_s1 + loss_s2 + loss_s3 + loss_att
            if train:
                optimizer.zero_grad(); total.backward(); optimizer.step()
        bs = img.size(0); tot += bs
        m = dorga_metrics(out["logits_s3"], lab)
        agg["loss"] += total.item() * bs
        agg["s1"] += loss_s1.item() * bs; agg["s2"] += loss_s2.item() * bs
        agg["s3"] += loss_s3.item() * bs; agg["att"] += loss_att.item() * bs
        for k in ("mae", "acc", "global_mae"): agg[k] += m[k] * bs
    return {k: v / tot for k, v in agg.items()}


def train_dorga(base_train, base_val, tr_idx, va_idx, labels_all,
                epochs=80, batch=8, lr=1e-4, K=7, classes=5, workers=4,
                out_dir=".", weights_path=None):
    """base_train/base_val = InhaUHDataset (mask_dir 지정).  labels_all [N,4] 전체 라벨.
    tr_idx/va_idx = 환자분할 인덱스 (labels_all 기준, base_* 의 df 순서와 동일)."""
    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    # 패턴/프라이어는 train+val 로 생성 (저장소와 동일: train/valid split 사용)
    tv_idx = np.concatenate([tr_idx, va_idx])
    pat_tv = generate_severity_patterns(labels_all[tv_idx], n_patterns=K)
    # tv 인덱스 -> 패턴을 원위치에 매핑
    pat_full = np.zeros(len(labels_all), dtype=np.int64)
    pat_full[tv_idx] = pat_tv
    pi_global, pi_patterns = compute_priors(labels_all[tv_idx], pat_tv, R=4, C=classes, K=K)
    roi_class = compute_roi_class_counts(labels_all[tr_idx], R=4, C=classes).to(device)

    # 모델 (prior 주입)
    model = build_masked_model(weights_path=weights_path, device=device,
                               R=4, C=classes, K=K, verbose=weights_path is not None)
    model.dynamic_prior = type(model.dynamic_prior)(pi_global, pi_patterns,
                                                    smoothing=0.05).to(device)
    model.register_buffer("pi_global", pi_global.to(device))
    model.register_buffer("pi_patterns", pi_patterns.to(device))
    model = model.to(device)

    train_ds = DorgaDataset(base_train, pat_full[tr_idx])
    val_ds = DorgaDataset(base_val, pat_full[va_idx])
    train_ld = DataLoader(train_ds, batch, shuffle=True, num_workers=workers,
                          pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch, shuffle=False, num_workers=workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float("inf")
    for ep in range(1, epochs + 1):
        tr = run_dorga_epoch(model, train_ld, optimizer, device, roi_class, train=True)
        va = run_dorga_epoch(model, val_ld, optimizer, device, roi_class, train=False)
        print(f"E{ep:03d} | train loss {tr['loss']:.3f} (s1 {tr['s1']:.2f} s2 {tr['s2']:.2f} "
              f"s3 {tr['s3']:.2f} att {tr['att']:.2f}) mae {tr['mae']:.3f} "
              f"| val mae {va['mae']:.3f} gMAE {va['global_mae']:.3f} acc {va['acc']:.3f}")
        if va["global_mae"] < best:
            best = va["global_mae"]
            torch.save(model.state_dict(), os.path.join(out_dir, "dorga_best.pth"))
            print(f"  saved (val global_mae {best:.3f})")
    print(f"완료 [dorga]. best val global_mae = {best:.3f}")
    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="bsnet", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    train(a.model, epochs=a.epochs, batch=a.batch, lr=a.lr)