"""
scorer.py — DORGA(mask-pooling) 의 공통 계약(ScorerBase) wrapper.

원본 DORGA.py 는 수정하지 않는다. 가중치 키는 self.net(=BrixiaViT512Dynamic)
기준으로 저장/로드되어 기존 체크포인트(dorga_best.pth)와 호환된다.

  입력  : 흑백 1ch 512x512 + 폐 마스크.
          rel(영역박스)/roi_masks 는 wrapper 가 마스크에서 배치별로 생성한다.
  출력  : {"logits": [B,4,C]} (= logits_s3) 순서 [RT, LT, RB, LB]
          + 원본 dict 의 나머지 키(logits_s2, rho, u, v, gnn_out ...) 유지
  손실  : s1(패턴 CE) + s2(RoiClass CE + projection + v_orth) + s3(RoiClass CE)
          + attention KL. 라벨 통계 기반 프라이어/roi_class 는 prepare_training
          에서 만들고, per-sample 패턴 id 를 extras 로 요구한다.

DORGA 전용 손실·프라이어 루틴은 train.py 에서 이 파일로 이관했다 (동작 동일).
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):          # 자기 폴더 + Model/ 루트
    if p not in sys.path:
        sys.path.insert(0, p)

from DORGA import (DynamicPrior, build_masked_model,          # noqa: E402
                   make_rel_and_roimasks)
from common import CLASSES, ScorerBase                        # noqa: E402


# ═════════════════════ DORGA 전용 손실 (원 train.py 에서 이관) ═════════════════════
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


# ═════════════════ 패턴/프라이어/roi_class (원 train.py 에서 이관) ═════════════════
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


# ═════════════════════════════ Scorer ═════════════════════════════
class DorgaScorer(ScorerBase):
    name = "dorga"
    in_channels = 1
    needs_mask = True
    freeze_stage = False          # 원 루틴대로 전체 파라미터 동시 학습.
    optimizer_cls = torch.optim.AdamW       # lr 은 공통 고정값(1e-4/1e-5) 상속

    def __init__(self, classes=CLASSES, K=7, weights_path=None):
        super().__init__()
        # ViT-base 를 소규모 데이터로 fresh 학습은 사실상 무리 -> 파인튜닝 권장.
        # weights_path=None 이면 prior 버퍼가 0 -> prepare_training 에서 주입된다.
        self.net = build_masked_model(weights_path=weights_path, C=classes, R=4,
                                      K=K, verbose=weights_path is not None)
        self.roi_class = None                # prepare_training 에서 설정

    @property
    def backbone(self):
        return self.net.vit

    # ── forward: 마스크 -> rel/roi_masks 생성 후 원본 호출 ──
    def forward(self, img, mask=None, target=None):
        if mask is None:
            raise ValueError("DORGA 는 폐 마스크가 필요함 (needs_mask=True)")
        rels, rois = [], []
        for b in range(img.size(0)):
            r, rm = make_rel_and_roimasks(mask[b, 0], img_size=img.size(-1),
                                          num_regions=4)
            rels.append(r); rois.append(rm)
        rel = torch.stack(rels).to(img.device)
        roi = torch.stack(rois).to(img.device)
        out = self.net(img, rel, masks=roi, labels=target)
        out["logits"] = out["logits_s3"]     # 공통 키. 순서 [RT,LT,RB,LB]
        return out

    # ── 학습 전 1회: 패턴/프라이어/roi_class 생성, per-sample 패턴 id 반환 ──
    def prepare_training(self, labels_all, tr_idx, va_idx, device):
        K = self.net.K
        classes = self.net.C
        tv_idx = np.concatenate([tr_idx, va_idx])
        pat_tv = generate_severity_patterns(labels_all[tv_idx], n_patterns=K)
        pat_full = np.zeros(len(labels_all), dtype=np.int64)
        pat_full[tv_idx] = pat_tv

        pi_global, pi_patterns = compute_priors(labels_all[tv_idx], pat_tv,
                                                R=4, C=classes, K=K)
        self.net.dynamic_prior = DynamicPrior(pi_global, pi_patterns,
                                              smoothing=0.05).to(device)
        self.net.pi_global.copy_(pi_global.to(device))
        self.net.pi_patterns.copy_(pi_patterns.to(device))
        self.roi_class = compute_roi_class_counts(labels_all[tr_idx],
                                                  R=4, C=classes).to(device)
        return pat_full

    # ── 5-손실 조합 (원 run_dorga_epoch 과 동일) ──
    def compute_loss(self, out, target, extras=None):
        if extras is None:
            raise ValueError("DORGA 손실은 패턴 id(extras)가 필요함. "
                             "prepare_training 을 거친 학습 루프에서 호출할 것.")
        pat = extras
        net = self.net
        lab = target.long()

        loss_s1 = pattern_loss(out["rho"], pat)
        loss_s2_ce = loss_function(out["logits_s2"], lab, self.roi_class, "RoiClass")
        loss_s2_proj, _ = loss_function_projection(out["u"], lab,
                                                   net.class_anchors, self.roi_class)
        # v_orth: roi_specific v 가 클래스축(anchor 0->C-1)과 직교하도록.
        v = out["v"]
        a = F.normalize(net.class_anchors, dim=-1)
        w_vec = F.normalize(a[net.C - 1] - a[0], dim=-1)
        v_w_cos = (v * w_vec).sum(dim=-1).clamp(-1.0, 1.0)
        loss_v_orth = (torch.acos(v_w_cos) - math.pi / 2).abs().mean()
        loss_s2 = loss_s2_ce + loss_s2_proj + loss_v_orth

        loss_s3 = loss_function(out["logits_s3"], lab, self.roi_class, "RoiClass")

        rho_gt = F.one_hot(pat.long(), num_classes=net.K).float()
        pi_oracle = net.dynamic_prior(rho_gt)
        alpha_target, conf = compute_alpha_oracle(
            pi_gt=pi_oracle.detach(), labels=lab,
            p_model_logits=out["logits_s2"].detach(), oracle_tau=0.1)
        alpha_pred = out["gnn_out"]["attached"]["alpha"][0]
        loss_att = kl_attention_loss(alpha_pred, alpha_target, conf)

        total = loss_s1 + loss_s2 + loss_s3 + loss_att
        return total, {"s1": loss_s1.detach(), "s2": loss_s2.detach(),
                       "s3": loss_s3.detach(), "att": loss_att.detach()}


def build_scorer(classes=CLASSES, **kw):
    return DorgaScorer(classes=classes, **kw)


if __name__ == "__main__":
    s = build_scorer()
    B = 2
    img = torch.randn(B, 1, 512, 512)
    mask = torch.zeros(B, 1, 512, 512)
    mask[:, :, 100:400, 60:230] = 1      # 영상 좌측 = 환자 우폐(R)
    mask[:, :, 100:400, 282:452] = 1     # 영상 우측 = 환자 좌폐(L)
    with torch.no_grad():
        out = s(img, mask=mask)
    print("logits :", tuple(out["logits"].shape), "순서 [RT, LT, RB, LB]")
