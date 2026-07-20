"""
DORGA mask-pooling 스코어링 모델 (통짜) — 영역 순서 [RT, LT, RB, LB] 로 통일한 버전

⚠️ 이 파일은 fresh 학습 전용이다. best_model.pth 는 native [RT,RB,LT,LB] 로 학습됐으므로
   이 순서로 그 가중치를 그대로 로드하면 per-region net(roi_specific_projector.nets[r]) 과
   영역이 어긋난다. 로드해서 파인튜닝하려면 native 버전을 쓰고 출력만 재배열할 것.

이 버전에서 바꾼 것 (fresh 라 안전):
  - REL_BOXES 순서: [RT, RB, LT, LB] -> [RT, LT, RB, LB]
  - split_lungs_to_four 박스 생성 순서: 상행(R,L) 먼저, 하행(R,L) -> [RT, LT, RB, LB]
  - NATIVE_TO_DISPLAY / to_display_order 제거 (더 이상 재배열 불필요, 항등)

입력: STN 정렬 1채널 512x512 + 같은 좌표계 폐 마스크. Normalize([0.56],[0.17]) 가정.
출력: forward(x, rel, masks) -> dict, "logits_s3" [B, R, C], 순서 [RT, LT, RB, LB].
"""

import math
from functools import partial
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DORGA_MEAN = [0.56]
DORGA_STD = [0.17]

# 고정 사분면 fallback (폐 분리 실패시). 순서 [RT, LT, RB, LB].
#   RT=좌상(영상좌=환자우/상), LT=우상(영상우=환자좌/상), RB=좌하, LB=우하
REL_BOXES = torch.tensor([
    [0.0, 0.0, 0.5, 0.5],   # RT 좌상
    [0.0, 0.5, 0.5, 1.0],   # LT 우상
    [0.5, 0.0, 1.0, 0.5],   # RB 좌하
    [0.5, 0.5, 1.0, 1.0],   # LB 우하
], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════
# 폐 마스크 → 4분할.  순서 [RT, LT, RB, LB] (상행 먼저, 하행 나중)
# ═══════════════════════════════════════════════════════════
def split_lungs_to_four(mask_bin, min_area: int = 1000):
    """폐 마스크(2D) → 좌/우 폐 bbox 를 상/하 2등분 → 4 box (정규화 y0,x0,y1,x1).
    순서 [RT, LT, RB, LB]. 폐 성분<2 면 None(→ 고정 fallback).

    영상 좌측(centroid_x 작음)=환자 우폐(R), 영상 우측=환자 좌폐(L).
    상행(RT,LT) 을 먼저, 하행(RB,LB) 을 나중에 담아 [RT,LT,RB,LB] 순서를 만든다.
    """
    from skimage.measure import label, regionprops
    comps = [p for p in regionprops(label((mask_bin > 0).astype(np.uint8))) if p.area >= min_area]
    if len(comps) < 2:
        return None
    R = min(comps, key=lambda p: p.centroid[1])   # 영상 좌측 = 환자 우폐
    L = max(comps, key=lambda p: p.centroid[1])   # 영상 우측 = 환자 좌폐
    H, W = mask_bin.shape

    def halves(reg):
        y0, x0, y1, x1 = reg.bbox
        hs = np.linspace(y0, y1, 3)               # 상/하 2등분
        top = (int(round(hs[0])), int(round(hs[1])))
        bot = (int(round(hs[1])), int(round(hs[2])))
        return (x0, x1), top, bot

    (Rx0, Rx1), R_top, R_bot = halves(R)
    (Lx0, Lx1), L_top, L_bot = halves(L)
    # [RT, LT, RB, LB]
    return [
        (R_top[0] / H, Rx0 / W, R_top[1] / H, Rx1 / W),   # RT
        (L_top[0] / H, Lx0 / W, L_top[1] / H, Lx1 / W),   # LT
        (R_bot[0] / H, Rx0 / W, R_bot[1] / H, Rx1 / W),   # RB
        (L_bot[0] / H, Lx0 / W, L_bot[1] / H, Lx1 / W),   # LB
    ]


def make_rel_and_roimasks(lung_mask, img_size: int = 512, num_regions: int = 4):
    """폐 마스크(2D) → (rel [R,4], roi_masks [R,H,W]), 순서 [RT, LT, RB, LB]."""
    if torch.is_tensor(lung_mask):
        lung_mask = lung_mask.detach().cpu().numpy()
    m = (np.asarray(lung_mask) > 0).astype(np.float32)
    if m.shape != (img_size, img_size):
        mt = torch.from_numpy(m)[None, None]
        m = F.interpolate(mt, size=(img_size, img_size), mode="nearest")[0, 0].numpy()

    coords = split_lungs_to_four(m)
    if coords is None:
        coords = REL_BOXES.tolist()
    rel = torch.tensor(coords, dtype=torch.float32)
    roi_masks = np.zeros((num_regions, img_size, img_size), dtype=np.float32)
    for i in range(num_regions):
        y0, x0, y1, x1 = coords[i]
        y0, y1 = int(y0 * img_size), int(y1 * img_size)
        x0, x1 = int(x0 * img_size), int(x1 * img_size)
        roi_masks[i, y0:y1, x0:x1] = m[y0:y1, x0:x1]
    return rel, torch.from_numpy(roi_masks)


# ═══════════════════════════════════════════════════════════
# DORGA 구성요소 (원본 그대로)
# ═══════════════════════════════════════════════════════════
class ConvPatchImportance(nn.Module):
    def __init__(self, grid, embed_dim, use_pos_bias=True, temp_max=5.0):
        super().__init__()
        self.grid = grid
        self.P = grid * grid
        self.norm = nn.LayerNorm(embed_dim)
        self.dwconv = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.pwconv = nn.Conv2d(embed_dim, 1, 1)
        self.use_pos = use_pos_bias
        if use_pos_bias:
            self.pos_bias = nn.Parameter(torch.zeros(self.P))
        self.temperature = nn.Parameter(torch.tensor(2.0))
        self.temp_max = temp_max

    def forward(self, patch_tok):
        B, P, C = patch_tok.shape
        g = self.grid
        x = self.norm(patch_tok).view(B, g, g, C).permute(0, 3, 1, 2)
        x = F.gelu(self.dwconv(x))
        s = self.pwconv(x).flatten(2).squeeze(1)
        if self.use_pos:
            s = s + self.pos_bias.view(1, self.P).expand(B, -1)
        temp = self.temperature.clamp(min=0.1, max=self.temp_max)
        s_softmax = torch.softmax(s / temp, dim=1)
        entropy = -(s_softmax * (s_softmax + 1e-8).log()).sum(dim=1)
        return s_softmax * float(self.P), entropy


class Pooler_Box(nn.Module):
    def __init__(self, img_size=512, patch_size=16, num_regions=4, eps=1e-6):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_regions = num_regions
        self.eps = eps
        grid = img_size // patch_size
        self.grid = grid
        P = grid * grid
        patch_boxes = []
        for py in range(grid):
            for px in range(grid):
                y0, x0 = py * patch_size, px * patch_size
                patch_boxes.append((y0, x0, y0 + patch_size, x0 + patch_size))
        self.register_buffer("patch_boxes",
                             torch.tensor(patch_boxes, dtype=torch.float32).view(P, 4),
                             persistent=False)
        self.patch_area = patch_size * patch_size

    def _rel_to_pixel(self, rel):
        pix = rel * float(self.img_size)
        pix[..., 0::2] = pix[..., 0::2].clamp(0, self.img_size)
        pix[..., 1::2] = pix[..., 1::2].clamp(0, self.img_size)
        return pix

    def _weights_area(self, rel):
        rois_pix = self._rel_to_pixel(rel)
        patches = self.patch_boxes.view(1, 1, -1, 4)
        rois = rois_pix.unsqueeze(2)
        y0 = torch.maximum(rois[..., 0], patches[..., 0])
        x0 = torch.maximum(rois[..., 1], patches[..., 1])
        y1 = torch.minimum(rois[..., 2], patches[..., 2])
        x1 = torch.minimum(rois[..., 3], patches[..., 3])
        inter = (y1 - y0).clamp(min=0) * (x1 - x0).clamp(min=0)
        return inter / float(self.patch_area)

    def _weights_from_mask(self, masks):
        w_grid = F.adaptive_avg_pool2d(masks, output_size=(self.grid, self.grid))
        return w_grid.flatten(2)

    def forward(self, rel, patch_tokens, *, masks=None, mode="PC", s_patch=None):
        B, P, C = patch_tokens.shape
        w_area = self._weights_from_mask(masks) if masks is not None else self._weights_area(rel)

        if mode == "Mask":
            base = torch.ones((B, P), device=patch_tokens.device, dtype=patch_tokens.dtype)
        elif mode == "PC":
            if s_patch is None:
                raise ValueError("mode='PC' requires s_patch.")
            s = s_patch.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
            if torch.isnan(s).any() or torch.isinf(s).any():
                s = torch.nan_to_num(s, nan=1.0, posinf=3.0, neginf=0.0)
            s_mean = s.mean(dim=1, keepdim=True).clamp_min(self.eps)
            base = (s / s_mean).clamp(min=0.0, max=3.0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        w_base = w_area * base.unsqueeze(1)
        w_base = torch.nan_to_num(w_base, nan=0.0, posinf=1.0, neginf=0.0)
        w_sum = w_base.sum(dim=-1, keepdim=True)
        zero_mask = (w_sum < self.eps)
        w = w_base / w_sum.clamp_min(self.eps)
        if zero_mask.any():
            w = torch.where(zero_mask, torch.ones_like(w_base) / P, w)
        z = torch.einsum("brp,bpc->brc", w, patch_tokens)
        return z, w, w_base


class SharedProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, out_dim))

    def forward(self, z):
        return F.normalize(self.proj(z), p=2, dim=-1)


class ROISpecificProjector(nn.Module):
    def __init__(self, in_dim, out_dim, num_regions=4):
        super().__init__()
        self.num_regions = num_regions
        self.nets = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, out_dim))
            for _ in range(num_regions)])

    def forward(self, x):
        vs = [self.nets[r](x[:, r, :]) for r in range(self.num_regions)]
        return F.normalize(torch.stack(vs, dim=1), p=2, dim=-1)


class PatternPredictor(nn.Module):
    def __init__(self, in_dim, num_patterns, num_regions=4, num_classes=5):
        super().__init__()
        self.K = num_patterns
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + num_regions * num_classes, in_dim),
            nn.LayerNorm(in_dim), nn.GELU(), nn.Linear(in_dim, num_patterns))

    def forward(self, cls_token, logits):
        feat = torch.cat([cls_token, logits.detach().flatten(1)], dim=-1)
        return F.softmax(self.mlp(feat), dim=-1)


class DynamicPrior(nn.Module):
    def __init__(self, pi_global, pi_patterns, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("pi_global", pi_global)
        self.register_buffer("pi_patterns", pi_patterns)
        self.K = pi_patterns.shape[0]
        self.R = pi_patterns.shape[1]
        self.C = pi_patterns.shape[3]
        pp = (1 - smoothing) * pi_patterns + smoothing / self.C
        pg = (1 - smoothing) * pi_global + smoothing / self.C
        pp = pp / pp.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        pg = pg / pg.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        self.register_buffer("pi_patterns", pp)
        self.register_buffer("pi_global", pg)

    def forward(self, rho):
        pi_dyn = torch.einsum("bk,kijcd->bijcd", rho, self.pi_patterns)
        return pi_dyn / pi_dyn.sum(dim=-2, keepdim=True).clamp(min=1e-8)


class DecoupledOrdinalGraphAttentionNet(nn.Module):
    def __init__(self, dim, num_classes, num_regions=4, cls_head=None,
                 num_heads=4, step_size_init=0.0, dropout=0.1):
        super().__init__()
        self.D = dim
        self.C = num_classes
        self.R = num_regions
        self.cls_head = cls_head
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.att_scale = nn.Parameter(torch.tensor(10.0))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.att_dropout = nn.Dropout(dropout)
        self.step_size = nn.Parameter(torch.tensor(step_size_init))

    def _l2n(self, x, eps=1e-8):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    def _slerp(self, h, target, t, eps=1e-7):
        cos_theta = (h * target).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        is_close = (theta.abs() < eps) | (sin_theta.abs() < eps)
        coef_h = torch.sin((1 - t) * theta) / sin_theta.clamp_min(eps)
        coef_t = torch.sin(t * theta) / sin_theta.clamp_min(eps)
        out = coef_h * h + coef_t * target
        fallback = self._l2n((1 - t) * h + t * target)
        return self._l2n(torch.where(is_close, fallback, out))

    def forward(self, u, v, rho, pi_prior, class_anchors, labels=None, **kwargs):
        B, R, D = u.shape
        device = u.device
        C = self.C
        h = self._l2n(u)
        a_unit = self._l2n(class_anchors)
        with torch.no_grad():
            cos_sim = torch.einsum("brd,cd->brc", h, a_unit)
            p = F.one_hot(cos_sim.argmax(-1), C).float()
        pi_base = pi_prior.detach()
        pi_refined = pi_base.clone()
        pi_refined[:, torch.eye(R, device=device, dtype=torch.bool)] = torch.eye(C, device=device)
        q_ij = torch.einsum("bijcd,bjd->bijc", pi_refined, p)
        q_attn = self.q_proj(v).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        k_attn = self.k_proj(v).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        attn_logits = (q_attn @ k_attn.transpose(-2, -1)) * self.att_scale
        alpha_heads = self.att_dropout(F.softmax(attn_logits, dim=-1))
        alpha = alpha_heads.mean(dim=1)
        q_bar = torch.einsum("bij,bijc->bic", alpha, q_ij)
        class_idx = torch.arange(C, device=device, dtype=q_bar.dtype)
        mu = (q_bar * class_idx).sum(-1)
        var = (q_bar * (class_idx - mu.unsqueeze(-1)).pow(2)).sum(-1)
        xi = (1.0 - var / (((C - 1) ** 2) / 4.0)).clamp(0, 1)
        A = torch.einsum("brc,cd->brd", q_bar, a_unit.detach())
        t_eff = torch.sigmoid(self.step_size) * xi
        h_updated = self._slerp(h, self._l2n(A), t_eff)
        return {"h": h_updated, "rho": rho, "attached": {"alpha": [alpha], "q": [q_ij]}}


class AngularClassifier(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin
        self.logit_scale_s2 = nn.Parameter(torch.tensor(5.0).log())
        self.logit_scale_s3 = nn.Parameter(torch.tensor(5.0).log())

    def forward(self, x, anchors, stage="s3"):
        x_n = F.normalize(x, dim=-1)
        a_n = F.normalize(anchors, dim=-1)
        theta = torch.acos(torch.einsum("...d,cd->...c", x_n, a_n).clamp(-1 + 1e-7, 1 - 1e-7))
        scale = (self.logit_scale_s2 if stage == "s2" else self.logit_scale_s3).exp().clamp(1.0, 10.0)
        return scale * (-theta)


# ═══════════════════════════════════════════════════════════
# 메인 모델 (원 BrixiaViT512Dynamic)
# ═══════════════════════════════════════════════════════════
class BrixiaViT512Dynamic(nn.Module):
    def __init__(self, vit, num_regions=4, num_classes=5, num_patterns=7, pool_mode="PC",
                 proj_dim=768, pi_global=None, pi_patterns=None, gnn_num_heads=4, gnn_dropout=0.1):
        super().__init__()
        self.vit = vit
        self.R = num_regions
        self.C = num_classes
        self.K = num_patterns
        self.D_vit = vit.embed_dim
        self.pool_mode = pool_mode
        self.proj_dim = proj_dim
        self.register_buffer("pi_patterns", pi_patterns)
        self.register_buffer("pi_global", pi_global)

        isz = vit.patch_embed.img_size
        isz = isz[0] if isinstance(isz, (list, tuple)) else isz
        psz = vit.patch_embed.patch_size
        psz = psz[0] if isinstance(psz, (list, tuple)) else psz

        self.patch_importance = ConvPatchImportance(grid=isz // psz, embed_dim=self.D_vit)
        self.pooler = Pooler_Box(img_size=isz, patch_size=psz, num_regions=num_regions)
        self.pattern_predictor = PatternPredictor(self.D_vit, num_patterns, num_regions, num_classes)
        self.dynamic_prior = DynamicPrior(self.pi_global, self.pi_patterns, smoothing=0.05)
        self.shared_projector = SharedProjector(self.D_vit, proj_dim)
        self.roi_specific_projector = ROISpecificProjector(self.D_vit, proj_dim, num_regions)

        angles = torch.linspace(0, math.pi, num_classes)
        anch = torch.zeros(num_classes, proj_dim)
        anch[:, 0] = torch.cos(angles)
        anch[:, 1] = torch.sin(angles)
        self.class_anchors = nn.Parameter(anch)

        self.cls_head = AngularClassifier()
        self.gnn = DecoupledOrdinalGraphAttentionNet(
            dim=proj_dim, num_classes=num_classes, num_regions=num_regions,
            cls_head=self.cls_head, num_heads=gnn_num_heads, dropout=gnn_dropout)

    def _forward_tokens(self, x):
        vit = self.vit
        B = x.shape[0]
        x = vit.patch_embed(x)
        cls = vit.cls_token.expand(B, -1, -1)
        if getattr(vit, "pos_embed", None) is not None:
            x = x + vit.pos_embed[:, 1:1 + x.shape[1]]
            cls = cls + vit.pos_embed[:, :1]
        x = vit.pos_drop(torch.cat([cls, x], dim=1))
        for blk in vit.blocks:
            x = blk(x)
        return vit.norm(x)

    def forward(self, x, rel, masks=None, labels=None) -> Dict:
        seq = self._forward_tokens(x)
        cls_tok, patch_tok = seq[:, 0, :], seq[:, 1:, :]
        s_patch, entropy = self.patch_importance(patch_tok)
        z, w, w_base = self.pooler(rel=rel, patch_tokens=patch_tok, masks=masks, mode="PC", s_patch=s_patch)
        u = self.shared_projector(z)
        v = self.roi_specific_projector(z)
        logits_s2 = self.cls_head(u, self.class_anchors, stage="s2")
        rho = self.pattern_predictor(cls_tok.detach(), logits_s2.detach())
        pi_prior = self.dynamic_prior(rho)
        gnn_out = self.gnn(u=u, v=v, rho=rho, pi_prior=pi_prior,
                           class_anchors=self.class_anchors, labels=labels)
        logits_s3 = self.cls_head(gnn_out["h"], self.class_anchors, stage="s3")
        return {"logits_s2": logits_s2, "logits_s3": logits_s3, "rho": rho,
                "u": u, "v": v, "z": z, "w": w, "h": gnn_out["h"], "gnn_out": gnn_out}

    @torch.no_grad()
    def infer_from_mask(self, x, lung_mask):
        """정렬이미지 x + 폐마스크 → rel/roi_masks 자동생성 후 추론.
        반환 logits_s3 [B, R, C], 순서 [RT, LT, RB, LB]."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        isz = x.shape[-1]
        lm = lung_mask
        if torch.is_tensor(lm) and lm.dim() == 4:
            lm = lm[:, 0]
        rels, rms = [], []
        for b in range(B):
            mb = lm[b] if (torch.is_tensor(lm) and lm.dim() == 3) else lm
            rel_b, rm_b = make_rel_and_roimasks(mb, img_size=isz, num_regions=self.R)
            rels.append(rel_b); rms.append(rm_b)
        rel = torch.stack(rels).to(x.device)
        roi_masks = torch.stack(rms).to(x.device)
        return self.forward(x, rel, masks=roi_masks)["logits_s3"]


# ═══════════════════════════════════════════════════════════
# ViT 팩토리 (timm)
# ═══════════════════════════════════════════════════════════
def vit_base_patch16_512(**kwargs):
    from timm.models.vision_transformer import VisionTransformer
    return VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12,
                             num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


# ═══════════════════════════════════════════════════════════
# 빌드 헬퍼 (fresh 학습 전용 — 가중치 로드는 순서 불일치로 비권장)
# ═══════════════════════════════════════════════════════════
def build_masked_model(weights_path: Optional[str] = None, device: str = "cpu",
                       R: int = 4, C: int = 5, K: int = 7, proj_dim: int = 768,
                       verbose: bool = True):
    """mask-pooling DORGA(1ch·R4·C5·K7, 순서 [RT,LT,RB,LB]) 생성.

    ⚠️ weights_path 로 native [RT,RB,LT,LB] 가중치를 로드하면 per-region net 이 어긋난다.
       이 파일은 fresh 학습용. 로드가 필요하면 native 버전을 쓸 것.
    """
    pi_global = torch.zeros(R, R, C, C)
    pi_patterns = torch.zeros(K, R, R, C, C)
    vit = vit_base_patch16_512(num_classes=C, in_chans=1, drop_path_rate=0.1, global_pool="avg")
    model = BrixiaViT512Dynamic(vit, num_regions=R, num_classes=C, num_patterns=K, proj_dim=proj_dim,
                                pi_global=pi_global, pi_patterns=pi_patterns).to(device)
    if weights_path is not None:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        res = model.load_state_dict(sd, strict=False)
        if verbose:
            print(f"[masked] loaded {weights_path}")
            print(f"[masked] missing {len(res.missing_keys)} · unexpected {len(res.unexpected_keys)}")
            print("     ⚠️ 이 파일은 [RT,LT,RB,LB] 순서. native 가중치면 per-region net 불일치 주의.")
    model.eval()
    return model


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_masked_model(weights_path=None, device=dev, verbose=False).eval()
    B = 2
    x = torch.randn(B, 1, 512, 512, device=dev)
    mask = torch.zeros(B, 1, 512, 512)
    mask[:, :, 100:400, 60:230] = 1     # 영상 좌측 = 환자 우폐(R)
    mask[:, :, 100:400, 282:452] = 1    # 영상 우측 = 환자 좌폐(L)
    with torch.no_grad():
        logits = model.infer_from_mask(x, mask.to(dev))
    print("logits_s3 :", tuple(logits.shape), "(B, R, C) 순서 [RT, LT, RB, LB]")
    n = sum(p.numel() for p in model.parameters())
    print(f"params    : {n/1e6:.2f}M")