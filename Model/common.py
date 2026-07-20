"""
common.py — 스코어링 모델(BSNet / PAFE / DORGA) 공통 계약·상수·손실·지표.

모든 모델은 ScorerBase 를 구현한 wrapper(각 폴더의 scorer.py)로 노출된다.
원본 모델 파일(BSNet.py / PAFE.py / DORGA.py)은 수정하지 않는다 → 기존 학습
가중치의 파라미터 키가 그대로 유지된다.

계약 (ScorerBase):
  속성
    name         모델 이름 (레지스트리 키와 동일)
    in_channels  데이터로더가 공급하는 채널 수. 전 모델 1(흑백)로 통일.
                 3채널이 필요한 모델(PAFE)은 wrapper 내부에서 복제한다.
    needs_mask   True 면 로더가 폐 마스크를 함께 공급해야 한다.
    default_lr   권장 학습률
    freeze_stage True 면 head_epochs 동안 백본 freeze 후 해제하는 2단계 학습
    backbone     freeze 대상 모듈 (property)
  메서드
    forward(img, mask=None, target=None) -> dict
        필수 키 "logits" [B,4,C], 영역 순서 [RT, LT, RB, LB].
        rel/roi 생성·채널 복제 등 모델 고유 전처리는 wrapper 내부에서 처리.
        target 은 학습 중 내부 모듈이 라벨을 쓰는 모델(DORGA)만 사용.
    compute_loss(out, target, extras=None) -> (total, parts dict)
        out 은 forward 반환 dict. extras 는 prepare_training 이 만든
        per-sample 부가정보(예: DORGA 패턴 id) 배치 텐서.
    prepare_training(labels_all, tr_idx, va_idx, device) -> np.ndarray | None
        학습 전 1회 호출. 라벨 통계(프라이어 등)가 필요한 모델만 구현.
        전체 샘플 순서의 per-sample extras 배열을 반환하면 로더가 배치에 실어준다.
    make_optimizer(lr) -> torch.optim.Optimizer
    save_weights / load_weights
        내부 원본 모델(self.net)의 state_dict 기준 → 기존 체크포인트와 키 호환.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────────────────────── 공통 상수 ─────────────────────────
IMG_SIZE = 512
CLASSES = 5                                   # 라벨 0~4
NORM_MEAN, NORM_STD = [0.56], [0.17]
SCORE_COLS = ["RT", "LT", "RB", "LB"]         # CSV 순서 = 공통 영역 순서
REGIONS = tuple(SCORE_COLS)


# ───────────────────────── 공통 손실 ─────────────────────────
class BrixiaLoss(nn.Module):
    """L = alpha*CE + (1-alpha)*MAEd.  입력은 logits [B,4,C].

    내부에서 log_softmax 를 적용하므로 raw logits(BSNet), log 확률(PAFE 의
    log(softmax)) 어느 쪽을 넣어도 동일하게 동작한다.
    """

    def __init__(self, alpha=0.7, classes=CLASSES):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("levels", torch.arange(classes, dtype=torch.float32))

    def forward(self, logits, target):
        fl = F.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1)  # [B*4, C]
        ft = target.reshape(-1).long()
        nll = F.nll_loss(fl, ft)
        exp = (fl.exp() * self.levels).sum(-1)
        mae_d = (ft.float() - exp).abs().mean()
        return self.alpha * nll + (1 - self.alpha) * mae_d, \
            {"nll": nll.detach(), "mae_d": mae_d.detach()}


# ───────────────────────── 공통 지표 ─────────────────────────
@torch.no_grad()
def metrics(logits, target):
    """logits [B,4,C], target [B,4] -> mae / acc / global_mae."""
    pred = logits.argmax(-1)                                  # [B,4]
    tgt = target.view(pred.shape)
    mae = (pred - tgt).abs().float().mean()
    acc = (pred == tgt).float().mean()
    gmae = (pred.sum(-1).float() - tgt.sum(-1).float()).abs().mean()
    return {"mae": mae.item(), "acc": acc.item(), "global_mae": gmae.item()}


# ───────────────────────── 계약 베이스 ─────────────────────────
class ScorerBase(nn.Module):
    name = "?"
    in_channels = 1
    needs_mask = False
    default_lr = 1e-4
    freeze_stage = False

    def __init__(self):
        super().__init__()
        self.net = None                       # 원본 모델. 하위 클래스가 채운다.

    # ── 필수 구현 ──
    @property
    def backbone(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, img, mask=None, target=None) -> dict:
        raise NotImplementedError

    def compute_loss(self, out, target, extras=None):
        raise NotImplementedError

    # ── 선택 구현 ──
    def prepare_training(self, labels_all, tr_idx, va_idx, device):
        return None

    def make_optimizer(self, lr):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=lr)

    # ── 공통 제공 ──
    def freeze_backbone(self, flag=True):
        for p in self.backbone.parameters():
            p.requires_grad_(not flag)

    def backbone_bn_eval(self):
        """백본 freeze 시 BN 통계가 갱신되지 않도록."""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_weights(self, path):
        torch.save(self.net.state_dict(), path)

    def load_weights(self, path, map_location="cpu", strict=True):
        state = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        return self.net.load_state_dict(state, strict=strict)
