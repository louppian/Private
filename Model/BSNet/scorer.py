"""
scorer.py — BSNet 의 공통 계약(ScorerBase) wrapper.

원본 BSNet.py 는 수정하지 않는다. 가중치 키는 self.net(=BSNet) 기준으로
저장/로드되어 기존 체크포인트(bsnet_best.pth)와 호환된다.

  입력  : 흑백 1ch 512x512 + 폐 마스크(하드 어텐션용)
  출력  : {"logits": [B,4,C]}  순서 [RT, LT, RB, LB]
  손실  : BrixiaLoss(0.7*CE + 0.3*MAEd)  — BSNet 논문 손실
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):          # 자기 폴더 + Model/ 루트
    if p not in sys.path:
        sys.path.insert(0, p)

from BSNet import BSNet                              # noqa: E402
from common import CLASSES, BrixiaLoss, ScorerBase   # noqa: E402


class BSNetScorer(ScorerBase):
    name = "bsnet"
    in_channels = 1
    needs_mask = True
    default_lr = 1e-3
    freeze_stage = True

    def __init__(self, classes=CLASSES, vertical_overlap=0.25,
                 pretrained_backbone=True):
        super().__init__()
        self.net = BSNet(in_channels=1, classes=classes, hard_attention=True,
                         vertical_overlap=vertical_overlap,
                         pretrained_backbone=pretrained_backbone, seg_model=None)
        self.criterion = BrixiaLoss(alpha=0.7, classes=classes)

    @property
    def backbone(self):
        return self.net.backbone

    def forward(self, img, mask=None, target=None):
        raw = self.net(img, mask=mask)               # [B,2,2,C] = [[RT,LT],[RB,LB]]
        B, C = raw.size(0), raw.size(-1)
        return {"logits": raw.reshape(B, 4, C)}      # 행 우선 -> [RT,LT,RB,LB]

    def compute_loss(self, out, target, extras=None):
        return self.criterion(out["logits"], target)


def build_scorer(classes=CLASSES, **kw):
    return BSNetScorer(classes=classes, **kw)


if __name__ == "__main__":
    import torch
    s = build_scorer(pretrained_backbone=False)
    img = torch.randn(2, 1, 512, 512)
    mask = torch.rand(2, 1, 512, 512)
    out = s(img, mask=mask)
    loss, parts = s.compute_loss(out, torch.randint(0, 5, (2, 4)))
    print("logits :", tuple(out["logits"].shape))
    print("loss   :", float(loss), {k: float(v) for k, v in parts.items()})
