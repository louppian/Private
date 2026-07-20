"""
scorer.py — PAFE(V3 GUI Hybrid) 의 공통 계약(ScorerBase) wrapper.

원본 PAFE.py 는 수정하지 않는다. 가중치 키는 self.net(=Hybrid) 기준으로
저장/로드되어 기존 체크포인트(inha_all_weights_57.pth 등)와 호환된다.

  입력  : 로더 기준 흑백 1ch — wrapper 가 내부에서 3ch 로 복제(원본 GUI 규약).
          마스크 미사용.
  출력  : {"logits": [B,4,C]}  순서 [RT, LT, RB, LB]
          원본 forward 는 softmax 확률을 내므로 log 를 취해 logits 로 통일한다.
          (softmax(log p) = p 이므로 손실/지표에서 등가)
  손실  : BrixiaLoss(alpha=1.0) = 순수 CE
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):          # 자기 폴더 + Model/ 루트
    if p not in sys.path:
        sys.path.insert(0, p)

from PAFE import build_v3_model                      # noqa: E402
from common import CLASSES, BrixiaLoss, ScorerBase   # noqa: E402


class PAFEScorer(ScorerBase):
    name = "pafe"
    in_channels = 1                                  # 3ch 복제는 내부 처리
    needs_mask = False
    freeze_stage = True           # lr 은 공통 고정값(헤드 1e-4 / 백본 1e-5) 상속

    def __init__(self, classes=CLASSES, kind="hybrid", backbone="resnet34",
                 weights_path=None):
        super().__init__()
        self.net = build_v3_model(kind=kind, backbone=backbone,
                                  num_classes=classes, num_regions=4,
                                  weights_path=weights_path)
        self.criterion = BrixiaLoss(alpha=1.0, classes=classes)

    @property
    def backbone(self):
        return self.net.backbone

    def forward(self, img, mask=None, target=None):
        if img.size(1) == 1:
            img = img.repeat(1, 3, 1, 1)             # 흑백 3복제 (원본 GUI 규약)
        probs = self.net(img)                        # softmax 확률 [B,4,C]
        return {"logits": probs.clamp_min(1e-8).log()}

    def compute_loss(self, out, target, extras=None):
        return self.criterion(out["logits"], target)


def build_scorer(classes=CLASSES, **kw):
    return PAFEScorer(classes=classes, **kw)


if __name__ == "__main__":
    import torch
    s = build_scorer()
    img = torch.randn(2, 1, 512, 512)
    out = s(img)
    loss, parts = s.compute_loss(out, torch.randint(0, 5, (2, 4)))
    print("logits :", tuple(out["logits"].shape))
    print("loss   :", float(loss), {k: float(v) for k, v in parts.items()})
